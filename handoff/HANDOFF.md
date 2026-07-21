# Motus Handoff

Last updated: 2026-07-21

This note is the quick handoff for the current Rot6D20 Motus runs in this repo.
It focuses on where to find checkpoints, how the models were trained, how
normalization is applied, how to run the minimum single-sample inference script,
and how the multi-camera image is arranged.

## 1. Checkpoints

Latest verified local training checkpoints:

| Variant | Config | Run name | Latest verified step | Current source path |
| --- | --- | --- | --- | --- |
| clean-only | `configs/action_following_lerobot_rot6d20_clean_only.yaml` | `action_following_rot6d20_clean_only_stage3` | `checkpoint_step_30000` | `/mnt/gyc_ckp/wth-Motus/checkpoints/action_following_lerobot_rot6d20_clean_only/action_following_rot6d20_clean_only_stage3/checkpoint_step_30000` |
| mix 4:1:1:1:1 | `configs/action_following_lerobot_rot6d20_mix_4_1_1_1_1.yaml` | `action_following_rot6d20_mix_4_1_1_1_1_stage3` | `checkpoint_step_30000` | `/mnt/gyc_ckp/wth-Motus/checkpoints/action_following_lerobot_rot6d20_mix_4_1_1_1_1/action_following_rot6d20_mix_4_1_1_1_1_stage3/checkpoint_step_30000` |

Each `checkpoint_step_30000` directory is about 96 GB and contains
`pytorch_model/mp_rank_00_model_states.pt` plus ZeRO optimizer shards.

Public-disk handoff paths to use after mirroring/copying the verified source
checkpoints:

| Variant | Public ckpt directory |
| --- | --- |
| clean-only | `/mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/action_following_lerobot_rot6d20_clean_only/action_following_rot6d20_clean_only_stage3/checkpoint_step_30000` |
| mix 4:1:1:1:1 | `/mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/action_following_lerobot_rot6d20_mix_4_1_1_1_1/action_following_rot6d20_mix_4_1_1_1_1_stage3/checkpoint_step_30000` |

Note: during this handoff update, a bounded search did not find an existing
Motus-specific checkpoint copy under `/mnt/public_ckp`. The paths above are
the intended public-disk locations to reference once the 96 GB checkpoint
directories are mirrored from the verified source paths.

Suggested one-time mirror commands, if the public paths are not populated yet:

```bash
mkdir -p /mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/action_following_lerobot_rot6d20_clean_only/action_following_rot6d20_clean_only_stage3
rsync -a --info=progress2 \
  /mnt/gyc_ckp/wth-Motus/checkpoints/action_following_lerobot_rot6d20_clean_only/action_following_rot6d20_clean_only_stage3/checkpoint_step_30000 \
  /mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/action_following_lerobot_rot6d20_clean_only/action_following_rot6d20_clean_only_stage3/

mkdir -p /mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/action_following_lerobot_rot6d20_mix_4_1_1_1_1/action_following_rot6d20_mix_4_1_1_1_1_stage3
rsync -a --info=progress2 \
  /mnt/gyc_ckp/wth-Motus/checkpoints/action_following_lerobot_rot6d20_mix_4_1_1_1_1/action_following_rot6d20_mix_4_1_1_1_1_stage3/checkpoint_step_30000 \
  /mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/action_following_lerobot_rot6d20_mix_4_1_1_1_1/action_following_rot6d20_mix_4_1_1_1_1_stage3/
```

## 2. Training configuration

Shared core settings:

- Model is fine-tuned from `/mnt/gyc/wth-Motus/pretrained_models/Motus`.
- WAN backbone/config/VAE are loaded from `/mnt/gyc_ckp/Wan2.2-TI2V-5B`.
- VLM checkpoint is `./pretrained_models/Qwen3-VL-2B-Instruct`, frozen, bf16.
- Action/state dimensions are both 20; the runs use Rot6D20 action following.
- Video setting is 8 frames, resized/padded to `384 x 320`.
- `video_action_freq_ratio: 6`; action chunk size is therefore `8 * 6 = 48`.
- Training uses `batch_size: 8`, `gradient_accumulation_steps: 1`,
  `max_steps: 40000`, `learning_rate: 5e-5`, `weight_decay: 0.01`,
  `warmup_steps: 200`, `grad_clip_norm: 0.5`, AMP enabled.
- Checkpoints are written every 10,000 steps; validation interval is 500 steps.
- Default checkpoint root in the YAMLs is `/mnt/gyc_ckp/wth-Motus/checkpoints`.

Clean-only run:

- Config: `configs/action_following_lerobot_rot6d20_clean_only.yaml`
- Dataset root:
  `/mnt/public_ckp/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train/demo_clean_zed2i_visible`
- Normalization key: `action_following_rot6d20_clean_only`
- Run name: `action_following_rot6d20_clean_only_stage3`
- Default launch:

```bash
cd /mnt/gyc/wth-Motus
bash scripts/train.sh
```

Mix 4:1:1:1:1 run:

- Config: `configs/action_following_lerobot_rot6d20_mix_4_1_1_1_1.yaml`
- Dataset root:
  `/mnt/public_ckp/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train`
- Normalization key: `action_following_rot6d20_mix_4_1_1_1_1`
- Run name: `action_following_rot6d20_mix_4_1_1_1_1_stage3`
- Sampling groups:
  - `clean`: weight 4, root `demo_clean_zed2i_visible`
  - `counterfactual_replay`: weight 1, root `counterfactual_replay/tasks`
  - `exploration`: weight 1, root `exploration/tasks`
  - `perturbed`: weight 1, roots `perturbed/raw/tasks`,
    `perturbed/pca/tasks`
  - `random_feasible`: weight 1, roots `random_feasible/uniform/tasks`,
    `random_feasible/weighted/tasks`
- Launch:

```bash
cd /mnt/gyc/wth-Motus
TASK=action_following_rot6d20_mix_4_1_1_1_1_stage3 \
CONFIG_FILE=configs/action_following_lerobot_rot6d20_mix_4_1_1_1_1.yaml \
bash scripts/train.sh
```

`scripts/train.sh` defaults to 8 processes via
`NPROC_PER_NODE=8` and `/mnt/gyc/miniconda3/envs/motus/bin/torchrun`.

## 3. Normalization

Normalization is enabled in both current Rot6D20 configs:

- `normalize_action_state: true`
- Stats file:
  `/mnt/gyc/wth-Motus/data/utils/action_following_rot6d20_stats.json`
- Keys:
  - clean-only: `action_following_rot6d20_clean_only`
  - mix: `action_following_rot6d20_mix_4_1_1_1_1`

The implementation in `data/utils/norm.py` uses min-max scaling to `[0, 1]`:

```text
normalized = (x - min) / (max - min)
denormalized = normalized * (max - min) + min
```

Both `action` and `observation.state` have explicit 20D min/max stats in the
JSON. During dataset loading, `initial_state` and `action_sequence` are
normalized before being returned to the model. During baseline inference, the
input state is treated as raw state, normalized before the model call, and
predicted actions are saved in both normalized and denormalized form when stats
are available.

If no state is passed to the baseline inference script, it uses a raw zero
20D state and then applies the configured state normalization. For real robot
or RobotWin evaluation, pass the actual 20D raw state instead of relying on the
zero default.

## 4. Minimum inference example

Two minimal wrappers are available:

- Clean-only: `scripts/infer_clean.sh`
- Mix 4:1:1:1:1: `scripts/infer_mix.sh`

They call `scripts/baseline_video_infer.py`, load the matching YAML config,
load the selected checkpoint, run one single-image instruction-conditioned
inference, and save outputs under `OUT_DIR` or
`outputs/baseline_<variant>_<timestamp>`.

Clean-only example:

```bash
cd /mnt/gyc/wth-Motus

IMAGE=/path/to/three_view_t_shape.png \
INSTRUCTION='pick up the object and place it into the target container' \
CKPT=/mnt/gyc_ckp/wth-Motus/checkpoints/action_following_lerobot_rot6d20_clean_only/action_following_rot6d20_clean_only_stage3/checkpoint_step_30000 \
OUT_DIR=/tmp/motus_clean_minimal \
bash scripts/infer_clean.sh
```

Mix example:

```bash
cd /mnt/gyc/wth-Motus

IMAGE=/path/to/three_view_t_shape.png \
INSTRUCTION='pick up the object and place it into the target container' \
CKPT=/mnt/gyc_ckp/wth-Motus/checkpoints/action_following_lerobot_rot6d20_mix_4_1_1_1_1/action_following_rot6d20_mix_4_1_1_1_1_stage3/checkpoint_step_30000 \
OUT_DIR=/tmp/motus_mix_minimal \
bash scripts/infer_mix.sh
```

If public checkpoint mirrors are populated, replace `CKPT=` with the matching
`/mnt/public_ckp/cscsx_projects/AF3/Motus/wth-Motus/checkpoints/...` path
from Section 1.

Expected generated files in `OUT_DIR`:

| File | Meaning |
| --- | --- |
| `input.png` | Resized input conditioning frame used by the model |
| `instruction.txt` | Text instruction for the run |
| `pred_video.mp4` | Generated rollout video; includes the conditioning frame followed by predicted frames |
| `pred_grid.png` | Horizontal image grid of the conditioning frame plus generated frames |
| `pred_actions_normalized.npy` | Predicted action chunk in normalized `[0, 1]` space |
| `pred_actions_denormalized.npy` | Predicted action chunk converted back to raw Rot6D20 scale |
| `manifest.json` | Config, checkpoint file, input paths, normalization key, output filenames, frame/action shapes |

The checkpoint argument can be either a `checkpoint_step_*` directory or the
checkpoint file itself. For a directory, the loader resolves
`pytorch_model/mp_rank_00_model_states.pt`.

## 5. Camera arrangement

The expected image is a three-view T-shape concatenation:

```text
+-------------------------------+
|                               |
|        cam_high / head        |
|      fixed/rear overview      |
|                               |
+---------------+---------------+
| left wrist    | right wrist   |
| camera        | camera        |
+---------------+---------------+
```

Dataset camera keys:

- `observation.images.cam_concatenated`: preferred if already present.
- Otherwise the loader stitches:
  - `observation.images.cam_high`
  - `observation.images.cam_left_wrist`
  - `observation.images.cam_right_wrist`

Stitching logic:

- Put `cam_high` on the top row at full target width.
- Put `cam_left_wrist` on the bottom-left half.
- Put `cam_right_wrist` on the bottom-right half.
- Wrist views are resized to fit the bottom row.
- The resulting concatenated frame is resized with padding to the configured
  model size `384 x 320`, preserving aspect ratio.

For manual single-sample inference, provide `IMAGE` as this already
concatenated T-shape RGB image. If you start from three separate camera images,
build the same layout before calling `scripts/infer_clean.sh` or
`scripts/infer_mix.sh`.

RobotWin/real-world notes:

- RobotWin observation names map to `head_camera`, `left_camera`,
  `right_camera`.
- The prompt prefix in RobotWin describes the views as a fixed rear camera plus
  movable left/right arm cameras; keep that convention for consistency.
- Do not feed a single unstitched camera image unless you intentionally train or
  evaluate a single-view variant.

