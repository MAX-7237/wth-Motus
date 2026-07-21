#!/usr/bin/env python3
"""Run single-sample Motus baseline inference and save video/action outputs."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.utils.norm import denormalize_actions, load_action_state_normalization_stats, normalize_actions
from utils.vlm_utils import preprocess_vlm_messages

DEFAULT_CONFIGS = {
    "clean": PROJECT_ROOT / "configs" / "action_following_lerobot_rot6d20_clean_only.yaml",
    "mix": PROJECT_ROOT / "configs" / "action_following_lerobot_rot6d20_mix_4_1_1_1_1.yaml",
}


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_config(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    return DEFAULT_CONFIGS[args.variant]


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"baseline_{args.variant}_{timestamp}"


def _resolve_checkpoint(args: argparse.Namespace, cfg: Dict[str, Any], config_path: Path) -> Path:
    if args.ckpt:
        return Path(args.ckpt)

    base_dir = Path(cfg.get("system", {}).get("checkpoint_dir", "./checkpoints"))
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    dataset_name = config_path.stem
    run_name = cfg.get("logging", {}).get("run_name", dataset_name)
    run_dir = base_dir / dataset_name / run_name
    candidates = sorted(
        [p for p in run_dir.glob("checkpoint_step_*") if p.is_dir()],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint was provided and no checkpoint_step_* directory was found under "
            f"{run_dir}. Pass --ckpt explicitly."
        )
    return candidates[-1]


def _dtype_from_config(name: str) -> torch.dtype:
    return torch.bfloat16 if str(name).lower() in {"bf16", "bfloat16"} else torch.float16


def _build_model_config(cfg: Dict[str, Any]):
    from models.motus import MotusConfig

    common = cfg["common"]
    model = cfg["model"]
    return MotusConfig(
        wan_checkpoint_path=model["wan"]["checkpoint_path"],
        vae_path=model["wan"]["vae_path"],
        wan_config_path=model["wan"]["config_path"],
        video_precision=model["wan"].get("precision", "bfloat16"),
        vlm_checkpoint_path=model["vlm"]["checkpoint_path"],
        und_expert_hidden_size=model.get("und_expert", {}).get("hidden_size", 512),
        und_expert_ffn_dim_multiplier=model.get("und_expert", {}).get("ffn_dim_multiplier", 4),
        und_expert_norm_eps=model.get("und_expert", {}).get("norm_eps", 1e-5),
        vlm_adapter_input_dim=model.get("und_expert", {}).get("vlm", {}).get("input_dim", 2048),
        vlm_adapter_projector_type=model.get("und_expert", {}).get("vlm", {}).get("projector_type", "mlp3x_silu"),
        num_layers=30,
        action_state_dim=int(common["state_dim"]),
        action_dim=int(common["action_dim"]),
        action_expert_dim=model["action_expert"]["hidden_size"],
        action_expert_ffn_dim_multiplier=model["action_expert"]["ffn_dim_multiplier"],
        action_expert_norm_eps=model["action_expert"].get("norm_eps", 1e-6),
        global_downsample_rate=int(common["global_downsample_rate"]),
        video_action_freq_ratio=int(common["video_action_freq_ratio"]),
        num_video_frames=int(common["num_video_frames"]),
        video_height=int(common["video_height"]),
        video_width=int(common["video_width"]),
        batch_size=1,
        video_loss_weight=model["loss_weights"]["video_loss_weight"],
        action_loss_weight=model["loss_weights"]["action_loss_weight"],
        training_mode=cfg.get("training_mode", "finetune"),
        load_pretrained_backbones=False,
    )


def _find_checkpoint_file(path: str | Path) -> Path:
    ckpt = Path(path)
    if ckpt.is_file():
        return ckpt
    candidates = [
        ckpt / "mp_rank_00_model_states.pt",
        ckpt / "pytorch_model" / "mp_rank_00_model_states.pt",
        ckpt / "model.safetensors",
    ]
    candidates.extend(sorted(ckpt.glob("*.safetensors")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"No supported checkpoint file found under {ckpt}. Tried:\n{tried}")


def _load_checkpoint(model: Any, path: str | Path) -> Tuple[int, int, Path]:
    ckpt_file = _find_checkpoint_file(path)
    if ckpt_file.suffix == ".safetensors":
        state_dict = load_safetensors(str(ckpt_file), device="cpu")
    else:
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        state_dict = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    return len(incompatible.missing_keys), len(incompatible.unexpected_keys), ckpt_file


def _load_image(path: str | Path, height: int, width: int) -> Tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return image, tensor


def _load_state(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    state_dim = int(cfg["common"]["state_dim"])
    if args.state_json:
        with open(args.state_json, "r", encoding="utf-8") as f:
            raw_state = json.load(f)
    elif args.state:
        raw_state = [float(x) for x in args.state.split(",") if x.strip()]
    else:
        raw_state = [0.0] * state_dim
    if len(raw_state) != state_dim:
        raise ValueError(f"State dim mismatch: expected {state_dim}, got {len(raw_state)}")

    raw = torch.tensor(raw_state, dtype=torch.float32)
    dataset_cfg = cfg.get("dataset", {}).get("params", {})
    if not dataset_cfg.get("normalize_action_state", False):
        return raw, None

    stats = load_action_state_normalization_stats(
        dataset_cfg["normalization_stats_path"],
        dataset_cfg["normalization_dataset_key"],
    )
    action_min, action_max, state_min, state_max = stats
    if any(x is None for x in stats):
        raise RuntimeError("Failed to load action/state normalization stats")
    normalized = normalize_actions(raw.unsqueeze(0), state_min, state_max).squeeze(0)
    return normalized, raw


def _as_single_t5_embedding(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected one T5 embedding shaped [seq, dim], got {tuple(tensor.shape)}")
    return tensor


def _encode_t5(args: argparse.Namespace, cfg: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    if args.t5_embeds:
        loaded = torch.load(args.t5_embeds, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            return [_as_single_t5_embedding(loaded)]
        if isinstance(loaded, list):
            return [_as_single_t5_embedding(x) for x in loaded]
        raise TypeError(f"Unsupported T5 embed object: {type(loaded)}")

    from wan.modules.t5 import T5EncoderModel

    wan_root = Path(args.wan_path or cfg["model"]["wan"]["checkpoint_path"])
    if not (wan_root / "models_t5_umt5-xxl-enc-bf16.pth").is_file() and (wan_root / "Wan2.2-TI2V-5B").is_dir():
        wan_root = wan_root / "Wan2.2-TI2V-5B"
    ckpt = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = wan_root / "google" / "umt5-xxl"
    encoder = T5EncoderModel(
        text_len=int(args.text_len),
        dtype=dtype,
        device=str(device),
        checkpoint_path=str(ckpt),
        tokenizer_path=str(tokenizer),
    )
    out = encoder([args.instruction], device=str(device))
    if isinstance(out, torch.Tensor):
        return [_as_single_t5_embedding(out)]
    return [_as_single_t5_embedding(x) for x in out]


def _frames_bcthw_to_tchw(frames: torch.Tensor) -> torch.Tensor:
    if frames.dim() != 5 or frames.shape[0] != 1:
        raise ValueError(f"Expected frames [1,C,T,H,W] or [1,T,C,H,W], got {tuple(frames.shape)}")
    if frames.shape[1] == 3:
        return frames[0].permute(1, 0, 2, 3).contiguous()
    if frames.shape[2] == 3:
        return frames[0].contiguous()
    raise ValueError(f"Cannot infer channel axis from predicted frames shape {tuple(frames.shape)}")


def _tensor_frame_to_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    return (frame_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _save_outputs(
    out_dir: Path,
    instruction: str,
    input_image: Image.Image,
    first_frame: torch.Tensor,
    pred_frames_tchw: torch.Tensor,
    pred_actions_norm: torch.Tensor,
    pred_actions_raw: Optional[torch.Tensor],
    fps: int,
    manifest: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_image.save(out_dir / "input.png")
    with open(out_dir / "instruction.txt", "w", encoding="utf-8") as f:
        f.write(instruction.strip() + "\n")

    video_frames = [_tensor_frame_to_uint8(first_frame[0])]
    video_frames.extend(_tensor_frame_to_uint8(pred_frames_tchw[i]) for i in range(pred_frames_tchw.shape[0]))
    imageio.mimsave(out_dir / "pred_video.mp4", video_frames, fps=fps)

    grid = np.concatenate(video_frames, axis=1)
    Image.fromarray(grid).save(out_dir / "pred_grid.png")

    np.save(out_dir / "pred_actions_normalized.npy", pred_actions_norm.detach().cpu().float().numpy())
    if pred_actions_raw is not None:
        np.save(out_dir / "pred_actions_denormalized.npy", pred_actions_raw.detach().cpu().float().numpy())

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Motus baseline video generation inference")
    parser.add_argument("--variant", choices=sorted(DEFAULT_CONFIGS), default="clean", help="Default config/checkpoint family")
    parser.add_argument("--config", default=None, help="Training YAML used for the checkpoint")
    parser.add_argument("--ckpt", default=None, help="Checkpoint file or checkpoint_step_* directory; defaults to latest for the selected config")
    parser.add_argument("--image", required=True, help="Conditioning RGB image")
    parser.add_argument("--instruction", required=True, help="Text instruction")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--state", default=None, help="Comma-separated raw state values; defaults to zeros")
    parser.add_argument("--state_json", default=None, help="JSON list of raw state values")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan_path", default="/mnt/gyc_ckp/Wan2.2-TI2V-5B", help="WAN root")
    parser.add_argument("--t5_embeds", default=None, help="Optional precomputed T5 embedding .pt")
    parser.add_argument("--text_len", type=int, default=512)
    args = parser.parse_args()

    config_path = _resolve_config(args)
    cfg = _read_yaml(config_path)
    ckpt_path = _resolve_checkpoint(args, cfg, config_path)
    out_dir = _resolve_output_dir(args)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    dtype = _dtype_from_config(cfg["model"]["wan"].get("precision", "bfloat16"))

    from models.motus import Motus

    model = Motus(_build_model_config(cfg)).to(device).eval()
    missing, unexpected, ckpt_file = _load_checkpoint(model, ckpt_path)

    height = int(cfg["common"]["video_height"])
    width = int(cfg["common"]["video_width"])
    input_image, first_frame_cpu = _load_image(args.image, height, width)

    state_norm_cpu, state_raw_cpu = _load_state(args, cfg)
    language_embeddings = _encode_t5(args, cfg, device, dtype)
    processor = AutoProcessor.from_pretrained(cfg["model"]["vlm"]["checkpoint_path"], trust_remote_code=True)
    vlm_inputs = preprocess_vlm_messages(args.instruction, input_image, processor)
    vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vlm_inputs.items()}

    num_steps = args.num_inference_steps or int(cfg["model"]["inference"]["num_inference_timesteps"])
    with torch.no_grad():
        pred_frames, pred_actions_norm = model.inference_step(
            first_frame=first_frame_cpu.to(device),
            state=state_norm_cpu.unsqueeze(0).to(device),
            num_inference_steps=num_steps,
            language_embeddings=[emb.to(device) for emb in language_embeddings],
            vlm_inputs=vlm_inputs,
        )

    pred_frames_tchw = _frames_bcthw_to_tchw(pred_frames)
    pred_actions_norm_cpu = pred_actions_norm[0].detach().cpu().float()
    pred_actions_raw = None
    dataset_cfg = cfg.get("dataset", {}).get("params", {})
    if dataset_cfg.get("normalize_action_state", False):
        action_min, action_max, _state_min, _state_max = load_action_state_normalization_stats(
            dataset_cfg["normalization_stats_path"],
            dataset_cfg["normalization_dataset_key"],
        )
        pred_actions_raw = denormalize_actions(pred_actions_norm_cpu, action_min, action_max)

    manifest = {
        "config": str(config_path.resolve()),
        "checkpoint": str(ckpt_file.resolve()),
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
        "image": str(Path(args.image).resolve()),
        "instruction": args.instruction,
        "state_is_raw_input": True,
        "state_raw": state_raw_cpu.tolist() if state_raw_cpu is not None else state_norm_cpu.tolist(),
        "state_normalized": state_norm_cpu.tolist(),
        "normalize_action_state": bool(dataset_cfg.get("normalize_action_state", False)),
        "normalization_dataset_key": dataset_cfg.get("normalization_dataset_key"),
        "num_inference_steps": num_steps,
        "outputs": {
            "video": "pred_video.mp4",
            "grid": "pred_grid.png",
            "actions_normalized": "pred_actions_normalized.npy",
            "actions_denormalized": "pred_actions_denormalized.npy" if pred_actions_raw is not None else None,
        },
        "predicted_frames_shape_tchw": list(pred_frames_tchw.shape),
        "predicted_actions_shape": list(pred_actions_norm_cpu.shape),
    }
    _save_outputs(
        out_dir,
        args.instruction,
        input_image,
        first_frame_cpu,
        pred_frames_tchw,
        pred_actions_norm_cpu,
        pred_actions_raw,
        args.fps,
        manifest,
    )
    print(f"Saved baseline inference outputs to {out_dir}")
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
