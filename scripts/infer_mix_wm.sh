#!/bin/bash
set -euo pipefail

cd /mnt/gyc/wth-Motus

PYTHON="${PYTHON:-/mnt/gyc/miniconda3/envs/motus/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-configs/action_following_lerobot_rot6d20_mix_4_1_1_1_1.yaml}"
CKPT="${CKPT:-}"
IMAGE="${IMAGE:-${1:-}}"
INSTRUCTION="${INSTRUCTION:-${2:-}}"
ACTIONS="${ACTIONS:-${3:-}}"
OUT_DIR="${OUT_DIR:-}"
WAN_PATH="${WAN_PATH:-/mnt/gyc_ckp/Wan2.2-TI2V-5B}"
DEVICE="${DEVICE:-cuda}"
ACTIONS_ARE_RAW="${ACTIONS_ARE_RAW:-0}"
STATE="${STATE:-}"
STATE_JSON="${STATE_JSON:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
FPS="${FPS:-}"
T5_EMBEDS="${T5_EMBEDS:-}"
TEXT_LEN="${TEXT_LEN:-}"

if [ -z "$IMAGE" ] || [ -z "$INSTRUCTION" ] || [ -z "$ACTIONS" ]; then
    echo "Usage: IMAGE=/path/to/input.png INSTRUCTION='task text' ACTIONS=/path/to/actions.npy bash scripts/infer_mix_wm.sh"
    echo "   or: bash scripts/infer_mix_wm.sh /path/to/input.png 'task text' /path/to/actions.npy"
    echo "Notes: ACTIONS should be normalized [48,20] or [1,48,20] by default; set ACTIONS_ARE_RAW=1 for raw Rot6D20."
    exit 1
fi

args=(
    scripts/baseline_video_infer.py
    --mode wm
    --variant mix
    --config "$CONFIG_FILE"
    --image "$IMAGE"
    --instruction "$INSTRUCTION"
    --actions "$ACTIONS"
    --wan_path "$WAN_PATH"
    --device "$DEVICE"
)

case "${ACTIONS_ARE_RAW,,}" in
    1|true|yes|y)
        args+=(--actions_are_raw)
        ;;
esac

if [ -n "$CKPT" ]; then
    args+=(--ckpt "$CKPT")
fi

if [ -n "$OUT_DIR" ]; then
    args+=(--out_dir "$OUT_DIR")
fi

if [ -n "$STATE" ]; then
    args+=(--state "$STATE")
fi

if [ -n "$STATE_JSON" ]; then
    args+=(--state_json "$STATE_JSON")
fi

if [ -n "$NUM_INFERENCE_STEPS" ]; then
    args+=(--num_inference_steps "$NUM_INFERENCE_STEPS")
fi

if [ -n "$FPS" ]; then
    args+=(--fps "$FPS")
fi

if [ -n "$T5_EMBEDS" ]; then
    args+=(--t5_embeds "$T5_EMBEDS")
fi

if [ -n "$TEXT_LEN" ]; then
    args+=(--text_len "$TEXT_LEN")
fi

"$PYTHON" "${args[@]}"
