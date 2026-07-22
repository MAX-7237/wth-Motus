#!/bin/bash
set -euo pipefail

cd /mnt/gyc/wth-Motus

PYTHON="${PYTHON:-/mnt/gyc/miniconda3/envs/motus/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-configs/action_following_lerobot_rot6d20_clean_only_32a32f.yaml}"
CKPT="${CKPT:-}"
IMAGE="${IMAGE:-${1:-}}"
INSTRUCTION="${INSTRUCTION:-${2:-}}"
OUT_DIR="${OUT_DIR:-}"
WAN_PATH="${WAN_PATH:-/mnt/gyc_ckp/Wan2.2-TI2V-5B}"
DEVICE="${DEVICE:-cuda}"

if [ -z "$IMAGE" ] || [ -z "$INSTRUCTION" ]; then
    echo "Usage: IMAGE=/path/to/input.png INSTRUCTION='task text' bash scripts/infer_clean.sh"
    echo "   or: bash scripts/infer_clean.sh /path/to/input.png 'task text'"
    exit 1
fi

args=(
    scripts/baseline_video_infer.py
    --variant clean
    --config "$CONFIG_FILE"
    --image "$IMAGE"
    --instruction "$INSTRUCTION"
    --wan_path "$WAN_PATH"
    --device "$DEVICE"
)

if [ -n "$CKPT" ]; then
    args+=(--ckpt "$CKPT")
fi

if [ -n "$OUT_DIR" ]; then
    args+=(--out_dir "$OUT_DIR")
fi

"$PYTHON" "${args[@]}"
