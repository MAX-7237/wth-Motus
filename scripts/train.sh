#!/bin/bash
# Define your env settings here 
# e.g., nccl, network, proxy, etc.

TASK="${TASK:-action_following_rot6d20_clean_only_stage3}"  # Define your task name here
CONFIG_FILE="${CONFIG_FILE:-configs/action_following_lerobot_rot6d20_clean_only.yaml}"  # Define your dataset config path here
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
TORCHRUN="${TORCHRUN:-/mnt/gyc/miniconda3/envs/motus/bin/torchrun}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/gyc/wth-Motus/.cache/hf_datasets}"
export HF_HOME="${HF_HOME:-/mnt/gyc/wth-Motus/.cache/huggingface}"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}" # Define your output directory here

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Single-node training with torchrun
"${TORCHRUN}" \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config $CONFIG_FILE \
    --run_name $TASK \
    --report_to tensorboard
