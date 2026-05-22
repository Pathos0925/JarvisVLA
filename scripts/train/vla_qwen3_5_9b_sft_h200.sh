#!/usr/bin/env bash
# Qwen3.5-9B SFT for JarvisVLA — single-node H200 variant (no DeepSpeed).
#
# Per handoff §B200 simplification: 144 GB per H200 fits 9.41B model + Adam states +
# activations at bf16, so we drop ZeRO-2 and CPU optimizer offload. For 2× H200, torchrun
# DDP across both GPUs is ~2× throughput over single-process. Keep BATCH=4 unless this OOMs,
# then drop to 2 + GRAD_ACCUM=2 to preserve effective batch size.
#
# Hyperparams are identical to the production DeepSpeed script (vla_qwen3_5_9b_sft.sh):
# the only deltas are the launcher, paths, and the absence of --deepspeed.
#
# R2 (Cloudflare) checkpoint upload is opt-in via env vars; absent → disabled silently.
#   R2_BUCKET            (required)
#   R2_ACCOUNT_ID        (required, used to build the endpoint URL)
#   R2_ACCESS_KEY_ID     (required)
#   R2_SECRET_ACCESS_KEY (required)
#   R2_PREFIX            (optional, default: basename of output_dir)
#   R2_MAX_WORKERS       (optional, default: 4)
# Uploads happen async on rank-0 only; final wait_all() blocks process exit until done.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/hf_cache/datasets}"
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):${PYTHONPATH:-}"

epoch="${EPOCH:-1}"
batch="${BATCH:-4}"
gradient_accumulation_steps="${GRAD_ACCUM:-1}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1}"
training_port="${TRAINING_PORT:-24001}"

dataset_p="${DATASET_P:-1.0}"
max_steps_arg=""
if [[ -n "${MAX_STEPS:-}" ]]; then
    max_steps_arg="--max_steps ${MAX_STEPS}"
fi

dataset_name="${DATASET_NAME:-/workspace/datasets/jarvisvla-chunk4}"
base_model_path="${BASE_MODEL_PATH:-/workspace/models/Qwen3.5-9B}"
version="mc-vla-qwen3-5-9b-h200"
run_tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="$version-${run_tag}-e${epoch}-b${batch}-a${gradient_accumulation_steps}-p${dataset_p}"
output_dir="${OUTPUT_DIR:-/workspace/checkpoints/${WANDB_NAME}}"

min_pixels=$((4 * 28 * 28))
max_pixels=$((256 * 28 * 28))

nproc=$(echo "$cuda_visible_devices" | awk -F',' '{print NF}')

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
torchrun --nproc-per-node="$nproc" --master-port="$training_port" \
    jarvisvla/train/train.py \
    --backbone qwen3_5 \
    --attn_implementation flash_attention_2 \
    --dataset_name "$dataset_name" \
    --dataset_p "$dataset_p" \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --seed 43 \
    --model_name_or_path "$base_model_path" \
    --report_to "none" \
    --learning_rate 3e-6 \
    --max_grad_norm 1 \
    --weight_decay 0. \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    $max_steps_arg \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --output_dir "$output_dir" \
    --run_name "$WANDB_NAME" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 1024 \
    --collator_type "VLAMultimodalChatDataCollatorforVLM" \
    --fix_visual_encoder True \
    --fix_visual_adapter True \
    --min_pixels $min_pixels \
    --max_pixels $max_pixels
