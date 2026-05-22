#!/usr/bin/env bash
# Qwen3.5-9B SFT for JarvisVLA — Phase 1 of REIMPLEMENTATION_PLAN.md.
#
# Production config (validated by vla_qwen3_5_9b_sft_dryrun.sh, 2026-05-22):
#   - DeepSpeed ZeRO-2 with CPU optimizer offload (s2_offload.json) — needed because
#     ZeRO-3 + grad checkpointing + frozen vision tower hits a recompute shape
#     mismatch. CPU offload frees the ~11 GiB optimizer-step headroom on each GPU.
#   - Flash-attention-2 for full-attention layers + flash-linear-attention/causal-conv1d
#     for the gated DeltaNet layers. Without these the run is ~20x slower.
#   - Visual encoder + adapter frozen (matches the Qwen2-VL 7B recipe; A/B unfreeze later).
#
# Speedup: vLLM serves with --speculative-config qwen3_next_mtp at inference time.
# No training-side MTP wiring required — Qwen3.5-9B does not expose separately
# trainable MTP heads (see REIMPLEMENTATION_PLAN.md Step 7).
set -euo pipefail

export HF_HOME="${HF_HOME:-/ephemeral/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/ephemeral/.hf_cache/datasets}"
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):${PYTHONPATH:-}"

epoch="${EPOCH:-1}"
batch="${BATCH:-1}"
gradient_accumulation_steps="${GRAD_ACCUM:-1}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2}"
training_port="${TRAINING_PORT:-24001}"

# Dataset budget. Set DATASET_P (0..1) to scope down for shorter runs, or MAX_STEPS to
# hard-cap step count.  ~6h on 3 A100 ≈ 3000 steps at the current 6.7s/step.
dataset_p="${DATASET_P:-1.0}"
max_steps_arg=""
if [[ -n "${MAX_STEPS:-}" ]]; then
    max_steps_arg="--max_steps ${MAX_STEPS}"
fi

dataset_name="${DATASET_NAME:-/ephemeral/datasets/jarvisvla-chunk4}"
base_model_path="${BASE_MODEL_PATH:-/ephemeral/models/Qwen3.5-9B}"
version="mc-vla-qwen3-5-9b"
run_tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="$version-${run_tag}-e${epoch}-b${batch}-a${gradient_accumulation_steps}-p${dataset_p}"
output_dir="${OUTPUT_DIR:-/ephemeral/checkpoints/${WANDB_NAME}}"

# Vision-token budget: aggressive cap to keep ViT prefill from blowing the latency budget.
# 256 * 28 * 28 = 200,704 ≈ 256 vision tokens per frame. A/B at 512 if eval shows
# inventory-text legibility regressions.
min_pixels=$((4 * 28 * 28))
max_pixels=$((256 * 28 * 28))

deepspeed --include "localhost:$cuda_visible_devices" --master_port="$training_port" \
    jarvisvla/train/train.py \
    --deepspeed configs/deepspeed_config_s2_offload.json \
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
