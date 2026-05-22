#!/usr/bin/env bash
# Multi-GPU dry-run SFT for Qwen3.5-9B on 1% of the chunked train set.
#
# Purpose: validate the full train.py pipeline end-to-end on real data — catches data
# collator / chunked-target / masking issues that the smoke test (single forward pass)
# can't see. NOT for producing a real checkpoint; save_strategy=no.
#
# Expected wall time: ~15-30 min on 3× A100 80GB with sdpa attention (no flash-attn).
set -euo pipefail

export HF_HOME="${HF_HOME:-/ephemeral/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/ephemeral/.hf_cache/datasets}"
# Make `import jarvisvla` work without `pip install -e .` (setup.py uses deprecated pkg_resources).
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):${PYTHONPATH:-}"
# Extend NCCL watchdog: torch fallback for Qwen3.5-9B's linear attention is slow enough
# at seq_len=1024 that a per-step allreduce can blow the default 600s timeout. Production
# runs should install flash-linear-attention + causal-conv1d for the fast path.
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-3600000}"

epoch=1
batch=1
gradient_accumulation_steps=1
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2}"
training_port="${TRAINING_PORT:-24001}"

# Local chunked dataset produced by scripts/preprocess_chunked_actions.py.
dataset_name="${DATASET_NAME:-/ephemeral/datasets/jarvisvla-chunk4}"
base_model_path="${BASE_MODEL_PATH:-/ephemeral/models/Qwen3.5-9B}"
output_dir="${OUTPUT_DIR:-/ephemeral/checkpoints/qwen3_5_9b_dryrun}"

# Vision-token budget: aggressive cap (~256 vision tokens per frame) so prefill doesn't
# blow the per-step budget. A/B at 512 only if eval shows fidelity issues.
min_pixels=$((4 * 28 * 28))
max_pixels=$((256 * 28 * 28))

# Use sdpa explicitly — flash-attn isn't installed in this env, and the smoke test
# confirmed Qwen3.5-9B works with sdpa.
deepspeed --include "localhost:$cuda_visible_devices" --master_port="$training_port" \
    jarvisvla/train/train.py \
    --deepspeed configs/deepspeed_config_s2_offload.json \
    --backbone qwen3_5 \
    --attn_implementation flash_attention_2 \
    --dataset_name "$dataset_name" \
    --dataset_p 0.01 \
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
    --warmup_steps 50 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --max_steps 5 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --output_dir "$output_dir" \
    --logging_strategy "steps" \
    --logging_steps 1 \
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
