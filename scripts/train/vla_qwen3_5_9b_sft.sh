#!/usr/bin/env bash
# Qwen3.5-9B SFT for JarvisVLA — Phase 1 of REIMPLEMENTATION_PLAN.md.
#
# Prerequisites before running this end-to-end:
#   - Chunked-action SFT dataset re-rendered from CraftJarvis/JarvisVLA-Qwen2-VL-7B
#     (each assistant turn must be N actions, matching action_chunk_len at inference).
#   - vLLM version pinned that supports qwen3_next_mtp (see scripts/inference/serve_vllm.sh).
#   - Hybrid-attention smoke test passed (Week 1 of plan).
#
# Speedup: MTP loss is included by default via Qwen3.5-9B's native heads.
#          Verify nonzero grad norms on MTP head params after the first training step;
#          if zero, override Trainer.compute_loss to invoke the MTP loss explicitly.
set -euo pipefail

epoch=1
batch=4
gradient_accumulation_steps=2
card_number=1
cuda_visible_devices=0
node_number=1
total_number=$(($card_number * $node_number))

# Use the chunked dataset (Step 5). Until that preprocessor lands, point at the chunk-1
# baseline to validate the SFT pipeline end-to-end on the existing data.
dataset_name="${DATASET_NAME:-CraftJarvis/JarvisVLA-Qwen2-VL-7B}"
base_model_path="${BASE_MODEL_PATH:-Qwen/Qwen3.5-9B}"
version="mc-vla-qwen3-5-9b"
WANDB_NAME="$version-c$total_number-e$epoch-b$batch-a$gradient_accumulation_steps"
output_dir="${OUTPUT_DIR:-/public/JARVIS/checkpoints2/$WANDB_NAME}"

# Vision-token budget: aggressive cap to keep ViT prefill from blowing the latency budget.
# 256 * 28 * 28 = 200,704 ≈ 256 vision tokens per frame. A/B with 512 * 28 * 28 if
# inventory-text legibility regresses on eval.
min_pixels=$((4 * 28 * 28))
max_pixels=$((256 * 28 * 28))

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python jarvisvla/train/train.py \
    --backbone qwen3_5 \
    --dataset_name "$dataset_name" \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --seed 43 \
    --model_name_or_path "$base_model_path" \
    --report_to "wandb" \
    --learning_rate 3e-6 \
    --max_grad_norm 1 \
    --weight_decay 0. \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 1600 \
    --save_total_limit 5 \
    --output_dir "$output_dir" \
    --run_name "$WANDB_NAME" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 2048 \
    --collator_type "VLAMultimodalChatDataCollatorforVLM" \
    --fix_visual_encoder True \
    --fix_visual_adapter True \
    --min_pixels $min_pixels \
    --max_pixels $max_pixels
