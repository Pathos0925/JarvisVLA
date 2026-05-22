#!/bin/bash
# vLLM serve for the Qwen3.5-9B JarvisVLA executor.
#
# Speedup: native MTP via Qwen3.5-9B's qwen3_next_mtp speculative-decoding mode (no
# training-side wiring required — see REIMPLEMENTATION_PLAN.md Step 7).
#
# Override paths via env vars:
#   MODEL_PATH=...   path to SFT'd checkpoint  (default: Qwen/Qwen3.5-9B base)
#   PORT=...         (default 9052)
#   N_SPEC=...       num_speculative_tokens for MTP (default 2; raise to 3 only if
#                    second-token acceptance ≥80% on the eval suite)
set -euo pipefail

cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
card_num="${CARD_NUM:-4}"
model_name_or_path="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
port="${PORT:-9052}"
n_spec="${N_SPEC:-2}"

# max-model-len budget: text history + vision tokens + a small output budget. With
# max_pixels=256*28*28 a single image is ~256 vision tokens; cap the request payload at
# ~8K which comfortably fits chunk_len=4 inference with a few-frame history.
max_model_len="${MAX_MODEL_LEN:-8448}"

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve "$model_name_or_path" \
    --port "$port" \
    --max-model-len "$max_model_len" \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size "$card_num" \
    --trust-remote-code \
    --served_model_name "jarvisvla" \
    --limit-mm-per-prompt image=5 \
    --enable-prefix-caching \
    --speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$n_spec}"

# Thinking mode is a per-request kwarg, not a serve flag — pass enable_thinking=false in
# the client request (agent_wrapper.py sends this via extra_body once enabled).
