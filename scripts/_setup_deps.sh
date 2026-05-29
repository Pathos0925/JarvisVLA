#!/usr/bin/env bash
# One-shot deps install into the active conda env, matching README "Pinned deps".
# Logged section markers so progress is greppable.
set -uo pipefail
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

say() { echo "==== [$(date +%H:%M:%S)] $* ===="; }

say "PHASE 1: core ML stack (transformers/trl/deepspeed/accelerate + data + utils)"
pip install \
    "transformers==5.9.0" "trl==0.16.0" "deepspeed==0.16.3" \
    accelerate mergekit \
    datasets pyarrow openai wandb boto3 \
    "pillow==10.4.0" "numpy==1.26.4" qwen-vl-utils av \
    opencv-python-headless sentencepiece safetensors \
    || { echo "PHASE1_FAILED"; exit 1; }

say "PHASE 1b: upgrade networkx (mergekit pulls an ancient one)"
pip install -U networkx || { echo "PHASE1b_FAILED"; exit 1; }

say "PHASE 2: CUDA-compiled wheels (no build isolation; prefer prebuilt)"
pip install --no-build-isolation \
    "causal-conv1d==1.6.2.post1" \
    "flash-linear-attention==0.5.0" \
    "flash-attn==2.8.3" \
    || { echo "PHASE2_FAILED"; exit 1; }

say "PHASE 3: Hopper workaround — triton 3.3.1, drop tilelang"
pip install "triton==3.3.1" || { echo "PHASE3_FAILED"; exit 1; }
pip uninstall -y tilelang 2>/dev/null || true

say "ALL_DEPS_DONE"
