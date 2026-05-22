#!/usr/bin/env bash
# Download a JarvisVLA checkpoint from R2 and serve it via vLLM.
#
# Designed to be portable to any machine with:
#   * Python 3.10+
#   * CUDA driver supporting whatever torch vLLM ships (vllm 0.21 needs CUDA 12.8+)
#   * boto3, vllm, huggingface-hub installed
#   * R2 credentials in env or .env file (same vars used by jarvisvla/train/r2_callback.py)
#
# Usage (on a fresh machine):
#   # 0. One-time setup
#   git clone https://github.com/Pathos0925/JarvisVLA.git && cd JarvisVLA
#   python -m venv .venv && source .venv/bin/activate
#   pip install 'vllm>=0.21' boto3 huggingface-hub
#
#   # 1. R2 credentials (either in .env or shell)
#   cat > .env <<EOF
#   R2_BUCKET=jarvis
#   R2_ACCOUNT_ID=<your-cloudflare-account-id>
#   R2_ACCESS_KEY=<your-key-id>
#   R2_SECRET_ACCESS_KEY=<your-secret>
#   EOF
#
#   # 2. Run it
#   bash scripts/inference/r2_fetch_and_serve.sh                 # serves /final
#   bash scripts/inference/r2_fetch_and_serve.sh checkpoint-30000  # any subdir
#
# What it does:
#   1. Sources .env if present so R2_* / HF_TOKEN propagate.
#   2. Downloads the named subdir from R2 to $LOCAL_DIR, skipping resume-only files
#      (optimizer.pt, scheduler.pt, rng_state_*.pth) — saves ~34 GiB on a 51 GiB ckpt.
#   3. Fetches preprocessor_config.json + video_preprocessor_config.json from the base
#      model on HF Hub (Qwen/Qwen3.5-9B) — these aren't saved by HF Trainer in step
#      checkpoint subdirs but are required by AutoProcessor.from_pretrained.
#   4. Runs vllm serve with sane defaults for a 9B Qwen3.5 model (single GPU,
#      qwen3_next_mtp speculative decoding, prefix caching). Override via env.
#
# Env vars:
#   R2_PREFIX           remote dataset prefix (default: mc-vla-qwen3-5-9b-h200-full-e1-e1-b4-a1-p1.0)
#   LOCAL_DIR           where to download (default: ./jarvisvla-checkpoint)
#   BASE_MODEL          HF Hub repo for processor configs (default: Qwen/Qwen3.5-9B)
#   PORT                vllm port (default 9052)
#   N_SPEC              num speculative tokens (default 2)
#   MAX_MODEL_LEN       (default 8448)
#   CUDA_VISIBLE_DEVICES, CARD_NUM, GPU_MEMORY_UTILIZATION (vllm-side overrides)
#   SKIP_DOWNLOAD=1     reuse existing $LOCAL_DIR contents, only serve
#   SKIP_SERVE=1        download only, don't launch vllm
set -euo pipefail

CKPT_NAME="${1:-final}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env so R2 credentials propagate.
ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
fi

R2_PREFIX="${R2_PREFIX:-mc-vla-qwen3-5-9b-h200-full-e1-e1-b4-a1-p1.0}"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/jarvisvla-checkpoint}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-9052}"
N_SPEC="${N_SPEC:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8448}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CARD_NUM="${CARD_NUM:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

echo "==> r2_fetch_and_serve"
echo "  checkpoint:    $CKPT_NAME"
echo "  R2 prefix:     $R2_PREFIX/$CKPT_NAME"
echo "  local dir:     $LOCAL_DIR"
echo "  base model:    $BASE_MODEL"
echo "  vllm:          port=$PORT card_num=$CARD_NUM gpu_mem=$GPU_MEMORY_UTILIZATION max_len=$MAX_MODEL_LEN"

# Export local vars the python heredoc reads — must precede the heredoc.
export _R2_PREFIX="$R2_PREFIX/$CKPT_NAME"
export _LOCAL_DIR="$LOCAL_DIR"

# ---------- 1. Download from R2 ----------

if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "==> SKIP_DOWNLOAD=1; reusing $LOCAL_DIR"
else
    : "${R2_BUCKET:?R2_BUCKET not set (put it in .env)}"
    : "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID not set}"
    : "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY not set}"
    if [ -z "${R2_ACCESS_KEY_ID:-}${R2_ACCESS_KEY:-}" ]; then
        echo "ERROR: set R2_ACCESS_KEY_ID (or R2_ACCESS_KEY)" >&2; exit 1
    fi

    mkdir -p "$LOCAL_DIR"
    echo "==> downloading from R2..."
    python3 - <<PY
import os, sys, time
from fnmatch import fnmatch
from pathlib import Path
import boto3

bucket = os.environ["R2_BUCKET"]
prefix = os.environ["_R2_PREFIX"].rstrip("/") + "/"
local_dir = Path(os.environ["_LOCAL_DIR"])
endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
key_id = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")

# Files only needed for resuming training, not inference. Skipping saves ~34 GiB
# of bandwidth on a typical 51 GiB JarvisVLA checkpoint.
SKIP = ("optimizer.pt", "scheduler.pt", "rng_state_*.pth", "training_args.bin", "trainer_state.json")

s3 = boto3.client("s3", endpoint_url=endpoint,
                  aws_access_key_id=key_id,
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  region_name="auto")

paginator = s3.get_paginator("list_objects_v2")
total_bytes = 0
downloaded = 0
skipped = 0
t0 = time.time()
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        name = key[len(prefix):]
        if any(fnmatch(name, p) for p in SKIP):
            print(f"  skip {name}  ({obj['Size']/2**30:.2f} GiB)")
            skipped += 1
            continue
        local = local_dir / name
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size == obj["Size"]:
            print(f"  reuse {name}  ({obj['Size']/2**30:.2f} GiB)")
            continue
        print(f"  pulling {name}  ({obj['Size']/2**30:.2f} GiB)...", flush=True)
        s3.download_file(bucket, key, str(local))
        downloaded += 1
        total_bytes += obj["Size"]

if downloaded == 0 and not any(local_dir.iterdir()):
    print(f"ERROR: no objects under s3://{bucket}/{prefix}", file=sys.stderr)
    sys.exit(2)
dt = time.time() - t0
rate = total_bytes/dt/2**30 if dt > 0 else 0
print(f"==> done: {downloaded} downloaded, {skipped} skipped (resume-only), "
      f"{total_bytes/2**30:.2f} GiB in {dt:.1f}s ({rate:.2f} GiB/s)")
PY
fi

# ---------- 2. Fetch preprocessor configs from HF Hub ----------

# Why these aren't in the R2 checkpoint: HF Trainer's save_strategy=steps writes
# the tokenizer + model + processor_config.json + chat_template but NOT the per-
# modality preprocessor_config.json or video_preprocessor_config.json. They come
# from the base model and don't change during SFT.
for f in preprocessor_config.json video_preprocessor_config.json; do
    if [ ! -f "$LOCAL_DIR/$f" ]; then
        echo "==> fetching $f from $BASE_MODEL on HF Hub"
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$BASE_MODEL', filename='$f', local_dir='$LOCAL_DIR')
print('  wrote $LOCAL_DIR/$f')
"
    fi
done

# ---------- 3. Sanity check ----------

for required in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json chat_template.jinja preprocessor_config.json; do
    if [ ! -e "$LOCAL_DIR/$required" ] && [ ! -e "$LOCAL_DIR"/model*.safetensors* ]; then
        echo "WARNING: $required missing from $LOCAL_DIR"
    fi
done

echo "==> $LOCAL_DIR contents:"
ls -la "$LOCAL_DIR" | head -25

if [ "${SKIP_SERVE:-0}" = "1" ]; then
    echo "==> SKIP_SERVE=1; download done."
    exit 0
fi

# ---------- 4. Serve via vLLM ----------

echo "==> starting vllm serve on port $PORT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" exec vllm serve "$LOCAL_DIR" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 10 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$CARD_NUM" \
    --trust-remote-code \
    --served-model-name jarvisvla \
    --limit-mm-per-prompt '{"image":5}' \
    --enable-prefix-caching \
    --speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$N_SPEC}"

# Thinking mode is per-request: clients should pass enable_thinking=false via
# the extra_body kwarg (agent_wrapper.py does this for you).
# Action schema: this checkpoint emits Qwen2-VL action tokens — see README
# "Debugging history" for the fix. Set ACTION_SCHEMA=qwen2_vl on the client.
