# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse

[![arXiv](https://img.shields.io/badge/arXiv-2503.16365-df2a2a.svg?style=for-the-badge)](https://arxiv.org/pdf/2503.16365)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/collections/CraftJarvis/jarvis-vla-v1-67dc157a99d011efd7d7f7e4)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Project Website**](https://craftjarvis.github.io/JarvisVLA/) | [**Datasets**](https://huggingface.co/datasets/CraftJarvis/minecraft-vla-sft)

This repo is in the middle of a Phase-1 reimplementation that moves the executor backbone from `Qwen2-VL-7B` to `Qwen3.5-9B` and adds chunked-action SFT. The design rationale and step-by-step plan live in [REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md). This README is the operational reference — what to run, what's been validated, and what's known to bite.

## Updates

- **[2026.05.22 evening]** **Full Phase-1 SFT launched on 2× H200** (~32 hr at ~1 s/step on `DATASET_P=1.0`, `EMBED_LR=3e-5`, `LIGER=0`, `SAVE_STEPS=5000`). Pre-launch second-round review (Opus 4.7 / GPT-5.5 Pro / Gemini 3.1 Pro) caught and fixed: `--warmup_steps 100` overriding the ratio (dropped — warmup_ratio 0.03 now → 3480 warmup steps); EMBED_LR=30× over 116K steps risking catastrophic forgetting on pretrained vocab (lowered to 10×); R2 callback silently dropping failed uploads (now retries 4× with exponential backoff and surfaces failures); bash `${VAR:+...}` treating "0" as truthy (replaced with proper `_truthy` helper); `.env` not auto-loaded into training process (script now sources it). Also: `agent_wrapper.py` accepts `ACTION_SCHEMA` env to handle the Qwen2-VL action-token emission from a `--backbone=qwen3_5` checkpoint. Two prior smokes (`fix1`, `speedup1`) produced PARSEABLE checkpoints with final loss 0.66; speedup attempt (Liger+no-GC+workers=6) was net -2.3% so reverted; Liger machinery kept as opt-in. **R2 upload confirmed working** at training start.
- **[2026.05.22 morning]** Phase-1 SFT smoke run validated on 2× H200 144 GB. End-to-end pipeline (preprocessor → SFT → inspector) green, plus a multi-LLM code review surfaced one critical bug (image double-rescale in the data collator) and one important tuning issue (LR too low for newly-added action token embeddings) — both fixed; retrain landed PARSEABLE checkpoint with clean Qwen2-VL action grammar (4/4 chunks per response). New: H200-simplified SFT script (no DeepSpeed), async R2 checkpoint upload, checkpoint inspector, `DifferentialLRTrainer` for per-matrix learning rates, vllm in a separate conda env.
- **[2026.05.21]** Started reimplementation of the VLA backbone on **`Qwen/Qwen3.5-9B`** (multimodal). Planner for Phase 2 will be **`google/gemma-4-26B-A4B-it`**. See [REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md) for the full plan.
- **[2025.03.21]** Paper available on [arXiv](https://arxiv.org/pdf/2503.16365).

---

## Reimplementation status

The next generation of JARVIS-VLA targets:

- **Executor:** `Qwen/Qwen3.5-9B` (9B hybrid Gated-DeltaNet + Gated-Attention, multimodal despite the name). Inference speedup via native MTP through vLLM's `qwen3_next_mtp` speculative-decoding mode (no training-side wiring required).
- **Planner (Phase 2):** `google/gemma-4-26B-A4B-it` (sparse MoE, 3.8B activated of 25.2B, Apache 2.0, native tool calling). Emits sub-goals with predicate-based success criteria evaluated against MineStudio's info dict — no second VLM call needed for completion detection.

### At a glance

| Stage | Status |
| --- | --- |
| Plan + design review | ✅ done ([REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md); frontier-LLM critique under `runs/reviews/`) |
| Foundation refactor (action tokens, `--backbone`, helpers) | ✅ done, commits `65d0b2b` → `8c32ec5` |
| Smoke test against real Qwen3.5-9B | ✅ done (`tests/smoke_qwen3_5.py`) |
| Chunked-action SFT preprocessor | ✅ done; 935,440 chunks at 99.1% yield, 100% with exactly 4 action segments |
| Multi-GPU SFT dry-run (5 steps) | ✅ done; loss 10.51 → 9.76 on the 3× A100 rig, ~1 s/step on 2× H200 |
| Production DeepSpeed SFT script (3× A100) | ✅ done (`scripts/train/vla_qwen3_5_9b_sft.sh`) |
| H200 SFT script (no DeepSpeed, BATCH=4) | ✅ done (`scripts/train/vla_qwen3_5_9b_sft_h200.sh`) |
| Phase-1 3000-step smoke + inspector | ✅ done; image preprocessing bug + LR-for-new-tokens fixed; fix1 + speedup1 smokes both PARSEABLE (loss 0.66, 4/4 chunks emitted) |
| Async R2 checkpoint upload | ✅ done with retry (`jarvisvla/train/r2_callback.py`); auto-loads from `.env` |
| Full Phase-1 SFT run | ⏳ **running** — launched 2026-05-22 evening, ~32 hr ETA, 115,763 steps. checkpoint-5000 PARSEABLE (loss ~0.5), R2 upload verified (50.92 GiB in 549 s). |
| vLLM serve + smoke rollout | ⏳ pending SFT (deferred install — see [Known gotchas](#known-gotchas)) |
| Headline eval vs Qwen2-VL baseline | ⏳ pending SFT |
| Async pipelining in `agent_wrapper` | ⏳ deferred |
| Phase 2 (Gemma planner + sub-goal queue) | ⏳ pending Phase-1 gate |

### What's landed (Phase 1 foundation)

- **Programmatic action-token mapping** (`jarvisvla/inference/action_tokens.py`) — schemas for Qwen2-VL (backward-compat via reserved-special-token slot strings) and Qwen3.5 (canonical `<|act_*|>` names added via `add_special_tokens`). IDs resolved against the live tokenizer at startup, persisted next to the checkpoint, verified at inference. Replaces the previous hard-coded 168-line ID table in `action_mapping.py` (which assumed Qwen2-VL's 151,936-token vocab and silently broke on Qwen3.5's 248,320).
- **Backbone-agnostic training pipeline** (`jarvisvla/train/train.py`) — `--backbone {qwen2_vl,qwen3_5}` flag; the qwen3_5 branch uses `AutoProcessor` + `AutoModelForImageTextToText`; embedding resize, freeze-pattern assertion, thinking-mode disable all centralized in `utils_train.py`.
- **`DifferentialLRTrainer`** (`jarvisvla/train/train.py`) — subclass of `transformers.Trainer` that puts `embed_tokens` + `lm_head` weights in a separate optimizer group at a higher LR (default `30 × args.learning_rate`, override with `EMBED_LR`). Necessary because the 73 newly-added action-token rows are randomly initialized and need many orders of magnitude more LR than the pretrained backbone to escape the noise floor.
- **Chunked-action SFT data preprocessor** (`scripts/preprocess_chunked_actions.py`) — re-renders the source dataset so each assistant turn contains N concatenated action segments (default N=4). Sort-then-chunk with a trajectory-level holdout for a real valid split.
- **DeepSpeed SFT script** (`scripts/train/vla_qwen3_5_9b_sft.sh`) — production config: ZeRO-2 + CPU optimizer offload, FA2 + fla, frozen vision encoder/adapter. Validated by the dry-run.
- **H200 SFT script** (`scripts/train/vla_qwen3_5_9b_sft_h200.sh`) — single-node simplified variant: `torchrun --nproc-per-node=2`, no DeepSpeed (144 GB fits the full model + Adam states without ZeRO sharding), `BATCH=4` default.
- **Async R2 checkpoint upload** (`jarvisvla/train/r2_callback.py`) — `TrainerCallback` that uploads each `checkpoint-N/` to Cloudflare R2 as soon as it's written, on a background thread pool. Rank-0 only, idempotent via HEAD checks, `wait_all()` at end of training to flush.
- **Checkpoint inspector** (`tests/inspect_checkpoint.py`) — load a checkpoint and generate actions on real valid-split samples; classify output as `PARSEABLE` / `PARTIAL` / `MALFORMED` based on whether the `<|act_start|>` … `<|act_end|>` grammar is emitted. Cheap proxy for "did training actually produce a working action policy?" before spinning up vLLM.
- **Inference path tightening** (`jarvisvla/evaluate/agent_wrapper.py`) — `max_tokens` capped at `chunk_len × 16` (was 1024), default temperature 0.5 → 0.1, decode-health log on every step.
- **Multi-seed eval support** (`jarvisvla/evaluate/evaluate.py`) — `--seed-base` flag.
- **Bug fixes caught during the refactor:** `group_action_2_token` was silently dropping the inventory-flag group; `apply_private_conversations` shadowed its input arg; the masking-loop silent-on-zero-matches behavior (loss leaked over user prompts and image regions). All covered by regression tests.

### What's pending

- Full Phase-1 SFT run (smoke with fixes → medium → full epoch) + headline eval vs the Qwen2-VL 7B baseline.
- Async pipelining in the rollout loop (kick off chunk N+1 generation while env executes chunk N).
- vLLM in a separate venv (deferred install — current vLLM versions want to clobber the SFT env's torch/triton/cuda).
- Phase 2: Gemma-4-26B-A4B planner + predicate-based sub-goal queue.

---

## Quickstart on a new machine

```bash
# 1. Clone + activate env (Python 3.11 is what's been tested)
git clone https://github.com/Pathos0925/JarvisVLA.git
cd JarvisVLA
conda create -n myenv python=3.11 -y && conda activate myenv

# 2. Env vars — point caches at a large, user-writable disk
export HF_HOME=/path/to/large/disk/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTHONPATH=$(pwd)         # setup.py is broken (deprecated pkg_resources); use PYTHONPATH

# 3. Install deps (see "Pinned deps" below for the exact versions that worked)
pip install -r requirements.txt                                          # may need to skip vllm — see notes
pip install accelerate mergekit 'trl==0.16.0' 'deepspeed==0.16.3'
pip install --no-build-isolation causal-conv1d flash-linear-attention flash-attn
pip install -U networkx
# On 2× H200 (Hopper) only:
pip install 'triton==3.3.1'                                              # avoids fla's "Triton ≥3.4 on Hopper" miscompile check
pip uninstall -y tilelang                                                # if present; requires CUDA 12.8+ to compile

# 4. Download model + regenerate dataset (~20 GB + ~300 GB cache → ~25 GB chunked output)
huggingface-cli download Qwen/Qwen3.5-9B --local-dir /path/to/models/Qwen3.5-9B
python scripts/preprocess_chunked_actions.py --chunk-len 4 \
    --output-dir /path/to/datasets/jarvisvla-chunk4 \
    --holdout-fraction 0.01 --splits train

# 5. Verify the foundation still works on this machine
python -m tests.test_action_mapping              # expect: 7/7 passed
MODEL_PATH=/path/to/models/Qwen3.5-9B python -m tests.smoke_qwen3_5
                                                  # expect: forward+backward, loss ~18, no errors

# 6. Launch the Phase-1 SFT — see "Training" below for tuning
BASE_MODEL_PATH=/path/to/models/Qwen3.5-9B \
DATASET_NAME=/path/to/datasets/jarvisvla-chunk4 \
OUTPUT_DIR=/path/to/checkpoints/qwen3_5_9b_phase1 \
nohup bash scripts/train/vla_qwen3_5_9b_sft_h200.sh > sft.log 2>&1 &     # or vla_qwen3_5_9b_sft.sh on multi-A100
```

---

## Hardware

Two known-good rigs, with the production script picked per setup.

| Setup | Estimated step time | Script | Notes |
| --- | --- | --- | --- |
| 3× A100 80GB | 6.7 s/step | `vla_qwen3_5_9b_sft.sh` | Original handoff rig. DeepSpeed ZeRO-2 + CPU optimizer offload is required — without offload the optimizer step OOMs. |
| 2× H100 80GB | ~3 s/step (proj.) | `vla_qwen3_5_9b_sft.sh` | Same DeepSpeed config, less host RAM needed. |
| 1× H100 80GB | ~3.5 s/step (proj.) | `vla_qwen3_5_9b_sft.sh` | Marginal — barely fits without offload at seq=1024. |
| **2× H200 144GB** | **~1 s/step (measured)** | **`vla_qwen3_5_9b_sft_h200.sh`** | **Current dev rig.** 144 GB per GPU fits full model + Adam states + activations without ZeRO sharding. |
| 1× B200 192GB | ~1.5 s/step (proj.) | not yet written | Single-process, no DeepSpeed, can probably push `BATCH=8+`. FP8 via transformer engine could give 1.5–2× more on matmul-heavy parts. |

H100/H200 = Hopper, A100 = Ampere, B200 = Blackwell. Hopper has a known fla/Triton miscompile that needs the workaround in [Known gotchas](#known-gotchas).

### Throughput projection at ~1 s/step on 2× H200

Batch=4 per GPU × 2 GPUs × 1 grad-accum = 8 effective samples/step.

| Scope | Steps | Wall on 2× H200 | Original handoff (3× A100) |
| --- | --- | --- | --- |
| `DATASET_P=0.01 MAX_STEPS=3000` (smoke) | 3,000 | ~50 min | 6 hr |
| `DATASET_P=0.12` (medium) | ~14,000 | ~4 hr | 24 hr |
| Full epoch (DATASET_P=1.0) | ~116,000 | ~32 hr | 24 days |

Full epoch went from "infeasible" to "overnight + a workday" on the hardware change. That should be the canonical Phase-1 gate run.

---

## Pinned deps (what actually worked)

| Package | Version | Notes |
| --- | --- | --- |
| python | 3.11 | conda env named `myenv` in tested setup |
| torch | 2.12.0+cu128 (handoff) / **2.12.0+cu126** (H200 box) | match CUDA driver — cu128 wheels need driver ≥ 12.8 |
| transformers | **5.9.0** | introduces `processing_class` arg; `train.py` handles both old and new APIs at runtime |
| trl | **0.16.0** | **Pinned** — 0.12 lacks `trl.scripts`; ≥1.0 renames `max_seq_length` → `max_length` |
| deepspeed | **0.16.3** | **Pinned** — 0.19+ has a broken Muon optimizer import |
| accelerate | 1.6.0 | latest works; handoff originally pinned 1.2.1 but mergekit pulls newer |
| mergekit | 0.1.4 | TRL 0.16 imports it transitively |
| networkx | ≥3.0 | mergekit pulls an ancient one with `from collections import Mapping`; upgrade fixes it |
| flash-attn | 2.8.3 | install with `--no-build-isolation` |
| flash-linear-attention | 0.5.0 | for Qwen3.5-9B's gated DeltaNet layers |
| fla-core | 0.5.0 | dep of flash-linear-attention |
| causal-conv1d | 1.6.2.post1 | needed by fla |
| triton | 3.7.0 (default with torch 2.12) / **3.3.1 on Hopper** | downgrade on H100/H200; see [Known gotchas](#known-gotchas) gotcha 11 |
| tilelang | **uninstalled on CUDA<12.8** | fla suggests installing it as the Hopper workaround, but it needs CUDA 12.8+ FP8 E8M0 intrinsics to compile |
| boto3 | latest | only required if R2 checkpoint upload is enabled |
| datasets, pyarrow, openai | latest | regular deps |
| vllm | **not installed** | latest 0.21 wants to downgrade torch/triton and pull cu13 packages — would break SFT env. Install in a separate venv for serving. |

`--no-build-isolation` is required for the CUDA-compiled wheels because pip's isolated build env grabs a newer PyTorch with a different CUDA version, which fails the extension build.

`setup.py` uses deprecated `pkg_resources` and will fail under modern setuptools. **Use `PYTHONPATH` instead of `pip install -e .`.** Fixing setup.py is a small follow-up — convert to `pyproject.toml`.

---

## Training

### Production (multi-A100, DeepSpeed)

`scripts/train/vla_qwen3_5_9b_sft.sh` — DeepSpeed launcher across all visible GPUs, ZeRO-2 + CPU optimizer offload (`configs/deepspeed_config_s2_offload.json`), FA2 + fla, frozen vision encoder/adapter.

```bash
# 6-hour smoke SFT on 1% of data (~10K samples, ~3000 micro-steps)
DATASET_P=0.01 MAX_STEPS=3000 bash scripts/train/vla_qwen3_5_9b_sft.sh

# 24-hour run on 12% slice (~111K samples)
DATASET_P=0.12 bash scripts/train/vla_qwen3_5_9b_sft.sh

# Full epoch (only attempt with B200 or many H100s; ~24 days on 3× A100)
bash scripts/train/vla_qwen3_5_9b_sft.sh
```

### Single-node H200 (no DeepSpeed)

`scripts/train/vla_qwen3_5_9b_sft_h200.sh` — `torchrun --nproc-per-node=$N`, no DeepSpeed (144 GB fits), `BATCH=4` default.

```bash
# Smoke (~50 min on 2× H200)
DATASET_P=0.01 MAX_STEPS=3000 RUN_TAG=smoke bash scripts/train/vla_qwen3_5_9b_sft_h200.sh

# Medium (~4 hr)
DATASET_P=0.12 RUN_TAG=medium bash scripts/train/vla_qwen3_5_9b_sft_h200.sh

# Full epoch (~32 hr)
RUN_TAG=full-e1 bash scripts/train/vla_qwen3_5_9b_sft_h200.sh
```

### Env var overrides (both scripts)

| Var | Default | Effect |
| --- | --- | --- |
| `BASE_MODEL_PATH` | `/ephemeral/models/Qwen3.5-9B` | model dir |
| `DATASET_NAME` | `/ephemeral/datasets/jarvisvla-chunk4` (DS script) / `/workspace/datasets/jarvisvla-chunk4` (H200) | dataset dir |
| `OUTPUT_DIR` | `${EPHEMERAL_OR_WORKSPACE}/checkpoints/<run_tag>` | checkpoint dir |
| `CUDA_VISIBLE_DEVICES` | `0,1,2` (DS) / `0,1` (H200) | GPUs to use; H200 script derives `nproc` from this |
| `BATCH` | 1 (DS) / 4 (H200) | `--per_device_train_batch_size` |
| `GRAD_ACCUM` | 1 | `--gradient_accumulation_steps` |
| `DATASET_P` | 1.0 | fraction of the train set to use (0..1) |
| `MAX_STEPS` | unset | hard cap on optimizer steps |
| `EPOCH` | 1 | `--num_train_epochs` |
| `RUN_TAG` | timestamp | suffix on output dir + wandb name |
| `EMBED_LR` | `30 × learning_rate` | LR for `embed_tokens` + `lm_head` matrices (see [DifferentialLRTrainer](#differential-learning-rate)) |
| `TRAINING_PORT` | 24001 | torchrun / deepspeed master port |

### Differential learning rate

`DifferentialLRTrainer` puts `embed_tokens.weight` and `lm_head.weight` into their own optimizer group with `EMBED_LR` (default 30× the base LR). Required because the 73 newly-added action-token rows are randomly initialized — a global LR of `3e-6` is too low by orders of magnitude to pull them out of the noise floor in a few thousand steps, and the model emits English/nonsense instead of action tokens. Pretrained rows in the same matrix get the same elevated LR; over short SFT runs the drift is tolerable, but if you're doing long full-epoch training and want sharper isolation, consider per-row gradient masking instead.

### R2 (Cloudflare) checkpoint upload

Opt-in via env vars; absent → callback returns `None` and training proceeds unchanged.

```bash
# Required
export R2_BUCKET=your-bucket
export R2_ACCOUNT_ID=your-cloudflare-account-id
export R2_ACCESS_KEY_ID=your-key-id
export R2_SECRET_ACCESS_KEY=your-secret

# Optional
export R2_PREFIX=runs/jarvisvla   # default: basename of OUTPUT_DIR
export R2_MAX_WORKERS=4           # default: 4 background upload workers
```

Then launch training as usual. The callback uploads each `checkpoint-N/` as Trainer writes it (async, rank-0 only, ThreadPoolExecutor) plus the post-`save_model` end-of-training save under `<prefix>/final/`. Idempotent per-file via HEAD/ContentLength check, so resumed runs and re-launches don't re-upload. `wait_all()` blocks process exit until all in-flight uploads finish.

---

## Inference

### vLLM serve (currently in a separate venv — see [Known gotchas](#known-gotchas) gotcha 12)

```bash
MODEL_PATH=/path/to/checkpoint bash scripts/inference/serve_vllm_qwen3_5.sh
```

Uses native MTP via Qwen3.5-9B's `qwen3_next_mtp` speculative-decoding mode (no training-side wiring required). Thinking mode is a per-request kwarg, not a serve flag — pass `enable_thinking=false` in the client request body.

### Checkpoint inspector (no vLLM needed)

```bash
PYTHONPATH=. MODEL_PATH=/path/to/checkpoints/.../checkpoint-3000 \
    python -m tests.inspect_checkpoint
```

Loads the checkpoint and generates actions for 3 valid-split samples using `transformers.generate`. Reports per-sample stats (chunks opened, chunks closed, non-action tokens, verdict). Useful as a cheap "did this checkpoint learn the action grammar at all" sanity check before spinning up the full vLLM serve + rollout. Note: HF Trainer's `save_strategy="steps"` checkpoint subdirs are missing `preprocessor_config.json` and `video_preprocessor_config.json`; copy them from the base model dir before running the inspector.

### Rollout + eval

After SFT lands and vLLM is up, the order of operations is:

1. **Single-episode rollout** — `python -m jarvisvla.evaluate.evaluate --workers 0 --checkpoints /path/to/checkpoint --base-url http://localhost:9052/v1`. Look for action-decode warnings and smoothness.
2. **Headline eval** — multi-seed (`--seed-base 0`, `--workers 5+`) on the existing config suite. Compare success rate + p50 step latency to the Qwen2-VL 7B baseline. Phase-1 gate is "matches or exceeds baseline" — see REIMPLEMENTATION_PLAN.md §Step 8.
3. **If gate passes** → start Phase 2 (Gemma planner).
4. **If gate doesn't pass** → bisect via the recipe knobs in REIMPLEMENTATION_PLAN.md §Step 8 (LR sweep, unfreeze vision encoder, vision-token budget 256 → 512).

### Monitoring a live training run

```bash
# Is the SFT process still alive?
pgrep -af "torchrun.*jarvisvla" | head -3

# Latest training metrics (filters out tqdm progress noise)
grep -E "'loss':|'grad_norm':|train_runtime" /tmp/sft_full.log | tail -10

# How many R2 uploads have happened?
grep -E "r2-upload" /tmp/sft_full.log | tail -20

# Inspect the most recent saved checkpoint (need to copy preprocessor configs first)
CKPT=$(ls -t /workspace/checkpoints/*/checkpoint-* -d | head -1)
cp /workspace/models/Qwen3.5-9B/preprocessor_config.json /workspace/models/Qwen3.5-9B/video_preprocessor_config.json "$CKPT"/
PYTHONPATH=. MODEL_PATH="$CKPT" python -m tests.inspect_checkpoint

# GPU utilization (sanity: both H200s should be busy)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# Disk pressure (save_total_limit=3 keeps ~150 GB rotating)
df -h /workspace; du -sh /workspace/checkpoints/*/
```

If you need to **kill** the run cleanly:
```bash
pkill -TERM -f "torchrun.*jarvisvla"  # SIGTERM first
sleep 5
pkill -KILL -f "jarvisvla/train"      # SIGKILL if still alive
```

To **resume from a checkpoint** (saved with `SAVE_ONLY_MODEL=0`, the default):
Trainer auto-resumes if `OUTPUT_DIR` contains `checkpoint-*` subdirs. Re-run the same launch command and it picks up where it left off.

### Rollout after SFT completes

Use the separate `vllm` conda env (vllm 0.21 + cu13 packages live there, isolated from SFT):

```bash
conda activate vllm  # /home/user1/miniconda3/envs/vllm
MODEL_PATH=/path/to/final/checkpoint bash scripts/inference/serve_vllm_qwen3_5.sh
```

For rollout via `jarvisvla.evaluate.evaluate`, **set `ACTION_SCHEMA=qwen2_vl`** (until the
re-preprocessing or in-collator translation lands — see "Debugging history"):

```bash
ACTION_SCHEMA=qwen2_vl python -m jarvisvla.evaluate.evaluate \
    --workers 0 --checkpoints /path/to/final/checkpoint \
    --base-url http://localhost:9052/v1
```

### Multi-LLM review

`scripts/review.py` sends the plan + selected files to multiple frontier models via OpenRouter in parallel, saves their critiques under `runs/reviews/<timestamp>/`. Useful for getting a second opinion before committing to a long run. Requires `OPENROUTER_API_KEY` in `.env` or env.

```bash
python scripts/review.py \
    --include README.md \
    --include scripts/train/vla_qwen3_5_9b_sft_h200.sh \
    --include jarvisvla/train/data_collator.py \
    --question "What's wrong with this setup before I commit to a 32-hour run?"
```

The 2026-05-22 review (Opus 4.7 + GPT-5.5 Pro + Gemini 3.1 Pro) caught two bugs the smoke run didn't surface — see [Debugging history](#debugging-history) below.

---

## Project layout — files added or substantively changed during the reimplementation

| File | Role |
| --- | --- |
| `jarvisvla/inference/action_tokens.py` *(NEW)* | Per-backbone schemas + `build_id_maps()` (resolves action-token strings to IDs at runtime). |
| `jarvisvla/inference/action_mapping.py` | `OneActionTokenizer.from_tokenizer(...)`, no more module-level dispatch functions; **fix**: inventory-flag bug in `group_action_2_token`. |
| `jarvisvla/inference/load_model.py` | Accepts explicit `backbone=` override; recognizes `qwen3_5`. |
| `jarvisvla/train/train.py` | `--backbone` flag; `qwen3_5` branch; embedding + aux-head resize; disable thinking mode; assert freeze patterns; processor `save_pretrained`; `use_cache` restore; `processing_class`/`tokenizer` compat shim. **New**: `DifferentialLRTrainer`, R2 callback wiring. |
| `jarvisvla/train/utils_train.py` | `MoreConfig.backbone`; `resize_aux_heads`, `disable_thinking_mode`, `assert_freeze_patterns_match`. |
| `jarvisvla/train/data_collator.py` | Accepts `backbone=`; chat-template prefix IDs derived dynamically from the live tokenizer; loud assertion on zero matches; removed broken `apply_private_conversations`. **Fix**: removed `transforms.ToTensor()` from `image_process` — see [Debugging history](#debugging-history). |
| `jarvisvla/train/r2_callback.py` *(NEW)* | Async R2 (Cloudflare S3-compatible) upload of checkpoints via TrainerCallback. |
| `jarvisvla/evaluate/agent_wrapper.py` | `from_tokenizer`-based action_tokenizer; max_tokens capped; temp 0.5→0.1; decode-health log. |
| `jarvisvla/evaluate/evaluate.py` | `--seed-base` flag (per-worker seed = base + worker_id). |
| `scripts/preprocess_chunked_actions.py` *(NEW)* | Sort-then-chunk preprocessor with trajectory-level holdout. |
| `scripts/verify_chunked_dataset.py` *(NEW)* | Structural verification of chunked output. |
| `scripts/train/vla_qwen3_5_9b_sft.sh` *(NEW)* | Production multi-GPU DeepSpeed SFT (ZeRO-2 + CPU offload + FA2 + fla). |
| `scripts/train/vla_qwen3_5_9b_sft_dryrun.sh` *(NEW)* | Bounded 5-step dry-run (DeepSpeed). |
| `scripts/train/vla_qwen3_5_9b_sft_h200.sh` *(NEW)* | Single-node H200 SFT (torchrun, no DeepSpeed, BATCH=4). |
| `scripts/inference/serve_vllm_qwen3_5.sh` *(NEW)* | vLLM serve with `qwen3_next_mtp` speculative decoding. |
| `scripts/review.py` *(NEW)* | Multi-LLM plan review via OpenRouter. |
| `configs/deepspeed_config_s2_offload.json` *(NEW)* | ZeRO-2 + CPU optimizer offload (the only validated production config on A100). |
| `tests/test_action_mapping.py` *(NEW)* | 7 round-trip tests (null, inventory, jump+camera, hotbar+attack, etc.). |
| `tests/smoke_qwen3_5.py` *(NEW)* | End-to-end load + resize + forward+backward smoke test. |
| `tests/inspect_checkpoint.py` *(NEW)* | Generate actions on real valid-split samples and classify output. |

---

## Known gotchas

Twelve things that have bitten us. The first ten are from the original handoff; 11–12 surfaced during the H200 port.

1. **`setup.py` is broken under modern setuptools** (deprecated `pkg_resources`). Use `PYTHONPATH` instead. Fix is a small `pyproject.toml` migration.
2. **`--torch_dtype` and `--max_seq_length` aren't recognized by TRL ≥1.0.** Stay on TRL 0.16.0 or rename to `--dtype`/`--max_length` and update `train.py`.
3. **TRL 0.16 transitively imports `mergekit` → `networkx`** at module load. If you see `ImportError: cannot import name 'Mapping' from 'collections'`, upgrade networkx.
4. **The RichProgressCallback import in train.py is lazy** for the same reason (TRL pulls in `judges` → `llm_blender` → big deps when `TRL_USE_RICH=1`).
5. **DeepSpeed 0.19.x has a broken Muon optimizer import** (`NameError: BaseMuonWithAuxAdam is not defined`). Pin to 0.16.3.
6. **ZeRO-3 + grad checkpointing + frozen vision tower** fails with a recompute shape mismatch (params get partitioned to shape 0 during recompute). Use ZeRO-2 + CPU optimizer offload (`configs/deepspeed_config_s2_offload.json`).
7. **Without CPU optimizer offload on 80GB cards: OOM in optimizer step.** The flattened gradient allreduce buffer is ~11 GiB extra; we have ~4 GiB headroom without offload. The offload moves the 36 GB optimizer states to host RAM.
8. **MTP for Qwen3.5-9B is vLLM-side only**, not separately trainable heads. `resize_aux_heads` correctly reports 0 candidates. No training-side wiring needed; the speedup comes from `--speculative-config qwen3_next_mtp` at serve time.
9. **Thinking mode** has no model-config flag; pass `enable_thinking=False` to `generate()` or via vLLM serve kwargs. (The Qwen3.5 chat template hardcodes the closed `<think>\n\n</think>\n\n` for assistant turns with existing content, so training and inference structures match when inference uses `enable_thinking=False`. See [Debugging history](#debugging-history) for the full analysis.)
10. **HF dataset cache must be on a large, user-writable disk** — the default `~/.cache/huggingface` will fill the root partition when downloading the source minecraft-vla-sft dataset (216 parquet files, several GB each). Set `HF_HOME`.
11. **Hopper GPUs (H100/H200) + fla 0.5.0**: `fla/ops/common/chunk_o.py::chunk_bwd_dqkwg` raises a hard `RuntimeError` when Triton ≥ 3.4.0 is detected ("Triton >= 3.4.0 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg; install tilelang"). But tilelang requires CUDA 12.8+ headers (E8M0 FP8 intrinsics) and won't compile against CUDA 12.6. The workaround that gets the backward pass to run on H200 + CUDA 12.6:
    - `pip install 'triton==3.3.1'` (overrides torch 2.12's pin to 3.7.0; pip warns, install + runtime work)
    - `pip uninstall -y tilelang`
    Reviewer concern: Triton 3.3.1 predates some Hopper-specific kernel tuning, and could miscompile WGMMA/TMA paths used by gated DeltaNet. Validate with a numerical sanity check against a known-good reference (e.g., 3× A100) before any long run. If CUDA can be upgraded to 12.8, prefer that and revisit Triton 3.7 + tilelang.
12. **vLLM clobbers the SFT env.** Current vLLM (0.21) wants to downgrade torch 2.12 → 2.11, change triton 3.3.1 → 3.6, and pull cu13 nvidia packages — would break the SFT pipeline. Install vLLM in a separate venv for serving; do not let it touch the SFT env's torch/triton/cuda.

---

## Debugging history

Notes on bugs found post-Phase-1-foundation-refactor, mostly from the 2026-05-22 smoke + inspector + multi-LLM review session. Kept here so future sessions don't re-do the same analyses.

### Critical: image double-rescale in `data_collator.py`

**Symptom.** 3000-step smoke trained cleanly (loss 18 → 5.3, no spikes), but the resulting checkpoint emitted zero action tokens at inference — only English-language thinking text or short nonsense. Diagnosing the inference path turned up no bug.

**Cause.** `data_collator.py::image_process` called `transforms.ToTensor()` on the PIL image before passing it to the HF processor. `ToTensor` scales `uint8 [0,255]` → `float [0,1]`. The HF processor then re-applied its default `do_rescale=True` (divide by 255), squashing pixel values to `[0, 0.004]`, then mean/std normalized to `~-1` for every pixel. Result: the model trained on effectively-blank images and learned to predict actions from text/positional priors alone, with no visual signal.

**Verified empirically.** Feeding a gray PIL image `(128,128,128)` through both paths:
- PIL → processor: normalized pixel value `+0.0039` (correct: `(128/255 - 0.5) / 0.5 ≈ 0.004`)
- ToTensor → processor: normalized pixel value `-0.9961` (wrong: double rescale)

**Fix.** Remove `transforms.ToTensor()` from `image_process` and return the PIL image. The HF processor handles tensor conversion + rescale + normalize correctly when given a raw PIL.

### Important: LR too low for newly-added action-token rows

**Symptom.** Same smoke run as above; loss plateaued at ~5.3 from step ~1000 onward despite cosine LR schedule still being significant for the first ~2000 steps. Loss 5.3 is between log(73) ≈ 4.3 (uniform-over-action-tokens) and log(248K) ≈ 12.4 (uniform-over-full-vocab), suggesting partial learning of "some action token goes here" but no confident pick.

**Cause.** The 73 new action-token rows added to `embed_tokens.weight` and `lm_head.weight` are randomly initialized (via `mean_resizing` in `resize_token_embeddings`). A global LR of `3e-6` over 3000 steps is too low by orders of magnitude to pull randomly-initialized weights out of the noise floor. The logits for the pretrained vocabulary dominate, and greedy decoding picks something other than `<|act_start|>` at the first generation position.

**Fix.** Added `DifferentialLRTrainer` (subclass of `transformers.Trainer`) that puts `embed_tokens` and `lm_head` matrices in their own optimizer param group with `EMBED_LR` (default `30 × args.learning_rate`). This is coarser than per-row LR (the pretrained rows in the same matrix also get the higher LR), but it's enough to unblock the new rows and acceptable on the short SFT timeline.

### Important: training data uses Qwen2-VL action tokens, not Qwen3.5

**Symptom.** After the image+LR fixes brought smoke loss to 0.66 (vs the broken 5.78), the inspector still classified the output as MALFORMED — zero `<|act_start|>` tokens in the generation. But the raw generated text showed perfect 4-chunk action structure.

**Cause.** The chunked preprocessor (`scripts/preprocess_chunked_actions.py`) emits action segments in the **Qwen2-VL schema** (`<|reserved_special_token_178|>` … `<|reserved_special_token_179|>` and reserved-slot group tokens). The data_collator passes those strings through `apply_chat_template` and tokenizes them as-is — there is no translation step from Qwen2-VL → Qwen3.5 schema before tokenization. So even though training runs with `--backbone=qwen3_5` (which adds `<|act_start|>` etc. to the vocab), the labels the model sees are still Qwen2-VL reserved-token IDs. The model correctly learns to predict those IDs at inference. The inspector's `_classify_output` checks against the Qwen3.5 schema action-token IDs and finds none, hence MALFORMED.

**Verified empirically.** Running both schemas through the same generated output:
- `qwen3_5` schema: 0 chunks_opened, 0 chunks_closed, 0/24 tokens in grammar
- `qwen2_vl` schema: 4 chunks_opened, 4 chunks_closed, 20/24 tokens in grammar (the remaining 4 are newlines + `<|im_end|>`)

**Status.** Inspector now reports both schemas and tells you which one parses (commit `819b835`). **Production inference will hit the same mismatch** — `agent_wrapper.py` builds its action tokenizer with the model's `--backbone` (qwen3_5), but the model emits qwen2_vl tokens. Two ways to fix it permanently:

1. **Re-preprocess data with the qwen3_5 schema** — modify `preprocess_chunked_actions.py` to render action segments using `QWEN3_5_SCHEMA.act_start / .group_tokens` instead of the qwen2_vl reserved-slot strings. Requires retraining (~hours).
2. **Add a token-ID translation step to the data_collator** — after tokenization, swap qwen2_vl reserved-slot IDs for the corresponding qwen3_5-added IDs. Lighter-weight; can be a regression test rather than a re-render.

Pick after the headline-eval gate is established, since the current data + model are internally consistent — they just need `agent_wrapper` to be configured with `backbone=qwen2_vl` (or to support both at parse time) to function in production.

### Investigated and ruled out: chat template thinking-mode mismatch

**Suspicion.** Hypothesized that training (no `enable_thinking` kwarg) and inference (handoff says pass `enable_thinking=False`) might render structurally different templates, causing the inference malformed output.

**Verified by rendering the template both ways on the same example.** For assistant turns with existing content, the Qwen3.5 chat template line 101 hardcodes the closed `<think>\n\n</think>\n\n` wrapper regardless of the `enable_thinking` kwarg. So training-time and inference-time-with-`enable_thinking=False` produce byte-identical structure. Inference with `enable_thinking` unset (default True) renders an open `<think>\n` that the model never saw — a real mismatch — but the handoff already prescribes `enable_thinking=False` to avoid this.

**Verdict.** Not a bug. No collator change needed. The two real bugs were image preprocessing and LR.

### Multi-LLM review findings (2026-05-22 — `runs/reviews/20260522-064614/`)

Three frontier reviewers (Opus 4.7, GPT-5.5 Pro, Gemini 3.1 Pro) reviewed the smoke run + scripts in parallel via `scripts/review.py`. Consensus: do **not** launch the 4-hour medium run until cheap diagnostics confirm bugs are fixed. Top findings beyond the two fixed above:

- **Label mask may supervise mostly boilerplate** (Opus #1). Most unmasked positions are `<think>\n\n</think>\n\n` + delimiters + `<|im_end|>`, not action tokens. Loss 5.3 could be "boilerplate well-learned + action tokens essentially not learned" weighted average. Diagnostic: per-token-class loss breakdown on a valid batch.
- **Inspector prompt may not byte-match training prompt** (Opus #4, GPT-5.5 #1). Different `apply_chat_template` callsite (`processor.` vs `processor.tokenizer.`); content order `[image, text]` vs source `[text, image]`. Render both and diff at token-id level.
- **Triton 3.3.1 may miscompile gated DeltaNet on Hopper** (Gemini #4 strong, Opus #3 medium, GPT-5.5 mild). Run a deterministic forward numerical comparison against a known-good config before any long run.
- **Frozen visual adapter blocks the projector from adapting to the new action vocab** (Gemini #3). Consider `--fix_visual_adapter False` for the medium/full run.
- **Memory math tight in vanilla DDP** (Opus #3). 9.4B + Adam states ≈ 150 GB; on 2× H200 144 GB we're only fitting because vision tower is frozen. ZeRO-2 (no offload) would add comfortable headroom.
- **Warmup config ambiguous** — `--warmup_ratio 0.03` and `--warmup_steps 100` both set; one overrides the other. Drop one.
- **Cosine-to-zero compresses learning into first ~30% of steps**. For short smoke runs, consider `constant_with_warmup` or a nonzero min LR.
- **`save_steps=1000` × 51 GB/checkpoint × 3-checkpoint rotation = 153 GB disk pressure**. Use `--save_only_model True` for the medium run to skip optimizer state in checkpoint writes.

The cheap diagnostic suite (~1–2 hr total) that the reviewers converged on:

1. Render-and-diff: training prompt vs inference prompt, byte-by-byte.
2. Label-mask audit: dump labels for one batch with `check=True`; confirm action token positions are supervised.
3. Per-token-class loss breakdown: separate CE for `<|act_start|>`, action groups, delimiters, boilerplate.
4. Next-token rank probe: rank of `<|act_start|>` at the first generation position.
5. Micro-overfit: 32–128 samples × 500–1000 steps; should overfit greedy. Failure to overfit = structural bug remains.
6. Numerical sanity for fla on Triton 3.3.1.

---

## After SFT lands

Order of operations once a checkpoint exists and the inspector reports `PARSEABLE`:

1. **vLLM serve smoke test** — `bash scripts/inference/serve_vllm_qwen3_5.sh` with `MODEL_PATH=/path/to/checkpoint`. Confirm `qwen3_next_mtp` initializes. (Requires vLLM in a separate venv per [gotcha 12](#known-gotchas).)
2. **Single-episode rollout** — `python -m jarvisvla.evaluate.evaluate --workers 0 --checkpoints /path/to/checkpoint --base-url http://localhost:9052/v1`. Look for action-decode warnings and step smoothness.
3. **Headline eval** — multi-seed (`--seed-base 0`, `--workers 5+`) on the existing eval config suite; compare success rate + p50 step latency vs the Qwen2-VL 7B baseline. Phase-1 gate is "matches or exceeds baseline" — see REIMPLEMENTATION_PLAN.md §Step 8.
4. **If gate passes**: start Phase 2 (Gemma-4-26B-A4B planner + predicate-based sub-goal queue).
5. **If gate doesn't pass**: bisect via the recipe knobs in REIMPLEMENTATION_PLAN.md §Step 8 (LR sweep, unfreeze vision encoder, vision-token budget 256 → 512).

---

## Citation

```bibtex
@article{li2025jarvisvla,
  title   = {JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse},
  author  = {Muyao Li and Zihao Wang and Kaichen He and Xiaojian Ma and Yitao Liang},
  journal = {arXiv preprint arXiv:2503.16365},
  year    = {2025}
}
```
