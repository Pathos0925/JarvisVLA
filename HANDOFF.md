# JARVIS-VLA Phase-1 Reimplementation — Handoff

_Last updated: 2026-05-22._

This document is a snapshot of exactly where Phase 1 stands and what's needed to
pick it up on a new machine (likely a single B200 or 2-4× H100, see §Hardware).

If you're new to the project, read [REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md)
first for the design rationale; this doc is operational status only.

## Status at a glance

| Stage | Status |
| --- | --- |
| Plan + design review | ✅ done ([REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md), reviewer output in `runs/reviews/` if kept) |
| Foundation refactor (action tokens, --backbone, helpers) | ✅ done, committed `65d0b2b` |
| Smoke test against real Qwen3.5-9B | ✅ done (tests/smoke_qwen3_5.py) |
| Chunked-action SFT preprocessor | ✅ done; 935,440 chunks at 99.1% yield |
| Multi-GPU SFT dry-run (5 steps) | ✅ done; loss 10.51 → 9.76 ([commit `0ee489b`](#)) |
| **Full Phase-1 SFT run** | ⏳ **next** — script ready (`scripts/train/vla_qwen3_5_9b_sft.sh`), needs to be launched on hardware |
| vLLM serve + smoke rollout | ⏳ pending SFT |
| Headline eval vs Qwen2-VL baseline | ⏳ pending SFT |
| Async pipelining in agent_wrapper | ⏳ task #10, deferred |
| Phase 2 (Gemma planner + sub-goal queue) | ⏳ pending Phase-1 gate |

## What I'd do first on the new machine

```bash
# 1. Clone + activate env (Python 3.11 was used here)
git clone https://github.com/Pathos0925/JarvisVLA.git
cd JarvisVLA

# 2. Set env vars for caches (point everything at a large disk)
export HF_HOME=/path/to/large/disk/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTHONPATH=$(pwd)  # setup.py is broken; use PYTHONPATH

# 3. Install Python deps (see §Pinned deps for the exact versions that worked)
pip install -r requirements.txt
pip install accelerate mergekit 'trl==0.16.0' 'deepspeed==0.16.3'
pip install --no-build-isolation causal-conv1d flash-linear-attention flash-attn
pip install -U networkx  # mergekit pulls an ancient networkx; the upgrade is safe

# 4. Download model + dataset (~20 GB + ~25 GB)
hf download Qwen/Qwen3.5-9B --local-dir /path/to/models/Qwen3.5-9B
# Or regenerate the chunked dataset (preferred — proves the preprocessor still works):
python scripts/preprocess_chunked_actions.py --chunk-len 4 \
    --output-dir /path/to/datasets/jarvisvla-chunk4 \
    --holdout-fraction 0.01 --splits train

# 5. Verify the foundation still works on this machine
python -m tests.test_action_mapping          # expect: 7/7 passed
python -m tests.smoke_qwen3_5                # expect: forward+backward loss ~18, no errors

# 6. Launch the Phase-1 SFT (see §Launching SFT for tuning)
BASE_MODEL_PATH=/path/to/models/Qwen3.5-9B \
DATASET_NAME=/path/to/datasets/jarvisvla-chunk4 \
OUTPUT_DIR=/path/to/checkpoints/qwen3_5_9b_phase1 \
nohup bash scripts/train/vla_qwen3_5_9b_sft.sh > sft.log 2>&1 &
```

## Hardware

The current 3× A100 80GB rig works but is constrained. From measured dry-run
(`vla_qwen3_5_9b_sft_dryrun.sh`) and architectural extrapolation:

| Setup | Estimated step time | Production-ready? |
| --- | --- | --- |
| 3× A100 80GB (current) | 6.7 s/step | Yes, with ZeRO-2 + CPU optimizer offload |
| 2× H100 80GB | ~3 s/step | Yes, same config works, less host RAM needed |
| 1× H100 80GB | ~3.5 s/step | Marginal — barely fits without offload at seq=1024 |
| **1× B200 (192 GB)** | **~1.5 s/step** | **Best** — fits comfortably without ZeRO, simpler script |

On a B200, the production config can be drastically simplified:
- Drop DeepSpeed entirely (single GPU, no NCCL).
- No CPU optimizer offload needed (192 GB fits everything).
- Can probably bump `--per_device_train_batch_size 4` or higher.
- FP8 via transformer engine could give another 1.5–2× on the matmul-heavy parts
  (would need a quick A/B to confirm convergence isn't hurt).

If you stay on A100/H100 80GB, **keep the current config** — ZeRO-2 + CPU offload
is the validated recipe.

## Pinned deps (what actually worked)

| Package | Version | Note |
| --- | --- | --- |
| python | 3.11 | conda env named `myenv` here |
| torch | 2.8.0+cu128 | from PyTorch index, CUDA 12.8 |
| transformers | 5.9.0 | newer; introduces `processing_class` arg, handled by train.py at runtime |
| trl | **0.16.0** | **Pinned** — 0.12 lacks `trl.scripts`; ≥1.0 renames `max_seq_length` → `max_length` |
| deepspeed | **0.16.3** | **Pinned** — 0.19+ has a broken Muon optimizer import |
| accelerate | latest | needed for `device_map` |
| mergekit | latest | TRL 0.16 imports it transitively |
| networkx | ≥3.0 | mergekit pulls an ancient one (Python-2 `from collections import Mapping`); upgrade fixes it |
| flash-attn | 2.8.3 | install with `--no-build-isolation` |
| flash-linear-attention | 0.5.0 | for Qwen3.5-9B's gated DeltaNet layers |
| causal-conv1d | 1.6.2.post1 | needed by fla |
| datasets, pyarrow, openai, vllm, etc. | latest | regular deps |

`--no-build-isolation` is required for the CUDA-compiled wheels because pip's
build env grabs a newer PyTorch with a different CUDA version, which fails the
extension build.

`setup.py` uses deprecated `pkg_resources` and will fail under modern setuptools;
**use `PYTHONPATH` instead of `pip install -e .`**. (Fixing setup.py is a small
follow-up — convert to `pyproject.toml`.)

## Launching SFT

The production script is `scripts/train/vla_qwen3_5_9b_sft.sh`. It's parametrized
via env vars; defaults are sane for the 3× A100 setup. Key knobs:

```bash
# 6-hour smoke SFT on 1% of data (~10K samples, ~3000 micro-steps)
DATASET_P=0.01 MAX_STEPS=3000 bash scripts/train/vla_qwen3_5_9b_sft.sh

# 24-hour run on 12% slice (~111K samples)
DATASET_P=0.12 bash scripts/train/vla_qwen3_5_9b_sft.sh

# Full epoch (only attempt with B200 or many H100s; ~24 days on 3× A100)
bash scripts/train/vla_qwen3_5_9b_sft.sh

# Single-GPU on a B200 (override deepspeed launch — see §B200 simplification)
```

All env-var overrides supported:
- `BASE_MODEL_PATH` (default `/ephemeral/models/Qwen3.5-9B`)
- `DATASET_NAME` (default `/ephemeral/datasets/jarvisvla-chunk4`)
- `OUTPUT_DIR` (default `/ephemeral/checkpoints/<run_tag>`)
- `CUDA_VISIBLE_DEVICES` (default `0,1,2`)
- `BATCH` (default 1 — only raise on H100/B200)
- `GRAD_ACCUM` (default 1)
- `DATASET_P`, `MAX_STEPS` for scope control
- `EPOCH` (default 1)
- `RUN_TAG` (default `$(date +%Y%m%d-%H%M%S)`)

### B200 simplification

On a single B200, replace the DeepSpeed launcher with plain `python`:

```bash
# 1. Drop the --deepspeed flag (no sharding needed)
# 2. Use accelerate or torchrun for single-GPU + AMP, or just python jarvisvla/train/train.py
# 3. Raise BATCH=4 (or higher — VRAM permitting) and GRAD_ACCUM=1
```

A clean B200 script would be ~20 lines. Worth writing once you're on the
hardware so the diff is grounded in real memory measurements.

## Known gotchas / things that bit me

1. **`setup.py` is broken under modern setuptools** (deprecated `pkg_resources`).
   Use `PYTHONPATH` instead. Fix is a small `pyproject.toml` migration.
2. **`--torch_dtype` and `--max_seq_length` aren't recognized by TRL ≥1.0.** Stay
   on TRL 0.16.0 or rename to `--dtype`/`--max_length` and update `train.py`.
3. **TRL 0.16 transitively imports `mergekit` → `networkx`** at module load. If
   you see `ImportError: cannot import name 'Mapping' from 'collections'`,
   upgrade networkx.
4. **The RichProgressCallback import in train.py is lazy** for the same reason
   (TRL pulls in `judges` → `llm_blender` → big deps when `TRL_USE_RICH=1`).
5. **DeepSpeed 0.19.x has a broken Muon optimizer import** (`NameError:
   BaseMuonWithAuxAdam is not defined`). Pin to 0.16.3.
6. **ZeRO-3 + grad checkpointing + frozen vision tower** fails with a recompute
   shape mismatch (params get partitioned to shape 0 during recompute). Use
   ZeRO-2 + CPU optimizer offload (`configs/deepspeed_config_s2_offload.json`)
   instead.
7. **Without CPU optimizer offload on 80GB cards: OOM in optimizer step.** The
   flattened gradient allreduce buffer is ~11 GiB extra; we have ~4 GiB headroom
   without offload. The offload moves the 36 GB optimizer states to host RAM.
8. **MTP for Qwen3.5-9B is vLLM-side only**, not separately trainable heads.
   `resize_aux_heads` correctly reports 0 candidates. No training-side wiring
   needed; the speedup comes from `--speculative-config qwen3_next_mtp` at serve
   time.
9. **Thinking mode** has no model-config flag; pass `enable_thinking=False` to
   `generate()` or via vLLM serve kwargs.
10. **HF dataset cache must be on a large disk** — the default `~/.cache/huggingface`
    will fill the root partition when downloading the source minecraft-vla-sft
    dataset (216 parquet files, several GB each). Set `HF_HOME`.

## What changed in the code (one-line summary per file)

| File | Change |
| --- | --- |
| `jarvisvla/inference/action_tokens.py` (NEW) | Per-backbone schemas + `build_id_maps` (resolves action-token strings to IDs at runtime instead of hard-coding them) |
| `jarvisvla/inference/action_mapping.py` | `OneActionTokenizer.from_tokenizer(...)`, no more module-level dispatch functions; **fix**: inventory-flag bug in `group_action_2_token` |
| `jarvisvla/inference/load_model.py` | Accepts explicit `backbone=` override; recognizes `qwen3_5` |
| `jarvisvla/train/train.py` | `--backbone` flag; `qwen3_5` branch; embedding+aux-head resize; disable thinking mode; assert freeze patterns; processor save_pretrained; use_cache restore; processing_class/tokenizer compat shim |
| `jarvisvla/train/utils_train.py` | `MoreConfig.backbone`; `resize_aux_heads`, `disable_thinking_mode`, `assert_freeze_patterns_match` helpers |
| `jarvisvla/train/data_collator.py` | Accepts `backbone=`; chat-template prefix IDs derived dynamically; loud assertion on zero matches; removed broken `apply_private_conversations` |
| `jarvisvla/evaluate/agent_wrapper.py` | `from_tokenizer`-based action_tokenizer; max_tokens capped; temp 0.5→0.1; decode-health log |
| `jarvisvla/evaluate/evaluate.py` | `--seed-base` flag (per-worker seed = base + worker_id) |
| `scripts/preprocess_chunked_actions.py` (NEW) | Sort-then-chunk preprocessor with trajectory-level holdout |
| `scripts/verify_chunked_dataset.py` (NEW) | Structural verification of chunked output |
| `scripts/train/vla_qwen3_5_9b_sft.sh` (NEW) | Production SFT script (ZeRO-2 offload + FA2 + fla) |
| `scripts/train/vla_qwen3_5_9b_sft_dryrun.sh` (NEW) | Bounded 5-step dry-run |
| `scripts/inference/serve_vllm_qwen3_5.sh` (NEW) | vLLM serve with `qwen3_next_mtp` |
| `scripts/review.py` (NEW) | Multi-LLM plan review via OpenRouter |
| `configs/deepspeed_config_s2_offload.json` (NEW) | ZeRO-2 + CPU optimizer offload (the only validated production config) |
| `tests/test_action_mapping.py` (NEW) | 7 round-trip tests (null, inventory, jump+camera, hotbar+attack, etc.) |
| `tests/smoke_qwen3_5.py` (NEW) | End-to-end load + resize + forward+backward smoke test |

## After SFT lands

Order of operations once a checkpoint exists:

1. **vLLM serve smoke test** — `bash scripts/inference/serve_vllm_qwen3_5.sh`
   with `MODEL_PATH=/path/to/checkpoint`. Confirm `qwen3_next_mtp` initializes.
2. **Single-episode rollout** — `python -m jarvisvla.evaluate.evaluate
   --workers 0 --checkpoints /path/to/checkpoint --base-url
   http://localhost:9052/v1`. Looks for action-decode warnings, smoothness.
3. **Headline eval** — multi-seed (`--seed-base 0`, `--workers 5+`) on the
   existing config suite; compare success rate + p50 step latency to the
   Qwen2-VL 7B baseline. Phase-1 gate is "matches or exceeds baseline" — see
   REIMPLEMENTATION_PLAN.md §Step 8.
4. **If gate passes**: start Phase 2 (Gemma-4-26B-A4B planner).
5. **If gate doesn't pass**: bisect via the recipe knobs called out in the plan
   (LR sweep, unfreeze vision encoder, vision-token budget 256 → 512).

## Quick contact map

- **Plan / design rationale**: `REIMPLEMENTATION_PLAN.md`
- **Frontier-LLM critique that shaped Step 1**: `runs/reviews/20260521-221419/` (if you kept it; git-ignored by default)
- **Memory entries from prior sessions**: `/home/ubuntu/.claude/projects/-ephemeral-JarvisVLA/memory/`

That's it. The codebase is in a state where the next session can pick up at
"launch the SFT and watch it train" without re-doing any of the setup work.
