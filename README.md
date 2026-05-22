# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse

[![arXiv](https://img.shields.io/badge/arXiv-2503.16365-df2a2a.svg?style=for-the-badge)](https://arxiv.org/pdf/2503.16365)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/collections/CraftJarvis/jarvis-vla-v1-67dc157a99d011efd7d7f7e4)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Project Website**](https://craftjarvis.github.io/JarvisVLA/) | [**Datasets**](https://huggingface.co/datasets/CraftJarvis/minecraft-vla-sft) 

## Updates

* [2026.05.22] Multi-GPU SFT pipeline working end-to-end on Qwen3.5-9B against the chunked dataset. Phase-1 foundation refactor complete (see [Reimplementation status](#reimplementation-status-in-progress) below).
* [2026.05.21] Started reimplementation of the VLA backbone on **`Qwen/Qwen3.5-9B`** (multimodal). Planner for Phase 2 will be **`google/gemma-4-26B-A4B-it`**. See [REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md) for the full plan.
* [2025.03.21] Our paper can be found in [arXiv](https://arxiv.org/pdf/2503.16365).

## Reimplementation status (in progress)

The next generation of JARVIS-VLA targets:

- **Executor:** `Qwen/Qwen3.5-9B` (9B hybrid Gated-DeltaNet + Gated-Attention, multimodal despite the name). Inference speedup via native MTP through vLLM's `qwen3_next_mtp` speculative-decoding mode (no training-side wiring required).
- **Planner (Phase 2):** `google/gemma-4-26B-A4B-it` (sparse MoE, 3.8B activated of 25.2B, Apache 2.0, native tool calling). Emits sub-goals with predicate-based success criteria evaluated against MineStudio's info dict — no second VLM call needed for completion detection.

### What's landed (Phase 1 foundation)

- **Programmatic action-token mapping** (`jarvisvla/inference/action_tokens.py`) — schemas defined for Qwen2-VL (backward-compat via the reserved-special-token slot strings) and Qwen3.5 (canonical `<|act_*|>` names added via `add_special_tokens`). IDs resolved against the live tokenizer at startup, persisted next to the checkpoint, and verified against the loaded tokenizer at inference. Replaces the previous hard-coded 168-line ID table in `action_mapping.py` (which assumed Qwen2-VL's 151,936-token vocab and silently broke on the 248,320 of Qwen3.5).
- **Backbone-agnostic training pipeline** (`jarvisvla/train/train.py`) — explicit `--backbone {qwen2_vl,qwen3_5}` flag; the qwen3_5 branch uses `AutoProcessor` + `AutoModelForImageTextToText`; embedding resize and freeze-pattern assertion handled centrally in `utils_train.py`.
- **Chunked-action SFT data preprocessor** (`scripts/preprocess_chunked_actions.py`) — re-renders the source dataset so each assistant turn contains N concatenated action segments (default N=4). Runs sort-then-chunk with a trajectory-level holdout for a real valid split (the source `valid` is single-frame-per-trajectory and can't chunk). On the full train split: 935,440 chunks at 99.1% yield, 100% with exactly 4 action segments.
- **Multi-GPU dry-run SFT** (`scripts/train/vla_qwen3_5_9b_sft_dryrun.sh`) — DeepSpeed launcher across 3 GPUs, ZeRO-2 + gradient checkpointing, sdpa attention (until flash-linear-attention + causal-conv1d are installed for the production fast path).
- **Inference path tightening** (`jarvisvla/evaluate/agent_wrapper.py`) — capped `max_tokens` at `chunk_len × 16` (was 1024), default temperature 0.5 → 0.1, decode-health log on every step.
- **Multi-seed eval support** (`jarvisvla/evaluate/evaluate.py`) — `--seed-base` flag so `--workers N` produces N independent seeds.
- **Existing-code bug fixes** caught during the refactor: `group_action_2_token` was silently dropping the inventory-flag group; `apply_private_conversations` shadowed its input arg; the masking-loop silent-on-zero-matches behavior (loss leaked over user prompts). All covered by regression tests in `tests/`.

### What's pending

- Async pipelining in the rollout loop (kick off chunk N+1 generation while env executes chunk N).
- Full Phase-1 SFT run + headline eval vs the Qwen2-VL 7B baseline.
- Phase 2: Gemma-4-26B-A4B planner + predicate-based sub-goal queue.

See [REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md) for the full design, all touchpoints, and the validation gates.

## Installation
Install dependencies.
```shell
git clone https://github.com/CraftJarvis/JarvisVLA.git
conda create -n mcvla python=3.10
conda activate mcvla
cd JarvisVLA
conda install --channel=conda-forge openjdk=8 -y
pip install -e .
```

After the installation, you can run the following command to check if the installation is successful and the environment is working:

```shell
# After the installation, you can run the following command to check if the installation is successful:
python -m minestudio.simulator.entry # using Xvfb
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry # using VirtualGL
```

## Inference 

You can serve the model with vllm to support multi-GPU and multi-process rollout.
```sh
CUDA_VISIBLE_DEVICES=0 vllm serve jarvis_vla_qwen2_vl_7b_sft --port 8000
```

Then you need to edit the rollout script to the use the correct base_url and port. 
Finally, you can run the rollout script.
```sh
sh scripts/evaluate/rollout-kill.sh
```

## Train

Prepare the dataset and base model, and write their locations in the shell below.

- Single GPU
```shell
sh scripts/vla/vla_qwen2_vl_7b_sft.sh
```
- Multi-GPU
```shell
sh scripts/vla/vla_qwen2_vl_7b_sft-multi-GPU.sh
```
- Multi-Node
```shell
sh scripts/vla/vla_qwen2_vl_7b_sft-multi-node.sh
```

---

### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{li2025jarvisvla,
  title   = {JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse},
  author  = {Muyao Li and Zihao Wang and Kaichen He and Xiaojian Ma and Yitao Liang},
  journal = {arXiv preprint arXiv:2503.16365}, 
  year    = {2025}
}
```
