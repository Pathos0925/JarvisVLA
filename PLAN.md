# JARVIS-VLA — Project Plan: Toward Playing (and Beating) Minecraft

> Supersedes the old `REIMPLEMENTATION_PLAN.md` (deleted; provenance in git history). This is
> the single strategic plan. **Operational "what to run / what's validated / what bites"** lives
> in [README.md](README.md) — this doc does not duplicate it. Frontier-LLM critiques are under
> `runs/reviews/`; the ROCKET-3 paper is `rocket3.pdf` in repo root.
>
> Status date: **2026-05-28.**

---

## 0. Thesis (what the research says, and why our architecture is right)

A deep multi-source survey of 2024–2026 Minecraft agents (full evidence in
[§8](#8-evidence-appendix--deep-research-2024-2026)) converges on one structure:

**SOTA long-horizon Minecraft = a slow MLLM *planner* that decomposes goals into sub-goals,
driving a fast visuomotor *controller* that executes each from pixels.** Pure end-to-end RL
(VPT, DreamerV3) never broke `ObtainDiamond`; the hierarchical planner+controller pattern
did (GITM, then Optimus-1/2/3). **That is exactly the JARVIS-VLA executor + planner design** —
so the architecture is validated by the field, not speculative.

Two sober calibrations from the same survey:

- **A bigger planner does not rescue a weak goal interface or a reactive-only executor.** The
  frontier (ROCKET-1/2/3, same lab as this repo) shows *language* sub-goals fail for occluded /
  spatial targets, and that **RL post-training (ROCKET-3)** is what converts an imitation policy
  into in-world competence (4× interaction success).
- **Reality check on "beat the game":** the best *open* agent reaches only **~15% on full
  `ObtainDiamond`-from-spawn** (Optimus-3); **no surveyed open agent beats the Ender Dragon.**
  Beating the game needs *both* a robust planner *and* RL compute beyond 2× H200. Almost all
  published numbers are single-source / self-reported — treat them as directional.

**Net:** keep building the executor+planner, and prioritize the two cheapest high-leverage
wins the research identifies — (1) the planner + training-free memory + recipe-lookup, then
(2) KL-anchored RL post-training of the executor — before the expensive frontier move (a
visual goal channel, which needs an executor retrain).

---

## 1. Current status (2026-05-28)

| Stage | Status |
| --- | --- |
| Phase-1 foundation refactor (action tokens, `--backbone`, helpers, R2, inspector) | ✅ done — see [§3](#3-whats-landed-phase-1-foundation) |
| Chunked-action SFT dataset | ✅ built — **935,440 chunks (926,099 train + 9,341 valid)**, deterministic md5 holdout, 99.1% yield (counts verified against the parquet files) |
| Phase-1 full SFT (Qwen3.5-9B) | ⏳ **running** — resumed from `checkpoint-35000 / 115,763` (~30%) on **2× H200** (second box, R2 pull), loss ≈ 0.47, recipe `DATASET_P=1.0 EMBED_LR=3e-5 LIGER=0 SAVE_STEPS=5000`, triton 3.3.1, R2 upload active |
| Mid-train inspector | ✅ checkpoints through 30K classified `PARSEABLE` (4/4 chunks) on the original run — re-run needs `minestudio`; only `checkpoint-35000` is local on this box, so persist verdicts under `runs/` for auditability |
| Phase-1 headline-eval gate (vs Qwen2-VL 7B) | ⏳ pending SFT — [§4.3](#43-remaining-phase-1-gate) |
| vLLM serve + rollout smoke | ⏳ pending SFT (separate env; CUDA 12.8 box now unblocks vLLM) |
| Phase 2 — Gemma planner + sub-goal queue | ⏳ pending Phase-1 gate — [§5](#5-phase-2--planner-guided-agent) |
| Phase 3 — RL post-training of the executor | ⏳ optional, gated on Phase 1+2 — [§6](#6-phase-3--rl-post-training-of-the-executor-rocket-3) |

**Hardware note (resume box):** 2× H200 144 GB, CUDA 12.8 driver. SFT fits without DeepSpeed.
Resume requires the deterministic dataset regen (above) + the full R2 checkpoint (incl.
optimizer state) + a `torch.load(weights_only=False)` / `numpy._core` shim in `train.py` for
cross-version RNG-state loading. Run/monitoring details: README.

---

## 2. Architecture & pinned SKUs

```
User goal ──► Gemma-4-26B-A4B PLANNER (slow, sparse)         ← Phase 2
                │  emits sub-goals:
                │   • symbolic  → predicate vs MineStudio info dict (Pattern B)
                │   • spatial   → visual/cross-view goal (CVGS channel)   ← Phase 2/3
                ▼
              Qwen3.5-9B VLA EXECUTOR (fast, chunked actions)  ← Phase 1 (running)
                │  keyboard+mouse action tokens, chunk_len=4
                ▼
              MineStudio env  ──► info dict (inventory/voxel) ──┐
                                                                └─► predicate check + RL reward (Phase 3)
```

| Role | HF repo | Modality | Params | Notable |
| --- | --- | --- | --- | --- |
| **Executor** | `Qwen/Qwen3.5-9B` | text+image+video, tool-calling | 9B (hybrid Gated-DeltaNet + Gated-Attention; **not** a standard transformer) | Vocab **248,320** (≠ Qwen2-VL's 151,936 — every hard-coded ID breaks). Native MTP via vLLM `--speculative-config '{"method":"qwen3_next_mtp",...}'`. Thinking-mode off via `enable_thinking=False`. 262K ctx. |
| **Planner** | `google/gemma-4-26B-A4B-it` | text+image | 25.2B total / 3.8B active (8/128 experts + 1 shared) | Apache 2.0. Native tool/function calling. 256K ctx. ~550M vision encoder. |

**Committed assumptions:** no backbone fallback (iterate the recipe, don't swap families);
hybrid attention is the biggest serving/latency unknown (re-baseline empirically); reuse the
source SFT corpus but **re-rendered as N-action chunks**; all action/template IDs are
regenerated programmatically at startup, never hard-coded.

---

## 3. What's landed (Phase-1 foundation)

These are **done** (commits `65d0b2b → 8c32ec5` and later; regression-tested):

- **Programmatic action-token mapping** (`jarvisvla/inference/action_tokens.py`) — per-backbone
  schemas + `build_id_maps()` resolving token strings → IDs against the live tokenizer at
  startup, persisted next to the checkpoint, asserted at inference. Replaced the hard-coded
  168-line ID table that assumed Qwen2-VL's vocab.
- **Backbone-agnostic training** (`train.py`) — `--backbone {qwen2_vl,qwen3_5}`; the qwen3_5
  branch uses `AutoProcessor` + `AutoModelForImageTextToText`, embedding/aux-head resize,
  freeze-pattern assertions, thinking-mode disable centralized in `utils_train.py`.
- **`DifferentialLRTrainer`** — separate optimizer group for `embed_tokens` + `lm_head` at
  `EMBED_LR` (full run uses 10× = `3e-5`) so the randomly-initialized new action-token rows
  escape the noise floor without catastrophic forgetting.
- **Chunked-action preprocessor** (`scripts/preprocess_chunked_actions.py`) — sort-then-chunk
  with trajectory-level md5 holdout; N=4 segments/turn.
- **Async R2 checkpoint upload** (`jarvisvla/train/r2_callback.py`) — per-file retries w/
  backoff, idempotent HEAD checks, `wait_all()` flush; auto-sourced from `.env`.
- **Checkpoint inspector** (`tests/inspect_checkpoint.py`) + **HTML report**
  (`scripts/inspect_html.py`) — cheap `PARSEABLE/PARTIAL/MALFORMED` proxy before vLLM.
- **Remote inference bootstrap** (`scripts/inference/r2_fetch_and_serve.sh`) and **schema
  translator** (`scripts/translate_action_schema.py`).

**Bug ledger (old plan's 9-item list): 8 fixed, 1 open.** Verified fixed in code:
`group_action_2_token` inventory-flag drop (crafting blocker); `apply_private_conversations`
arg-shadowing (removed); masking-loop silent-on-zero-matches (now `raise`); `ValueError`
constructed-not-raised; the `special_token.json` hard path (now `from_tokenizer`); the
`151657` magic number; `train.py` glob guard; `processor.save_pretrained` + `use_cache`
restore. **Still open (#9):** the vLLM output path was only *hardened* (`add_special_tokens=False`
on the re-tokenize), **not** switched to raw engine `token_ids` — the brittle string round-trip
remains. Tracked in [§4.3](#43-remaining-phase-1-gate).

---

## 4. Phase 1 — Executor SFT (decisions, status, remaining gate)

### 4.1 Key technical decisions (settled)

- **Action/template IDs regenerated at startup**, asserted to round-trip, saved next to the
  checkpoint. No hard-coded tables. **Loss masking:** prefer
  `apply_chat_template(..., return_assistant_tokens_mask=True)` if the template has a
  `{% generation %}` block; otherwise subsequence-search the assistant header on a *real,
  non-empty 2-turn* conversation (never the empty-content trick — BPE merges at the empty
  boundary differ), with a non-overlap filter for repeated subsequences, and **raise on zero
  matches** (silent zero-match leaks loss over user/image regions). *Current `data_collator.py`
  uses the 2-turn subsequence search, not `return_assistant_tokens_mask`.*
- **Decode contract** (`remap_control_token`): `strict=True` (raise on unknown token) in
  training/validation; `strict=False` (ship null / last action) only inside the rollout loop,
  where we must emit *something*.
- **Vision prefill is the latency bottleneck, not decode.** Cap `max_pixels = 256·28·28`
  (~256 vision tokens); A/B 512 only if inventory text regresses. Confirm vLLM prefix-caching
  covers vision tokens across consecutive env steps (re-pay ViT cost otherwise).
- **Chunked SFT targets** — N=4 actions per assistant turn so the model sees
  `<|act_end|><|act_start|>` boundaries mid-generation; chunk_len matches at inference.
- **MTP is vLLM-side only *for Qwen3.5-9B*.** It exposes no separate MTP head modules
  (`resize_aux_heads` → 0 candidates); the `qwen3_next_mtp` speedup is pure speculative
  decoding at serve time — **no training-side wiring**. Thinking-mode is a serve kwarg
  (`enable_thinking=false`), not a config field. *Caveat for any future backbone that ships
  separate MTP heads:* the three-way alignment must hold — `resize_token_embeddings` (main
  head) + `_resize_aux_heads` (each MTP head) + `Trainer.compute_loss` actually exercising the
  MTP loss (HF may need a custom `compute_loss` override). Verify **nonzero grad norms on MTP
  head params after one step** — that is what `tests/smoke_qwen3_5.py` checks. Silent
  failure mode: training only the main head, discovering at inference that acceptance is at chance.
- **Recipe:** `lr 3e-6` cosine, warmup_ratio 0.03, batch 4 × 2 GPU, seq 1024, frozen vision
  encoder+adapter, `LIGER=0` (continuity), `EMBED_LR=3e-5`. **`fla` 0.5.0 rejects triton ≥ 3.4
  on Hopper (hard error), so we pin triton 3.3.1** — validated by a forward/backward smoke
  (loss ≈ 18, healthy grads). The pin is itself a residual risk ([§7](#7-risks--open-questions)).

### 4.2 Known issue carrying forward — action-schema mismatch

The chunked data encodes actions in the **Qwen2-VL** reserved-token schema, but training runs
`--backbone qwen3_5` (which *adds* `<|act_*|>` tokens). The model therefore learns to emit
Qwen2-VL tokens. **Production inference must set `ACTION_SCHEMA=qwen2_vl`** (env override in
`agent_wrapper.py`) until one of these lands:

1. **Re-preprocess** with the qwen3_5 schema (clean, requires a retrain), or
2. **In-collator token-ID translation** qwen2_vl → qwen3_5 (lighter; add a regression test).

`scripts/translate_action_schema.py` already does the offline conversion. Decide after the
headline-eval gate — the current data + model are internally consistent.

### 4.3 Remaining Phase-1 gate

**Startup checks (before trusting any rollout/latency number):**

- **vLLM raw-output round-trip** — on the un-finetuned base, confirm the engine's raw
  `token_ids` equal what re-tokenizing the *string* output produces. If they diverge for
  Qwen3.5-9B's tokenizer, the string path is unsafe and the raw-`token_ids` bypass below is
  **mandatory**, not optional.
- **FA2 per-layer routing** — confirm `attn_implementation="flash_attention_2"` routes
  correctly on the hybrid stack (FA2 on Gated-Attention blocks; fallback on Gated-DeltaNet)
  via a no-train forward pass, before trusting latency math.

**Gate sequence:**

1. **Inspector** on the final checkpoint → `PARSEABLE` (4/4 chunks). *(through 30K: green)*
2. **vLLM serve smoke** (`qwen3_next_mtp` initializes; separate env — CUDA 12.8 box now
   supports vLLM 0.21). Tune `num_speculative_tokens` (k=2→3) by acceptance rate; fall back to
   plain decode below ~40–50% second-token acceptance.
3. **Single-episode rollout** (`ACTION_SCHEMA=qwen2_vl`). Wire up the inference path that's
   still open from [§3](#3-whats-landed-phase-1-foundation):
   - **Raw `token_ids` from the engine** — consume vLLM's `token_ids`/logprobs directly; stop
     re-tokenizing the string completion (`agent_wrapper.py:314`).
   - **Mandatory action-token logit constraint** (not a tuning knob — it *guarantees* validity):
     at `<|act_start|>` mask all non-action tokens; inside an action group mask tokens outside
     that group's allowed range; at `<|act_end|>` allow the next action's start or EOS.
     Implement via vLLM guided decoding or a custom logits processor. *(Not yet implemented.)*
   - **Cap `max_tokens`** at ~`chunk_len×16` **and** add a stop condition firing when
     `<|act_end|>` has occurred N times, so generation halts at exactly the chunk size.
   - **Chunk-shortfall fallback:** if the model emits fewer than `chunk_len` actions (early
     EOS), log it and repeat the last valid action or emit null — **do not silently under-act**
     (current code only truncates).
   - **Async pipelining:** `agent_wrapper.forward` returns immediately from a cached
     `self.actions` queue and refills it in a background task (generate chunk N+1 while
     executing chunk N). Without this the env stalls every `chunk_len` steps — the difference
     between smooth playback and burst-and-stall, and the whole reason chunking exists.
   - **Pass gate:** single rollout completes with **zero decode warnings** and an
     **invalid-action rate (`(-1,-1)`) < 1% over 200 steps** (what the decode-health log measures).
4. **Headline eval** — **multi-seed (≥5), temperature 0.1**, vs the Qwen2-VL 7B baseline.
   Gate = *matches or exceeds* baseline success rate at the mean (same-family 7B→9B lift is
   not guaranteed; flat is acceptable, regressed needs a recipe fix); p50 time-per-action
   ideally ≥1.3× faster at matching chunk size.
   - **If gate passes →** Phase 2. **If not →** bisect: LR sweep → unfreeze vision encoder →
     vision-token budget 256→512.

---

## 5. Phase 2 — Planner-guided agent

Premise: the VLA does *reactive* control well but composes weakly over long horizons. Keep the
fast VLA as the executor; add the Gemma planner that emits sub-goals. **This is the SOTA
pattern** (Optimus-1: GPT-4V + HDKG memory → STEVE-1; Optimus-2: planner → GOAP controller;
Optimus-3: MoE) — see [§8](#8-evidence-appendix--deep-research-2024-2026).

### 5.1 Sub-goal interface (Pattern B + a visual channel)

| Pattern | Planner emits | Verdict |
| --- | --- | --- |
| A. Sub-goal text injection (replaces `instruction`) | a language sub-goal | **No** — OOD for a VLA trained on high-level goals; no completion signal. |
| **B. Predicate-augmented sub-goal queue** | `{subgoal, success_predicate, abort_after_steps, original_goal}` | **Recommended** — predicates checked against the MineStudio info dict; no second VLM call. |
| C. Latent-conditioning | a latent vector | Defer; revisit only if text leaks info the executor can't parse. |

```json
{"subgoal":"collect 3 cobblestone","success_predicate":{"inventory_has":"cobblestone","count":3},
 "abort_after_steps":600,"original_goal":"craft an iron pickaxe"}
```
`subgoal_state` evaluates the predicate each step (no vision checker, no second LLM). Executor
sees **`"Original goal: X. Current subgoal: Y."`** (keep the original goal). Stall = predicate
unmet within `abort_after_steps` → re-plan.

**Goal-space caveat (ROCKET-1/2/3, `rocket3.pdf`).** Language is the *weak* channel for
spatially-grounded sub-goals: they ran language STEVE-1 through RL → near-zero, while
**Cross-View Goal Specification** (a segmentation mask in a third-person goal image) works
because the executor reasons ego-view ↔ goal-view and exploits shared landmarks even when the
target is occluded. **Recommendation:** Pattern B (language) for *symbolic* sub-goals
(`collect 3 cobblestone`); a **visual goal channel** (goal image + point/mask, CVGS-style) for
*spatial* sub-goals (`approach the village over the ridge`). The visual channel needs an
executor that accepts CVGS goals — *not a drop-in* (re-train / aux-conditioning); ROCKET-2's
executor already does this and could be adopted as the controller beneath the Gemma planner.

### 5.2 Memory + deterministic knowledge (cheap, high-leverage)

The research's clearest cheap win: **training-free memory** (Optimus-1's HDKG/AMEP improved
base MLLMs **2–6×** with *zero* parameter updates). Concretely for us:

- **`lookup_recipe(item_name)` tool** over the existing recipe JSONs
  (`jarvisvla/evaluate/assets/recipes/`, already used by `create_recipe_prompt_from_library`),
  exposed via Gemma's native function calling — the planner never hallucinates crafting trees.
- A lightweight **episodic memory** (what sub-goals were tried/failed, current inventory/biome)
  fed into the planner prompt. No training; pure prompt assembly + state.

### 5.3 Planner choice & ops

`google/gemma-4-26B-A4B-it` — strong reasoning at low active-param cost, native tool calling,
Apache 2.0 (executor is the license-constrained piece, not the planner), different family from
the executor (uncorrelated failure modes). **Memory caveat:** 4B *active* ≠ 4B *resident* — all
25.2B weights are stored; serve on a separate GPU or fp8/int8. The OpenAI-compatible client in
`agent_wrapper.py` makes a separate-host planner trivial.

- **Cadence:** every N≈40 frames (~2s @ 20Hz), on sub-goal completion, and on stall
  (no reward + low action-entropy for K frames).
- **Inputs:** current POV, last-3-POV strip, inventory text, original instruction, sub-goal
  stack. **Output:** JSON `{plan, next_subgoal, abort_conditions}`, schema-validated; on parse
  failure repeat the previous sub-goal.
- **Cache** planner responses on `(image-hash, original-goal, current-subgoal, inventory)`.
- **Integration:** new `jarvisvla/planning/{planner.py, subgoal_state.py}`;
  `agent_wrapper.VLLM_AGENT.forward` swaps `instructions[0]` for the current sub-goal when a
  planner is set (VLA unchanged); `evaluate.evaluate` instantiates it from config; new knobs
  `--planner-model/-cadence/-base-url`.

### 5.4 Phase-2 gate

- **Smoke:** planner agent solves a 2-step task the no-planner agent already solves (no regression).
- **Headline:** solves a curated long-horizon set (iron pickaxe from spawn, brew potion) the
  no-planner agent fails >80% of the time; target **≥30pp** absolute lift.
- **Cost:** planner adds **<15%** wall-clock to a successful episode (fires sparingly).

---

## 6. Phase 3 — RL post-training of the executor (ROCKET-3)

Our pipeline through Phase 1 is **IL/SFT only**. ROCKET-3's foundation-to-finesse result: RL
post-training turns the imitation prior into in-world competence — **4× interaction success
(7%→28%)** + zero-shot transfer. This is the lever for *playing well*, not just matching the
SFT baseline. Load-bearing knobs the paper proves:

- **Initialize *and* KL-anchor from the SFT policy.** `L = L_PPO + β·D_KL(π_θ ‖ π_ref)`,
  `π_ref` = frozen Phase-1 checkpoint. **KL is not optional** — without it the policy collapses
  late; pure RL from scratch failed.
- **Auto task synthesis + auto reward** from the simulator — reuses the same MineStudio
  introspection as the Pattern-B predicates (voxel/inventory deltas; no human reward design).
- **Mixed-difficulty curriculum** (e.g. 20/40/60-block spawn distance, 1:1:1) beats hard-only.
- **Optional aux heads** (target *visibility* + *centroid point*, learned in IL via backward
  relabeling) survive RL unsupervised and give the planner a "target in view / where" signal.
  Our `resize_aux_heads` plumbing is the mechanical hook.

**Scope caveat.** ROCKET-3 is *short-horizon, spatially-grounded interaction*
(approach/break/interact/hunt ≤~60 blocks) — **not** a long-horizon planner. It strengthens the
executor + interface; long-horizon tech-tree progression stays the planner's job (Phase 2).

**Compute caveat.** ROCKET-3's throughput came from a 72-instance / 3-node cluster (~1000 FPS)
with a custom async framework (Ray + shared-NAS index passing, **fragment-based KV-cache +
truncated BPTT** for long-sequence transformer policies — all confirmed in `rocket3.pdf`).
**2× H200 cannot match that** — this is the phase that justifies scaling GPUs. They state
they'll open-source the framework (github.com/CraftJarvis/ROCKET-3) — same lab, likely reusable.

---

## 7. Risks & open questions

- **Hybrid attention** (Gated-DeltaNet has no standard KV cache) — affects vLLM footprint, FA2
  per-layer applicability, every dense-9B-based latency estimate. Re-baseline empirically
  (the FA2-routing check in [§4.3](#43-remaining-phase-1-gate)).
- **Triton 3.3.1 pin on Hopper** — `fla` 0.5.0 forces triton ≤ 3.3.1 today, but 3.3.1 predates
  some Hopper kernel tuning and could miscompile gated DeltaNet. The forward/backward smoke
  passed (loss ≈ 18); for any *long* run, sanity-check against a known-good reference. Prefer
  **triton 3.7 + tilelang once `fla` supports them** (the box is already CUDA 12.8).
- **Action-schema mismatch** ([§4.2](#42-known-issue-carrying-forward--action-schema-mismatch))
  — must be resolved (re-preprocess or in-collator translate) before clean production inference.
- **Inference path still string-based** — raw-`token_ids` bypass + mandatory logit constraint +
  async pipelining are unbuilt (see [§4.3](#43-remaining-phase-1-gate)); without them, validity
  and smooth playback are not guaranteed.
- **vLLM `qwen3_next_mtp` maturity** — pin the version; keep a plain-decode fallback.
- **Eval determinism** — single-seed @ temp 0.5 is too noisy to trust 5pp; multi-seed @ 0.1 is
  mandatory for the gate.
- **Planner cost / colocation** — 25.2B resident; separate GPU or fp8/int8; cache responses.
- **"Beat the game" is unproven open-source** — best open ~15% on full ObtainDiamond-from-spawn,
  no Ender-Dragon clear. Expect this to need a robust planner *and* RL compute past 2× H200.
- **Evidence quality** — most surveyed numbers are single-source/self-reported; verify before
  betting a phase on any single figure (see [§8](#8-evidence-appendix--deep-research-2024-2026) caveats).

---

## 8. Evidence appendix — deep research (2024-2026)

Synthesized + adversarially verified (vote tallies shown). Primary sources cited inline.

1. **Hierarchical planner+controller is SOTA — our exact shape.** Optimus-1 (GPT-4V/HDKG →
   STEVE-1, NeurIPS'24), Optimus-2 (planner → GOAP, CVPR'25), Optimus-3 (MoE, '25).
   *(3-0)* — arXiv 2408.03615, 2502.19902, 2506.10357.
2. **Training-free memory drives long-horizon gains** — HDKG/AMEP, no parameter updates, 2–6×.
   Our analogue: `lookup_recipe` over the recipe JSONs. *(2-1; dissent: rides on a near-zero base)* — 2408.03615.
3. **A small VLM controller is feasible on 2× H200** — Optimus-2 GOAP = DeepSeek-VL-1.3B LoRA,
   2 days on 8× L40; our 9B SFT ≈ 32 h on 2× H200. *(3-0)* — 2502.19902.
4. **JARVIS-VLA's ActVLP** — self-supervised visual+linguistic post-training on *non*-trajectory
   data **before** action SFT → a language-conditioned VLA over 1000+ atomic tasks. *(3-0; a
   circulating "40%" figure was refuted)* — 2503.16365.
5. **Language fails for occluded/spatial sub-goals; cross-view visual masks scale; RL lifts
   interaction 4× (7→28%)** — ROCKET-1 (RGB+SAM-2 masks, CVPR'25), ROCKET-2 (human-view masks +
   cross-view/visibility losses, AAAI'26), ROCKET-3 (PPO+KL, auto task synthesis; the 4× is
   ROCKET-3's). *(3-0, 7 votes — strongest)* — 2410.17856, 2503.02505, 2507.23698.
6. **Visual goal-conditioning's gains are large** — ROCKET-1 0.82 avg over 12 interaction tasks
   (vs VPT-bc 0.07 / STEVE-1 0.19 / GROOT-1 0.09); **25% diamond in a controlled diamond-mining
   *interaction* setup** (vs 2–8%) — *not* full ObtainDiamond-from-spawn; 50% obsidian (vs 0%).
   *(3-0)* — 2410.17856.
7. **LLM planning broke the barrier pure RL couldn't, cheaply** — GITM (background, '23): all
   262 Overworld items (vs DreamerV3 13, VPT 15, DEPS 69), +47.5pp ObtainDiamond, **no GPU
   training** (32-core CPU). Optimus-3: single-model MoE, ~15% full ObtainDiamond. *(medium; single-source)*
   — 2305.17144, 2506.10357.

**Caveats.** 2024–2026 evidence is strongest; GITM is background (non-VPT-native eval). Almost
all numbers are single-source self-reported, **not reproduced**. ROCKET-1's "76%" headline is
the *+Human-prompt* row, not the Molmo 0.82 variant — don't conflate; and its "25% diamond" is
a controlled interaction task, distinct from the ~15% full `ObtainDiamond`-from-spawn figure.
Optimus-3 is mis-described in secondary sources (v2 retitles it "Dual-Router Aligned MoE"; a
STEVE-1/crafting-graph tie was refuted). **No open agent beats the Ender Dragon.** The
cross-view visual channel needs an executor retrain, not a drop-in. DreamerV3 / MineDojo /
Voyager / DEPS / Plan4MC / JARVIS-1 / GROOT / STEVE-1 appear here only as baselines/background.

---

## 9. Roadmap

Phase 1 is largely complete; the near-term critical path is the gate, then the planner.

1. **Now → SFT done** *(in progress)* — finish the resumed full-epoch SFT to step 115,763;
   inspect mid-train checkpoints (`PARSEABLE`). R2 uploads every 5K.
2. **Phase-1 gate** — startup checks (vLLM raw round-trip, FA2 routing) → vLLM serve → single
   rollout (`ACTION_SCHEMA=qwen2_vl`, <1% invalid over 200 steps) → multi-seed headline eval vs
   Qwen2-VL 7B. Resolve the action-schema mismatch
   ([§4.2](#42-known-issue-carrying-forward--action-schema-mismatch)) and the open inference-path
   items ([§4.3](#43-remaining-phase-1-gate)) before/at this point.
3. **Phase 2a — planner scaffold** — deploy Gemma (separate GPU / fp8); `subgoal_state` with
   predicate checks; `lookup_recipe` tool; episodic memory in the prompt.
4. **Phase 2b — integration + gate** — schema-validated planner output, original+subgoal to the
   executor, re-plan triggers, response cache; long-horizon eval (≥30pp lift, <15% cost).
5. **Phase 3 (optional) — RL post-training** — KL-anchored PPO from the SFT reference, MineStudio
   auto-reward, mixed curriculum, optional aux heads; **scale GPUs** for ROCKET-3-style throughput.
6. **Frontier — visual goal channel** — CVGS-style executor (retrain) for spatial sub-goals;
   the documented path past a language-only interface toward genuinely beating the game.
