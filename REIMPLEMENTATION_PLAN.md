# JARVIS-VLA Reimplementation Plan

## Goals

- **Phase 1**: Swap the Qwen2-VL 7B backbone for **`Qwen/Qwen3.5-9B`** (multimodal despite the name — see Pinned SKUs below) while keeping the action vocabulary and rollout loop intact. Use Qwen3.5-9B's native MTP (`qwen3_next_mtp` in vLLM) for decode speedup. Disable thinking mode for the executor (added latency, no benefit inside a 50ms env-step budget).
- **Phase 2**: Add **`google/gemma-4-26B-A4B-it`** (sparse MoE, 3.8B activated / 25.2B total, Apache 2.0, native tool/function calling) as a slower planner that supplements the fast VLA on long-horizon goals. The base VLA stays the policy; the planner emits structured sub-goals with predicate-based success criteria and uses tool calls for deterministic recipe lookups.

## Pinned SKUs (verified)

| Role | HF repo | Modality | Params | Notable |
| --- | --- | --- | --- | --- |
| Executor | `Qwen/Qwen3.5-9B` | text + image + video, tool-calling | 9B (hybrid Gated-DeltaNet + Gated-Attention; not standard transformer) | Vocab **248,320** (≠ Qwen2-VL's 151,936 — all hard-coded IDs will break). Native MTP via vLLM `--speculative-config '{"method":"qwen3_next_mtp",...}'`. Thinking mode on by default — disable via `enable_thinking=False`. 262K native context. |
| Planner | `google/gemma-4-26B-A4B-it` | text + image | 25.2B total / 3.8B active (8/128 experts + 1 shared) | Apache 2.0. Native tool/function calling. 256K context. ~550M vision encoder. Built-in `<\|think\|>` mode. |

## Assumptions

1. **No backbone fallback.** Qwen3.5-9B is committed; if it underperforms we iterate the recipe (data mix, freeze schedule, vision-token budget, thinking-mode A/B), not swap families.
2. **Hybrid attention is the biggest unknown.** Qwen3.5-9B layout is `3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)` × 8 blocks. Linear-attention layers do not have a standard KV cache, which affects (a) vLLM serving footprint, (b) flash-attention-2 applicability per-layer, and (c) any speed estimate based on dense-9B baselines. Re-baseline early in Week 1 with a no-train forward pass before trusting any latency math elsewhere in this plan.
3. **Dataset stays put.** Reuse `CraftJarvis/JarvisVLA-Qwen2-VL-7B` as the SFT corpus — **but the assistant targets must be re-rendered as N-action chunks** if we want chunking to give smooth playback (see Step 5). Image bytes / conversation schema do not change.
4. **All action-token IDs and chat-template IDs are wrong and must be regenerated.** This is now load-bearing. Qwen3.5-9B's 248,320 vocab is structurally different from Qwen2-VL's 151,936; do not smoke-test, regenerate programmatically at startup. See [§ Step 1](#step-1--programmatic-action-token-mapping).

## Current architecture (what we are replacing)

Read top-down so the touchpoints are easy to find:

| Concern | File | What's Qwen-specific |
| --- | --- | --- |
| Model & processor load | `jarvisvla/train/train.py:35-36, 89-100` | Hard-imports `Qwen2VLProcessor`, `Qwen2VLForConditionalGeneration`, branches on `'qwen2_vl' in model_name` |
| Special tokens | `assets/special_token.json` + `jarvisvla/__init__.py:3` | Uses Qwen's `<|reserved_special_token_NNN|>` slots |
| Action-token ↔ id table | `jarvisvla/inference/action_mapping.py:89-256` | Hard-codes 12 action groups against Qwen IDs 151833–151907 |
| User-mask templates | `jarvisvla/train/data_collator.py:99-110` | `user_template_token = [151644, 872]` (= `<\|im_start\|>user\n`), assistant template likewise |
| Image preprocessing | `jarvisvla/train/data_collator.py:99-110`, `jarvisvla/inference/processor_wrapper.py:149-156` | Hard-codes `image_factor=28`, Qwen's `min_pixels`/`max_pixels`, smart-resize policy |
| Inference backbone routing | `jarvisvla/inference/load_model.py:9-21` | Only knows the string `"qwen2_vl"` |
| vLLM serve | `scripts/inference/serve_vllm.sh` | Points at a Qwen2-VL checkpoint |
| Train shell | `scripts/train/vla_qwen2_vl_7b_sft*.sh` | Paths + DeepSpeed config tuned for 7B |
| Rollout agent | `jarvisvla/evaluate/agent_wrapper.py:71-80, 273-276` | Branches on `LLM_backbone == "qwen2_vl"` to re-tokenize vLLM output |

Everything downstream of `action_tokenizer.decode(...)` (mc env step, callbacks, ray rollout fan-out) is model-agnostic and stays as-is.

---

## Phase 1 — Replace the backbone

Order matters; each step builds on the previous one. Aim is for the Qwen3.5-VL branch to land **alongside** the existing Qwen2-VL branch so we can A/B the two on the same eval suite, then delete the Qwen2-VL path once Qwen3.5-VL matches or beats it.

### Step 1 — Programmatic action-token mapping

The hard-coded ID tables in `jarvisvla/inference/action_mapping.py` (`map_control_token` lines 89-256, `remap_control_token` lines 189-233, `tag_token` lines 236-256) are wrong for Qwen3.5-9B's 248,320-token vocab. Don't patch the numbers — replace the table with a startup-time lookup. This was the #1 finding from all three frontier-model reviewers and is the blocker for everything else in Phase 1.

**Concrete refactor:**

1. Define the action-token strings (not IDs) in a single asset, e.g. `assets/action_tokens.yaml`:
   ```yaml
   act_start: "<|act_start|>"
   act_end:   "<|act_end|>"
   groups:
     hotbar:        ["<|action_hotbar_0|>", ..., "<|action_hotbar_9|>"]
     fore_back:     ["<|action_fb_null|>", "<|action_forward|>", "<|action_back|>"]
     ...  # 12 groups, sizes [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21]
   ```
2. At process startup (after the tokenizer is loaded *and* any `add_special_tokens` has run), build the two maps:
   ```python
   token_to_action = {}   # int id -> (group_idx, value_idx)
   action_to_token = {}   # (group_idx, value_idx) -> int id
   for g_idx, group in enumerate(groups):
       for v_idx, tok_str in enumerate(group):
           tid = tokenizer.convert_tokens_to_ids(tok_str)
           assert tid != tokenizer.unk_token_id, f"action token {tok_str!r} not in tokenizer"
           token_to_action[tid] = (g_idx, v_idx)
           action_to_token[(g_idx, v_idx)] = tid
   assert len(token_to_action) == sum(map(len, groups)), "duplicate IDs in action map"
   ```
3. Save `action_token_map.json` next to the checkpoint at save-time. Load it at inference-time and **assert it round-trips against the loaded tokenizer**. This catches tokenizer/checkpoint drift loudly.
4. Plumb `token_to_action` / `action_to_token` through `OneActionTokenizer` instead of the literal dicts. `OneActionTokenizer.__init__` already takes a `tokenizer_type` — extend it to take the prebuilt maps.

**Choosing the strings vs reusing reserved slots:** Qwen3.5-9B's tokenizer may or may not expose reserved-special-token slots we can hijack. Default to *adding* the action tokens via `tokenizer.add_special_tokens({"additional_special_tokens": [...]})` — this is portable across any backbone. The embedding-resize concern is now handled explicitly in Step 2.

**Also fix while you're in `action_mapping.py`:**

- **`group_action_2_token` (`action_mapping.py:439-458`) silently drops the inventory-flag group.** It writes `group_action[:-4]` for buttons and `group_action[-2:]` for camera, skipping groups -4 and -3 entirely. The decode side (`token_2_group_action`) still reads inventory tokens. This is a crafting blocker independent of the backbone swap and was caught by GPT-5.5-pro. Fix the slice to emit the inventory flag, add a round-trip test (`encode_action(decode(...))` for the null + inventory-open + jump+camera cases).
- **`remap_control_token` returns `(-1, -1)` on unknown tokens (`action_mapping.py:233`).** Silent decode failure. Add a `strict=False` parameter; default `strict=True` in training/validation, allow `strict=False` only inside the rollout loop where we have to ship some action.
- **`map_control_token` mixes lists and tuples (`action_mapping.py:165`).** Standardize to tuples.

### Step 2 — Add the Qwen3.5-9B backbone branch

Add an explicit `--backbone {qwen2_vl,qwen3_5}` CLI flag to `train.py`. Stop inferring backbone from `model_name_or_path.lower().replace('-','_')` (it won't normalize `Qwen3.5-9B` to anything matchable, per GPT-5.5).

`jarvisvla/inference/load_model.py` returns `(LLM_backbone, VLM_backbone)` keyed on the `--backbone` arg.

In `train.py`, add an `elif backbone == 'qwen3_5':` branch that:

- Loads `AutoProcessor.from_pretrained(...)` and `AutoModelForImageTextToText.from_pretrained(...)`. Pin the transformers version that ships Qwen3.5-9B.
- Calls `n_added = tokenizer.add_special_tokens(...)` (Step 1 action tokens + existing point/visual/think/grounding specials).
- **Calls `model.resize_token_embeddings(len(tokenizer))` if `n_added > 0`** — this is currently missing from the codebase entirely. Asserted by all three reviewers as a P0 latent bug that would crash on any vocab extension. Also call `model.tie_weights()` afterward if tied.
- **Walks MTP heads explicitly.** `resize_token_embeddings` only touches the main `lm_head`. Qwen3.5-9B's MTP heads are separate output projections. After resize, iterate any module whose name matches `*mtp*` / `*next_token*` and resize their `nn.Linear` output to the new vocab size. Write a helper `_resize_aux_heads(model, new_vocab_size)` and call it unconditionally — silent OOB at inference is the failure mode if we skip this.
- **Disables thinking mode** in the model config (or via `enable_thinking=False` in `generate` kwargs). Chain-of-thought inside every env step blows the latency budget; we want raw action tokens.
- Sets `attn_implementation="flash_attention_2"` — but verify this doesn't error on the Gated-DeltaNet (linear-attention) layers. If FA2 only applies to the Gated-Attention blocks, transformers will likely route correctly per-layer, but confirm with a no-train forward pass.
- After everything is loaded, calls `processor.save_pretrained(output_dir)` at save-time *in addition to* `trainer.save_model(...)` — `Trainer` alone does not persist the processor (GPT-5.5 finding). Without this, inference reloads the wrong processor.
- After training, sets `model.config.use_cache = True` before the final save. The current code clears it at line 186 for training and never restores it (GPT-5.5 finding).

`train.py:182` (`if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))`) crashes on first run if `output_dir` does not exist (Gemini finding). Wrap with `if Path(training_args.output_dir).exists() and ...`.

The Qwen2-VL-specific freeze regexes at `train.py:107-125` won't match Qwen3.5-9B's parameter names. Build the freeze patterns from the backbone branch and **assert `sum(1 for n,_ in model.named_parameters() if re.match(pat, n)) > 0` for each pattern** — silent zero-match means full fine-tune of what we thought we froze (GPT-5.5 finding).

### Step 3 — Image preprocessing (latency-driven, not fidelity-driven)

**Vision prefill is the latency bottleneck, not decode** (consensus from all three reviewers; Gemini's TTFT analysis is the most concrete). At default Qwen processor settings, image-token count can exceed 1000 per frame, pushing TTFT past 100ms on H100. At a 50ms env-step budget this kills the whole real-time story.

Action:

- **Aggressively cap `max_pixels`** in `vla_qwen3_5_9b_sft.sh` and the inference `processor` config. Start with `max_pixels = 256 * 28 * 28` (~256 vision tokens) and A/B against `512 * 28 * 28` if inventory-text legibility regresses. Do not inherit the existing 7B run's defaults — they were not latency-constrained.
- Verify Qwen3.5-9B's image-processor knobs in the actual model card — patch/merge factor, recommended bounds, accepted image-tensor format. Don't assume Qwen2-VL parity (Opus + GPT-5.5 finding; Qwen3-VL uses different patch handling per the model docs).
- `DataAugment.image_resize` (`jarvisvla/train/data_collator.py:706-709`) — add a `"qwen3_5"` branch. If knobs differ, factor out the smart-resize logic; otherwise fold into the existing `"qwen2_vl"` branch.
- `ProcessorWrapper` (`jarvisvla/inference/processor_wrapper.py:149-156`) — same treatment.
- **Confirm vLLM prefix-caching covers the image tokens across consecutive env steps.** At high image-token counts, vLLM often does not cache vision prefill (Opus finding) — if it doesn't, we re-pay ViT cost every step regardless of chunking. Measure on day 1 of Week 3.
- `point_process` math is correct against variable resolutions; no change needed.

### Step 4 — Chat template & user-prompt masking

`data_collator.py` masks loss over user turns by searching for literal Qwen template token IDs `[151644, 872]` / `[151644, 77091]`. For Qwen3.5-9B these are wrong (vocab size differs, template layout may differ). The reviewers flagged two ways the obvious fix can also be wrong:

- **Opus**: rendering an empty assistant message through `apply_chat_template(...)` can produce a *different* tokenization than the same role header appears mid-conversation, because BPE merges at the empty-content boundary. The derived IDs won't match the real tokens.
- **GPT-5.5**: HF transformers exposes `apply_chat_template(..., return_assistant_tokens_mask=True)` on tokenizers whose template uses the `{% generation %}` block — this returns the exact assistant-token mask directly and sidesteps the whole subsequence-search problem.

**Plan:**

1. First try `apply_chat_template(..., return_assistant_tokens_mask=True)`. If Qwen3.5-9B's template supports it (check by inspecting the `chat_template` for `{% generation %}`), use the returned mask directly — no template-prefix search needed.
2. Fallback: tokenize a real 2-turn conversation with non-empty content, find the assistant-header subsequence in the resulting ids, cache it per backbone. Do *not* use the empty-content trick.
3. **Assert loudly on zero matches.** The current code (`data_collator.py:228-232`) `continue`s silently when `len_beg_matches == 0`, which means loss leaks over user prompts and image encodings — and the run keeps going. Replace with `raise RuntimeError(...)` in `--check` mode, hard-warn in production. (Gemini + GPT-5.5 finding.)
4. Also fix `data_collator.py:105-106`: `ValueError(f"{model_path} is not support")` is *constructed but never raised* — `ValueError(...)` without `raise` is a no-op (GPT-5.5 finding).
5. Add a non-overlap filter to the contiguous-match search (`data_collator.py:259`). Qwen3 templates can have repeated subsequences that produce spurious beg/end pairs (Opus finding).
6. **Delete or fix `apply_private_conversations` (`data_collator.py:296`).** It shadows the input `conversations` arg with `conversations = []`, references undefined `image_count`, and would be activated by any non-`qwen2_vl` path that doesn't use `apply_chat_template` directly (Opus + GPT-5.5 finding). Either delete it (the qwen3_5 path will use `apply_chat_template` like the qwen2_vl path) or fix the shadowing and undefined name.

Verify with `--check` mode that masking lights up only assistant content on a small sample.

### Step 5 — Training scripts, configs, and chunked-action SFT

**Major design change from the previous draft of this plan**: the SFT targets need to be **N-action assistant turns**, not single actions. If we train on one action per assistant turn and try to chunk at inference, the model has never seen the `<|act_end|><|act_start|>` boundary mid-generation and will mis-predict (GPT-5.5 finding). Pick `N` once and bake it into the data render *and* the inference `action_chunk_len`.

**Re-rendering the dataset:**

- `CraftJarvis/JarvisVLA-Qwen2-VL-7B` ships with one action per assistant turn. Write a one-off preprocessor that groups consecutive `(observation, action)` pairs into `(observation, [action_1..action_N])` and re-emits the conversation. Default `N=4` (200ms budget per request at 20Hz).
- Cache the re-rendered dataset to a new HF dataset name, e.g. `JarvisVLA-Qwen3-5-9B-sft-chunk4`. Keep the chunk size in the name so it's obvious which checkpoint pairs with which inference cadence.

**Training script `scripts/train/vla_qwen3_5_9b_sft.sh`:**

- `--backbone qwen3_5` (explicit, per Step 2).
- `--learning_rate 3e-6` starting point (a touch below the 5e-6 used for Qwen2-VL 7B). LR sweep before the full-data run.
- `--max_seq_length` retuned against chunked targets + Qwen3.5's per-image token cost at our chosen `max_pixels`. Likely 1024–2048.
- `--per_device_train_batch_size` retuned for hybrid-attention memory profile (linear-attention layers reduce activation memory vs dense — we may get a larger batch than a 9B dense would suggest, but verify).
- **Keep** `--fix_visual_encoder True` / `--fix_visual_adapter True` as the starting recipe. A/B against unfreezing once we have a baseline. Budget a buffer week here — same-family lift from 7B → 9B is not guaranteed on this narrow control task (Opus finding); have a bisection plan (vision-encoder unfreeze first, then LR sweep, then vision-token budget).
- **Decide MTP-or-not before starting full-data SFT** (Opus + GPT-5.5 finding). Native MTP requires the training loss to include the MTP heads; spec-decoding is purely an inference change. They are not interchangeable at the last minute. Default to **MTP on** (Qwen3.5-9B ships heads, we want the speedup). Surface as a `--mtp-loss-weight` flag with default = Qwen's pretraining value; bump 2–4× only if Step-7 acceptance is low.
- **MTP + new vocab + Trainer compatibility.** Verify the HF `Trainer` forward path actually invokes Qwen3.5-9B's MTP loss. Some HF integrations expose it via `model_kwargs`, some require a custom `compute_loss` override (GPT-5.5 finding). Confirm by checking that grad norms on the MTP head parameters are nonzero after one step.
- **Lower the rollout temperature.** `agent_wrapper.py:55` uses `temperature=0.5`, `agent_wrapper.py:284` uses `top_p=0.99`. For action-token decoding this hurts validity and MTP acceptance — drop to `temperature=0.1` (or greedy) for the eval loop. Keep the higher temperature as a separate config for any exploration-style sampling we want to study.

DeepSpeed: 9B + activations + flash-attn fits ZeRO-2 on 80GB; reuse `configs/deepspeed_config_s2.json`. Hybrid attention may shift memory profile — re-verify after the first overfit run.

### Step 6 — Inference path

**Stop re-tokenizing vLLM string output** (Gemini + GPT-5.5 finding). `agent_wrapper.py:273-276` takes the vLLM completion *as a string*, runs `self.tokenizer(outputs)["input_ids"]`, then decodes actions. Round-tripping reserved special tokens through string form is brittle: tokenizers can drop BOS, add whitespace, or re-merge BPE differently. Switch to vLLM's `Completion` API with `logprobs=True` (or the `token_ids` field on chat completions) to retrieve raw token IDs from the engine directly. The current `extra_body={"skip_special_tokens": False, ...}` config (`agent_wrapper.py:283`) keeps specials in the string, but the safe path is to bypass string entirely.

- Update `scripts/inference/serve_vllm.sh` to serve `Qwen/Qwen3.5-9B` (after SFT, the checkpoint dir) with:
  ```bash
  vllm serve <ckpt> \
      --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
      --max-model-len <chunk_seq_budget> \
      --enable-prefix-caching
  ```
  Pin the vLLM version that supports `qwen3_next_mtp`.
- `agent_wrapper.py:71-80` only re-tokenizes when `LLM_backbone in {"llama-3","llama-2","qwen2_vl"}`. Generalize to "always re-tokenize if `self.tokenizer is not None`," but also wire up the no-string-roundtrip path above.
- `agent_wrapper.py:76-80` reads a hard path `ultron/model/assets/special_token.json` — leftover from another repo, file doesn't exist here. Delete (specials are added at save-time in `construct.py`).
- `agent_wrapper.py:88`: replace the `len(self.tokenizer) == 151657` magic number with `len(self.tokenizer) < expected_size_with_specials` (Opus finding).
- **Cap `max_tokens` to expected per-chunk action-token count + a small slack** (GPT-5.5 finding). Current `max_tokens=1024` (`agent_wrapper.py:282`) is hostile to p99 latency — if anything goes wrong the model can hallucinate up to 1024 tokens before we get our action. For `N=4` chunks of ~10 tokens each, set `max_tokens=64` and add a stop condition on `<|act_end|>` occurring `N` times.
- **Action chunking semantics:** when `len_action < self.action_chunk_len` (model emitted fewer actions than requested due to early EOS), the current code silently truncates. Log this case and either repeat the last valid action or fall back to the null action (Opus finding).
- **Async pipelining.** Without it, env stalls every `chunk_len` steps on the next inference call. Kick off generation for chunk N+1 in a background thread/task while executing chunk N. The OpenAI client already supports async — wire `agent_wrapper.forward` to return immediately from a cached `self.actions` queue and refill the queue in background. This is the difference between "smooth playback" and "burst-and-stall" (Opus finding).
- **Mandatory action-token logit constraint** (not optional, per GPT-5.5 finding). At inference: when the last emitted token is `<|act_start|>`, mask all non-action tokens; when inside an action group, mask tokens outside that group's allowed range; at `<|act_end|>` allow the start of the next action or EOS. vLLM supports this via guided decoding or a custom logits processor.
- `jarvisvla/inference/construct.py:11-39` (`apply_full_model`) — add a `qwen3_5` branch using `AutoProcessor`. Persist the `action_token_map.json` from Step 1 next to the processor config so inference loads the same mapping training used.

### Step 7 — MTP / inference speedup

**Updated 2026-05-21 after smoke test against `/ephemeral/models/Qwen3.5-9B`:** the earlier
draft of this section assumed Qwen3.5-9B exposes separately-trained MTP head modules à la
DeepSeek-V3. **It does not.** `model.named_modules()` contains no `mtp/next_token/draft`
candidates, and `resize_aux_heads` correctly reports 0 modules to resize. The
`qwen3_next_mtp` speedup is implemented **entirely inside vLLM** as a speculative-decoding
mode that uses Qwen3.5-9B's own hidden states. There is no training-side wiring required
for it — `resize_aux_heads` and `disable_thinking_mode` stay in `utils_train.py` as
defensive infrastructure for future backbones, but they're no-ops for Qwen3.5-9B.

**Thinking mode** is similarly not a model config field — neither `model.config` nor
`model.generation_config` has any `think*` attribute on Qwen3.5-9B. Pass
`--enable-thinking false` (or the equivalent vLLM serve kwarg) at serve time, not at
training time.

Qwen3.5-9B native MTP via vLLM:
```
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```
This is the primary path. Logit constraints (now moved to Step 6 as mandatory) and async pipelining are complementary.

**Inference-side tuning** (no training-side change required):

- Tune `num_speculative_tokens` (k) by measured acceptance rate. Start with k=2; bump to
  k=3 only if second-token acceptance ≥80% on a 100-episode eval (otherwise you pay for
  drafts you throw away).
- Below 40–50% second-token acceptance, MTP isn't paying for itself — fall back to plain
  decode and chase the speedup via larger `action_chunk_len` instead.

**Vision prefill caveat:** MTP only accelerates decode. The action stream is short (~10 tokens/chunk), so even ideal MTP saves at most ~15–25ms per inference. If vision prefill is 80–150ms (Step 3), MTP is *not* the dominant lever — chunk size and prefix-cache hit rate are. Don't over-invest in MTP tuning if Step 3 measurements show vision-prefill-dominated latency.

Document the chosen MTP config in the training script with a `# Speedup:` comment so the SFT artifact and the inference config stay in sync.

### Step 8 — Validation gates before declaring Phase 1 done

**Startup tests** (must all pass before any training run):

- Action-token round-trip: every `(group_idx, value_idx)` → `tokenizer.convert_tokens_to_ids(...)` → back round-trips, no `unk`, no duplicates.
- `encode_action(decode(action_tokens))` round-trips for: null action, inventory-open, jump+camera, hotbar+attack. **This catches the `group_action_2_token` inventory-flag bug from Step 1.**
- Collator `--check` on 16 samples: assistant-mask covers assistant content only, no leakage onto user/image regions.
- Single forward+backward step with an example containing the max-ID action token (catches missing `resize_token_embeddings` and missing MTP-head resize — both surface here, not later).
- vLLM raw-output round-trip: serve the un-finetuned base, generate, confirm raw `token_ids` from the engine match what re-tokenizing the string would produce. If they don't, the no-string-roundtrip change from Step 6 is the only correct path.

**Training-time:**

- One-epoch overfit on 1024 samples: train loss < 0.05, eval loss tracks.
- After step 1: grad norm on MTP head parameters is nonzero. If it's zero, `Trainer` isn't exercising MTP loss — fix before continuing.

**Rollout:**

- A single rollout completes end-to-end without action-decode warnings.
- Invalid-action rate (decoder returns `(-1, -1)` for some token) is <1% over a 200-step rollout.

**Headline eval (with multi-seed determinism — Opus finding):**

- Run `multi_evaluate` with `--temperature 0.1` (not 0.5), **≥5 seeds per config**, report mean and std. Single-seed comparisons with the current temperature settings have too much variance to trust a 5pp difference.
- Success rate **matches or exceeds** the Qwen2-VL 7B baseline at the mean. Same-family 7B → 9B lift is *not* guaranteed on this narrow control task — flat is acceptable; regressed needs a recipe fix before declaring done.
- p50 time-per-action **at least 1.3× faster** than the Qwen2-VL 7B baseline at chunk_len matching the SFT chunk size. If MTP doesn't deliver and the limiter is vision prefix-caching, fix that before declaring done.

---

## Phase 2 — Planner-guided architecture

Premise: the Qwen2-VL VLA already does *reactive* control well but composes weakly over long horizons (open inventory → identify missing reagents → mine them → return → craft). We keep the fast VLA as the executor and add a slower planner that emits sub-goals.

### Architectural shape

Three patterns considered. Based on reviewer feedback (Opus + GPT-5.5 + Gemini all converged here), **skip naive Pattern A and go straight to Pattern B with predicate-based success criteria.**

| Pattern | What the planner emits | VLA input change | Verdict |
| --- | --- | --- | --- |
| **A. Sub-goal text injection** | A short natural-language sub-goal | Sub-goal replaces `instruction` field | **Not enough.** The Qwen2-VL VLA was trained on high-level goals ("craft iron pickaxe"); injecting fine-grained planner sub-goals into the `instruction` field is OOD and the executor will likely hallucinate (Gemini finding). Also, the planner has no completion signal. |
| **B. Predicate-augmented sub-goal queue** | `{"subgoal": "...", "success_predicate": {...}, "abort_after_steps": N, "original_goal": "..."}` | Executor sees `"Original goal: X. Current subgoal: Y."` (do not drop the original — GPT-5.5 finding) | **Recommended.** Predicates evaluated against MineStudio's inventory/state dicts, so no second VLM call needed for completion detection — sidesteps the visual-checker problem. |
| **C. Latent-conditioning** | A latent vector concatenated to VLA's prompt embeddings | Requires re-training VLA with conditioning | Defer. Only revisit if Pattern B sub-goal text empirically leaks information the executor can't parse (fine-grained spatial, e.g.). |

**Pattern B sub-goal completion is the actually-hard piece** (Opus finding). The plan in the previous draft handwaved "when reward signals or vision-side checkers fire" — neither works reliably at sub-goal scale (MineStudio rewards are sparse; vision checkers means another VLM call). Concrete fix: every planner-emitted sub-goal includes an explicit success predicate expressed against env-introspectable state:

```json
{
  "subgoal": "collect 3 cobblestone",
  "success_predicate": {"inventory_has": "cobblestone", "count": 3},
  "abort_after_steps": 600,
  "original_goal": "craft an iron pickaxe"
}
```

The `subgoal_state` module evaluates the predicate against the MineStudio info dict each step. No vision checker, no second LLM call. Stall detection is trivial (sub-goal not satisfied within `abort_after_steps`).

**Tool-calls for deterministic recipe lookups** (GPT-5.5 finding, enabled by Gemma-4-26B-A4B's native function calling — verified in Pinned SKUs). The planner should never hallucinate Minecraft crafting trees when the deterministic recipe graph is sitting in `jarvisvla/evaluate/assets/recipes/`. Expose a `lookup_recipe(item_name)` tool to the planner; it returns the ingredient list and grid pattern from the existing recipe JSONs (already used by `create_recipe_prompt_from_library` at `agent_wrapper.py:109`).

### Planner choice

**`google/gemma-4-26B-A4B-it`** (verified Apache 2.0, 3.8B active / 25.2B total, multimodal, native tool calling, 256K context). Reasons:

- Strong reasoning quality at this size (per your direct observation), which is the whole point.
- Native tool/function calling enables the recipe-lookup pattern above without prompt-engineering hacks.
- Activated-param count keeps planner inference cost comparable to the executor's per-call cost — cadence math stays sane.
- Apache 2.0 — no license risk if we publish the planner-augmented agent (Qwen3.5-9B's license is more restrictive; if publishing weights matters, the executor is the constrained piece, not the planner).
- Different family from the Qwen3.5 executor → uncorrelated failure modes (a vision-tower weakness on dense GUI text won't hit both). Trade-off: no shared tokenizer for debugging.

**Memory caveat (GPT-5.5 finding):** "4B activated" does not mean 4B memory. We still store all 25.2B weights. Colocating with Qwen3.5-9B on one 80GB H100 is tight without quantization — plan to serve the planner on a separate GPU or run the planner in fp8/int8. The existing OpenAI-compatible client path (`agent_wrapper.py:59-64`) makes a separate-host setup trivial.

Operational details:

- Call cadence: **every N frames** (start with N=40 = ~2s at 20Hz), **on sub-goal completion**, and **on stall** (no reward + low VLA action-entropy for K frames).
- Inputs to planner: current POV image, last 3 POVs as a strip (for short memory), inventory text (the env already exposes this), the user's original instruction, the current plan / sub-goal stack.
- Outputs: JSON `{plan: [...], next_subgoal: "...", abort_conditions: [...]}`. Validate with a schema; on parse failure, repeat the previous sub-goal.
- Hosting: serve with vLLM on a separate port from the executor. If we need cross-host, the OpenAI-compatible API the existing code already speaks (see `agent_wrapper.py:59-64`) makes that swap trivial.

### Code integration points

- New module `jarvisvla/planning/` with:
  - `planner.py`: client (OpenAI/Anthropic/Gemini compatible) + prompt assembly + JSON-schema validation.
  - `subgoal_state.py`: small state machine that owns the sub-goal queue, decides when to re-plan, and exposes the current sub-goal to the agent.
- `agent_wrapper.VLLM_AGENT.forward`: when `self.planner is not None`, replace `instructions[0]` with the current sub-goal from `subgoal_state`. The VLA itself is unchanged.
- `evaluate.evaluate`: instantiate a `Planner` if `model_config["planner"]` is set; pass into `VLLM_AGENT`.
- New eval config knobs: `--planner-model`, `--planner-cadence`, `--planner-base-url`.

### Phase-2 validation

- **Smoke**: agent with planner solves a 2-step task (mine wood → craft planks) that the no-planner agent can already solve. Confirms we did not regress.
- **Headline**: planner-augmented agent solves a curated long-horizon task set (e.g. craft iron pickaxe from spawn, brew a potion) that the no-planner agent fails >80% of the time. Target ≥30pp absolute lift.
- **Cost / latency**: planner adds <15% wall-clock to a successful episode (because it fires sparingly).

---

## Risks & open questions

- **Hybrid attention is the biggest unknown.** Qwen3.5-9B's Gated-DeltaNet (linear attention) layers don't have a standard KV cache. This affects vLLM serving footprint, FA2 applicability per-layer, and any speed estimate based on dense-9B baselines. Re-baseline with a no-train forward pass before trusting Step 7 latency math.
- **MTP heads + new vocab + Trainer integration.** Three things have to align: (1) `resize_token_embeddings` extends the main `lm_head`, (2) `_resize_aux_heads` extends every MTP head's output projection, (3) `Trainer.compute_loss` actually exercises the MTP loss. Verify nonzero grad norms on MTP head parameters after one training step. Most likely failure mode is silently training only the main head and discovering at inference that MTP acceptance is at chance.
- **vLLM `qwen3_next_mtp` maturity.** This is a relatively new vLLM feature. Pin the version that ships it; have a plain-decode fallback ready. Loss of speedup is acceptable if functionality is intact.
- **9B serving footprint with hybrid attention.** Per-action latency budget against the 50ms env-step (or 200ms chunk budget at chunk_len=4) must be re-verified. Step 3 vision-prefill cost is the dominant term; if it doesn't fit, drop `max_pixels` further or accept lower playback rate.
- **Re-render cost.** Re-rendering the SFT dataset as N-action chunks is a one-off but non-trivial preprocessor. Budget half a week for this in Week 1 — it's a prerequisite for any chunked inference to actually be smooth.
- **Eval determinism.** Single-seed eval at `temperature=0.5` (current code) gives high run-to-run variance. We can't tell a 5pp regression from noise. Multi-seed (≥5) at lower temperature is mandatory for the Phase-1 gate.
- **License.** Qwen3.5-9B's license should be re-checked before publishing weights. Gemma-4-26B-A4B is Apache 2.0 (verified) so the planner is unconstrained.
- **Planner cost in CI.** Even self-hosted, the 25B-total / 4B-active planner adds eval-suite wall time *and* GPU memory. Cache planner responses keyed on `(image-hash, original-goal, current-subgoal, inventory-summary)` so re-runs are cheap. Plan for a separate GPU or fp8/int8 colocation.
- **Cross-family executor/planner.** Qwen executor + Gemma planner means no correlated failure modes (good) but also no shared tokenizer. Planner output is sub-goal text, so cross-family is fine in practice.

## Existing bugs to fix while we're here

These were caught by the reviewers in the existing Qwen2-VL code path. They're independent of the backbone swap but will bite the moment we touch the surrounding code, so fix them in Week 1.

- **`action_mapping.py:439-458`** — `group_action_2_token` silently drops the inventory-flag group. Crafting blocker. Fix + round-trip test. *(GPT-5.5)*
- **`data_collator.py:105-106`** — `ValueError(...)` constructed but not raised. *(GPT-5.5)*
- **`data_collator.py:296`** (`apply_private_conversations`) — shadows the input `conversations` arg with `conversations = []`, references undefined `image_count`. *(Opus + GPT-5.5)*
- **`agent_wrapper.py:76-80`** — hard path to `ultron/model/assets/special_token.json`, file doesn't exist in this repo. *(All three)*
- **`agent_wrapper.py:88`** — magic number `151657`. *(Opus)*
- **`train.py:182`** — glob on `output_dir` crashes if the directory doesn't exist yet. *(Gemini)*
- **`train.py:215`** — `trainer.save_model()` doesn't save the processor; call `processor.save_pretrained(output_dir)` explicitly. *(GPT-5.5)*
- **`train.py:186`** — `model.config.use_cache=False` is set for training but never reset to `True` before the inference save. *(GPT-5.5)*
- **`agent_wrapper.py:273-276`** — re-tokenizing vLLM string output is fragile (BPE merge / whitespace / BOS issues). Use raw `token_ids` from the engine. *(Gemini + GPT-5.5)*

---

## Suggested milestones

1. **Week 1** — *Cleanup & scaffolding.*
   - Fix the "Existing bugs" list (most importantly the `group_action_2_token` inventory-flag bug — without this, crafting tasks are silently broken regardless of backbone).
   - Replace `action_mapping.py` hard-coded tables with the programmatic Step-1 lookup.
   - Add `--backbone` CLI flag + Qwen3.5-9B branch in `train.py` and `load_model.py`.
   - Add `resize_token_embeddings` + `_resize_aux_heads` helper.
   - Re-baseline the chat-template masking with `return_assistant_tokens_mask` (or the 2-turn fallback).
   - **Hybrid-attention smoke test:** load Qwen3.5-9B unmodified, run a no-train forward+backward on one batch with FA2. Measure per-layer attn-impl routing.
2. **Week 2** — *Data + first SFT.*
   - Build the chunked-action dataset preprocessor; cache `JarvisVLA-Qwen3-5-9B-sft-chunk4`.
   - First Qwen3.5-9B SFT on a 5% slice with chunk_len=4. Verify masking, image processor, MTP grad norms nonzero, save/load roundtrip, action-token round-trip on the saved checkpoint.
3. **Week 3** — *Full SFT + serve.*
   - Full-data Qwen3.5-9B SFT.
   - vLLM serve with `qwen3_next_mtp`. Measure MTP acceptance rate (k=2, then k=3).
   - Measure vision-prefix-cache hit rate across consecutive env steps. If it's low, fix before headline eval.
   - Wire up async pipelining + mandatory logit constraint + capped `max_tokens`.
4. **Week 4** — *Headline eval. **Phase 1 gate.***
   - Multi-seed eval (≥5 seeds, temperature=0.1) vs Qwen2-VL 7B baseline.
   - Decision: ship Phase 1, or one buffer week for recipe iteration (LR sweep, vision-encoder unfreeze, vision-token budget A/B).
5. **Week 5** — *Planner serve + Pattern B scaffold.*
   - Deploy `gemma-4-26B-A4B-it` (separate GPU or fp8-quantized colocation).
   - Implement `subgoal_state` with predicate evaluation against MineStudio info dict.
   - Wire `lookup_recipe(item_name)` tool against `evaluate/assets/recipes/`.
6. **Week 6** — *Planner integration.*
   - JSON-schema-validated planner outputs.
   - Executor sees `"Original goal: X. Current subgoal: Y."` (not just the subgoal).
   - Re-plan triggers: predicate met → next subgoal; predicate not met within `abort_after_steps` → re-plan.
   - Response cache keyed on `(image-hash, original-goal, current-subgoal, inventory-summary)`.
7. **Week 7** — *Long-horizon eval. **Phase 2 gate.***
   - Curated long-horizon task set (iron pickaxe from spawn, brew potion).
   - Target: ≥30pp absolute lift over no-planner agent.
   - Cost: planner adds <15% wall-clock to successful episodes.
