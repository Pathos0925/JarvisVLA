"""Inspect a JarvisVLA SFT checkpoint: load it, generate an action chunk for a
real validation sample, and report whether the output is well-formed.

Quick sanity signal — no vLLM, no minecraft sim. Answers:
  - Does the model emit <|act_start|> ... <|act_end|> grammar?
  - Are the emitted IDs registered action tokens (not random vocab noise)?
  - How does the predicted action compare to the ground-truth assistant turn?

Run with:
    PYTHONPATH=/workspace/JarvisVLA MODEL_PATH=/workspace/checkpoints/.../checkpoint-3000 \\
        python -m tests.inspect_checkpoint
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from jarvisvla import QWEN_SPECIAL_TOKENS
from jarvisvla.inference import action_tokens

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/workspace/checkpoints/mc-vla-qwen3-5-9b-h200-smoke-e1-b4-a1-p0.01/checkpoint-3000",
)
VALID_SHARD = "/workspace/datasets/jarvisvla-chunk4/valid-00000.parquet"
NUM_SAMPLES = 3
ACTION_CHUNK_LEN = 4
PER_ACTION_TOKEN_BUDGET = 16  # matches agent_wrapper.py


def _hr(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _load():
    _hr(f"Loading checkpoint: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    print(f"loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, vocab {len(processor.tokenizer)}")

    schema = action_tokens.get_schema("qwen3_5")
    maps = action_tokens.build_id_maps(schema, processor.tokenizer)
    print(f"qwen3_5 action tokens: act_beg={maps.act_beg_id} act_end={maps.act_end_id}, "
          f"{len(maps.action_to_token)} action token IDs registered")
    return model, processor, schema, maps


def _action_token_ids(maps):
    """All IDs that count as 'in the action grammar'."""
    return {maps.act_beg_id, maps.act_end_id} | set(maps.action_to_token.values())


def _classify_output(generated_ids, maps, processor):
    """Walk generated IDs, count well-formed chunks vs malformed/extraneous tokens."""
    action_ids = _action_token_ids(maps)
    chunks_closed = 0
    chunks_opened = 0
    inside = False
    non_action = 0
    total = 0
    for tid in generated_ids:
        total += 1
        if tid == maps.act_beg_id:
            chunks_opened += 1
            inside = True
        elif tid == maps.act_end_id:
            if inside:
                chunks_closed += 1
            inside = False
        elif tid in action_ids:
            pass  # action group token, inside a chunk
        else:
            if not (tid == processor.tokenizer.eos_token_id or processor.tokenizer.decode([tid]).strip() == ""):
                non_action += 1
    return {
        "total_tokens": total,
        "chunks_opened": chunks_opened,
        "chunks_closed": chunks_closed,
        "non_action_or_whitespace_tokens": non_action,
    }


def _build_prompt(processor, row):
    """Reconstruct the user turn (text + image) from a chunked parquet row.

    The user content in the source data is [{type:text}, {type:image}].
    We hand transformers' chat template a clean version that uses 'image' field on
    the image segment so processor.apply_chat_template can wire in the placeholder.
    """
    user_segments = row["conversations"][0]["content"]
    user_text = next(s["text"] for s in user_segments if s["type"] == "text")
    img = Image.open(io.BytesIO(row["image_bytes"][0])).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    return inputs, img, user_text


def _truth_summary(row):
    return [seg["text"] for seg in row["conversations"][1]["content"]]


def main():
    if not Path(MODEL_PATH).exists():
        print(f"checkpoint not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    if not Path(VALID_SHARD).exists():
        print(f"valid shard not found: {VALID_SHARD}", file=sys.stderr)
        return 1

    model, processor, schema, maps = _load()
    table = pq.read_table(VALID_SHARD)
    print(f"valid shard rows: {table.num_rows}")

    max_new_tokens = ACTION_CHUNK_LEN * PER_ACTION_TOKEN_BUDGET  # 64

    for i in range(NUM_SAMPLES):
        row = table.slice(i, 1).to_pylist()[0]
        _hr(f"sample {i}: id={row['id']}  label={row['label']}")
        print(f"truth (qwen2_vl-encoded):")
        for j, seg in enumerate(_truth_summary(row)):
            print(f"  chunk {j}: {seg[:120]}")

        inputs, img, user_text = _build_prompt(processor, row)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"prompt:  {user_text!r}  ({img.size[0]}x{img.size[1]} image)")
        print(f"input_ids: {inputs['input_ids'].shape}, generating up to {max_new_tokens} tokens...")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, prompt_len:].tolist()
        decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
        print(f"generated text:\n  {decoded}")
        stats = _classify_output(gen_ids, maps, processor)
        print(f"stats: {stats}")
        verdict = (
            "PARSEABLE" if stats["chunks_closed"] >= 1 and stats["non_action_or_whitespace_tokens"] == 0
            else "PARTIAL"  if stats["chunks_closed"] >= 1
            else "MALFORMED"
        )
        print(f"verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
