"""Generate predictions on diverse valid-set samples and render a self-contained HTML.

Loads a JarvisVLA checkpoint, picks N samples with distinct prompts from valid-00000,
runs greedy generation, decodes predicted + ground-truth action chunks via the
Qwen2-VL action tokenizer, and writes an HTML page (with embedded base64 images)
comparing them side by side.

Usage:
    PYTHONPATH=. MODEL_PATH=/path/to/checkpoint python scripts/inspect_html.py
    # writes /tmp/checkpoint_review.html
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import io
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from jarvisvla.inference import action_mapping, action_tokens

GROUP_NAMES = [
    "hotbar", "fore_back", "left_right", "sprint_sneak", "use",
    "drop", "attack", "jump", "camera_flag", "inventory", "cam_pitch", "cam_yaw",
]
# camera bins are 0..20 with 10 = center = no-op
CAMERA_CENTER = 10
# Friendly value names where useful; anything else shown as the raw integer.
VALUE_NAMES = {
    "fore_back": {0: "·", 1: "fwd", 2: "back"},
    "left_right": {0: "·", 1: "left", 2: "right"},
    "sprint_sneak": {0: "·", 1: "sprint", 2: "sneak"},
    "use": {0: "·", 1: "USE"},
    "drop": {0: "·", 1: "DROP"},
    "attack": {0: "·", 1: "ATTACK"},
    "jump": {0: "·", 1: "JUMP"},
    "camera_flag": {0: "·", 1: "cam"},
    "hotbar": {0: "·"},  # 1-9 shown as int; 0 means no change
}


def humanize(group_action: list[int]) -> dict:
    """Turn a 12-int group action into {name: value} where value is either an int or a friendly name."""
    out = {}
    for i, name in enumerate(GROUP_NAMES):
        v = group_action[i]
        if name in VALUE_NAMES and v in VALUE_NAMES[name]:
            out[name] = VALUE_NAMES[name][v]
        else:
            out[name] = str(v)
    return out


def render_action_table(actions: list[list[int]]) -> str:
    """Render a list of 12-int group actions as a small HTML table (one row per action chunk)."""
    if not actions:
        return "<i>(no actions parsed)</i>"
    rows = []
    header = "".join(f"<th>{n}</th>" for n in GROUP_NAMES)
    rows.append(f"<tr>{header}</tr>")
    for action in actions:
        h = humanize(action)
        cells = []
        for name in GROUP_NAMES:
            v = h[name]
            raw = action[GROUP_NAMES.index(name)]
            # Camera-bin midpoint counts as null (no-op); same for explicit 0
            is_camera_bin = name in ("cam_pitch", "cam_yaw")
            null = (v in {"·", "0"}) or (is_camera_bin and raw == CAMERA_CENTER)
            cls = "null" if null else "active"
            cells.append(f'<td class="{cls}">{html_mod.escape(str(v))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="action-table">{"".join(rows)}</table>'


def encode_image_data_uri(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_PATH",
        "/workspace/checkpoints/mc-vla-qwen3-5-9b-h200-full-e1-e1-b4-a1-p1.0/checkpoint-30000",
    ))
    p.add_argument("--valid-shard", default="/workspace/datasets/jarvisvla-chunk4/valid-00000.parquet")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--output", default="/tmp/checkpoint_review.html")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    print(f"loading {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model.eval()

    # Build a qwen2_vl action tokenizer (that's what the model emits — see README
    # Debugging history). Fall back to qwen3_5 only for the verdict marker.
    qwen2_vl_tok = action_mapping.OneActionTokenizer.from_tokenizer(
        backbone="qwen2_vl", tokenizer=processor.tokenizer,
    )
    print(f"  vocab={len(processor.tokenizer)}  act_beg(qwen2_vl)={qwen2_vl_tok.act_beg_id}", flush=True)

    print(f"picking {args.num_samples} samples with distinct prompts from {args.valid_shard}", flush=True)
    table = pq.read_table(args.valid_shard)
    seen = {}
    samples = []
    random.seed(args.seed)
    indices = random.sample(range(table.num_rows), min(500, table.num_rows))
    for i in indices:
        row = table.slice(i, 1).to_pylist()[0]
        user_text = next(s["text"] for s in row["conversations"][0]["content"] if s["type"] == "text")
        key = user_text[:50]
        if key in seen:
            continue
        seen[key] = True
        samples.append(row)
        if len(samples) >= args.num_samples:
            break

    cards = []
    for i, row in enumerate(samples):
        user_text = next(s["text"] for s in row["conversations"][0]["content"] if s["type"] == "text")
        img = Image.open(io.BytesIO(row["image_bytes"][0])).convert("RGB")
        # Ground truth — tokenize assistant content as text and feed to the action tokenizer.
        truth_text_parts = []
        for seg in row["conversations"][1]["content"]:
            if seg.get("type") == "text":
                truth_text_parts.append(seg["text"])
        truth_text = "".join(truth_text_parts)
        truth_ids = processor.tokenizer.encode(truth_text, add_special_tokens=False)
        truth_actions = qwen2_vl_tok.token_2_group_action(truth_ids)

        # Predicted — model.generate with greedy decoding.
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
        pred_actions = qwen2_vl_tok.token_2_group_action(gen_ids)

        # Compare: did predicted chunks match truth on a per-position basis?
        match_count = sum(
            1 for j in range(min(len(pred_actions), len(truth_actions)))
            if pred_actions[j] == truth_actions[j]
        )
        match_str = f"{match_count}/{len(truth_actions)} chunks identical to ground truth"

        img_uri = encode_image_data_uri(img)
        label = ", ".join(row.get("label", []))
        truth_html = render_action_table(truth_actions)
        pred_html = render_action_table(pred_actions)

        cards.append(f"""
<div class="card">
  <h2>sample {i}</h2>
  <div class="meta">
    <div><b>id:</b> {html_mod.escape(row['id'])}</div>
    <div><b>label:</b> {html_mod.escape(label)}</div>
    <div><b>prompt:</b> <code>{html_mod.escape(user_text)}</code></div>
    <div><b>image size:</b> {img.size[0]}×{img.size[1]}</div>
    <div class="match"><b>match:</b> {match_str}</div>
  </div>
  <div class="cols">
    <div class="col image">
      <img src="{img_uri}" alt="sample {i}" />
    </div>
    <div class="col actions">
      <h3>predicted (greedy)</h3>
      {pred_html}
      <h3>ground truth</h3>
      {truth_html}
    </div>
  </div>
</div>
""")
        print(f"  sample {i}: {user_text[:60]!r} → {match_str}", flush=True)

    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 2em; background: #1a1a1a; color: #e0e0e0; }
h1 { border-bottom: 2px solid #555; padding-bottom: 0.3em; }
.card { border: 1px solid #444; border-radius: 8px; padding: 1em; margin: 1.5em 0;
        background: #222; }
.card h2 { margin-top: 0; color: #88c0d0; }
.card h3 { color: #a3be8c; margin: 0.8em 0 0.3em; font-size: 0.95em; }
.meta { font-size: 0.9em; line-height: 1.6; margin-bottom: 1em; color: #b0b0b0; }
.meta code { background: #2e3440; padding: 0.1em 0.4em; border-radius: 3px; }
.match { color: #ebcb8b; font-weight: bold; }
.cols { display: flex; gap: 1em; }
.col.image { flex: 0 0 auto; max-width: 50%; }
.col.image img { max-width: 100%; border: 1px solid #555; border-radius: 4px; }
.col.actions { flex: 1; min-width: 0; }
.action-table { border-collapse: collapse; font-size: 0.78em;
                font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.action-table th, .action-table td {
    border: 1px solid #444; padding: 0.2em 0.5em; text-align: center; }
.action-table th { background: #2e3440; font-weight: 600; color: #88c0d0; }
.action-table td.active { background: #3b4252; color: #ebcb8b; font-weight: 600; }
.action-table td.null { color: #555; }
.intro { color: #b0b0b0; font-size: 0.95em; line-height: 1.6; }
"""
    body = "\n".join(cards)
    html_out = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>JarvisVLA checkpoint review</title>
<style>{css}</style></head><body>
<h1>JarvisVLA checkpoint review</h1>
<div class="intro">
Model: <code>{html_mod.escape(args.model_path)}</code><br>
Samples drawn at random from <code>{html_mod.escape(args.valid_shard)}</code> (seed={args.seed}).
Action tokens decoded via Qwen2-VL schema (what this model emits).
The model emits 4 action chunks per response; each row in a table is one chunk's 12 group values.
Active (non-null) values are highlighted; "·" is the no-op.
</div>
{body}
</body></html>
"""
    Path(args.output).write_text(html_out)
    print(f"wrote {args.output}  ({len(html_out)/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
