"""Translate a chunked dataset from one action-token schema to another.

The original chunked dataset (`preprocess_chunked_actions.py` output) uses the
Qwen2-VL schema (`<|reserved_special_token_178|>`...`<|reserved_special_token_179|>`)
because that's how the source CraftJarvis/minecraft-vla-sft data was rendered.
A model trained with --backbone=qwen3_5 then learns those tokens instead of the
canonical `<|act_start|>`...`<|act_end|>` Qwen3.5 tokens — see README "Debugging
history".

This script does a token-level string replacement on each parquet shard so the
same dataset reads as if it were rendered with the qwen3_5 schema. Only the
assistant content strings change; user content, images, and ids are untouched.

Usage:
    python scripts/translate_action_schema.py \\
        --src /workspace/datasets/jarvisvla-chunk4 \\
        --dst /workspace/datasets/jarvisvla-chunk4-q35 \\
        --from-schema qwen2_vl \\
        --to-schema qwen3_5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Lazy-imported only if PYTHONPATH points at the repo, since the script is meant
# to be run standalone too.
def _load_schemas(from_name: str, to_name: str):
    from jarvisvla.inference.action_tokens import get_schema  # type: ignore
    return get_schema(from_name), get_schema(to_name)


def build_replacement_map(from_schema, to_schema) -> dict[str, str]:
    """Map every token string in `from_schema` to its positional equivalent in `to_schema`."""
    if from_schema.bases != to_schema.bases:
        raise ValueError(
            f"schemas have different bases (from={from_schema.bases}, to={to_schema.bases})"
        )
    if len(from_schema.group_tokens) != len(to_schema.group_tokens):
        raise ValueError("schemas have different group counts")
    out: dict[str, str] = {
        from_schema.act_start: to_schema.act_start,
        from_schema.act_end: to_schema.act_end,
    }
    for g, (from_group, to_group) in enumerate(zip(from_schema.group_tokens, to_schema.group_tokens)):
        if len(from_group) != len(to_group):
            raise ValueError(f"group {g}: sizes differ ({len(from_group)} vs {len(to_group)})")
        for from_tok, to_tok in zip(from_group, to_group):
            if from_tok in out and out[from_tok] != to_tok:
                # Collision means the same source string maps to two destination strings — bug.
                raise ValueError(f"replacement collision for {from_tok!r}")
            out[from_tok] = to_tok
    return out


def translate_text(text: str, replacement: dict[str, str]) -> str:
    """Replace each source token string with its destination. Order-independent
    because all tokens look like <|...|> and don't overlap."""
    # The naive str.replace loop is fine here: there are ~330 distinct tokens, each
    # called at most a few times per assistant turn. We're disk-bound, not CPU-bound.
    for src, dst in replacement.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def translate_conversations(conversations, replacement) -> list:
    """conversations is [{role, content: [{type, text} | {type:'image', ...}]}, ...].
    Only translates 'text' fields of any role's content (we want to be conservative;
    in practice only assistant content has the action tokens)."""
    new_convs = []
    for turn in conversations:
        new_content = []
        for seg in turn.get("content", []):
            if seg.get("type") == "text" and isinstance(seg.get("text"), str):
                new_content.append({**seg, "text": translate_text(seg["text"], replacement)})
            else:
                new_content.append(seg)
        new_convs.append({**turn, "content": new_content})
    return new_convs


def translate_shard(src_path: Path, dst_path: Path, replacement: dict[str, str]) -> tuple[int, int]:
    """Translate one parquet shard. Returns (row_count, byte_count_in)."""
    table = pq.read_table(src_path)
    n = table.num_rows

    convs = table.column("conversations").to_pylist()
    new_convs = [translate_conversations(c, replacement) for c in convs]
    new_table = table.set_column(
        table.column_names.index("conversations"),
        "conversations",
        pa.array(new_convs),
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return n, src_path.stat().st_size


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True, type=Path, help="source dataset dir (parquet shards)")
    p.add_argument("--dst", required=True, type=Path, help="destination dataset dir")
    p.add_argument("--from-schema", default="qwen2_vl", help="source schema name")
    p.add_argument("--to-schema", default="qwen3_5", help="destination schema name")
    p.add_argument("--shard-glob", default="*.parquet", help="pattern for source shards")
    args = p.parse_args()

    from_schema, to_schema = _load_schemas(args.from_schema, args.to_schema)
    replacement = build_replacement_map(from_schema, to_schema)
    print(f"translating {args.src} → {args.dst}", flush=True)
    print(f"  schema: {args.from_schema} → {args.to_schema}", flush=True)
    print(f"  replacements: {len(replacement)} tokens", flush=True)

    src_shards = sorted(args.src.glob(args.shard_glob))
    if not src_shards:
        print(f"no shards matching {args.shard_glob} in {args.src}", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_bytes = 0
    t0 = time.time()
    for src in src_shards:
        dst = args.dst / src.name
        rows, bytes_ = translate_shard(src, dst, replacement)
        total_rows += rows
        total_bytes += bytes_
        print(f"  {src.name}  rows={rows}  ({bytes_/2**30:.2f} GiB → {dst.stat().st_size/2**30:.2f} GiB)", flush=True)

    dt = time.time() - t0
    print(
        f"done. {len(src_shards)} shards, {total_rows:,} rows, "
        f"{total_bytes/2**30:.1f} GiB in {dt:.1f}s ({total_bytes/dt/2**30:.2f} GiB/s)",
        flush=True,
    )

    # Quick verification: read the first row of the first shard back and confirm
    # the action token strings are in the target schema.
    sample = pq.read_table(args.dst / src_shards[0].name).slice(0, 1).to_pylist()[0]
    sample_text = ""
    for turn in sample["conversations"]:
        if turn.get("role") == "assistant":
            for seg in turn["content"]:
                if seg.get("type") == "text":
                    sample_text += seg["text"]
    if to_schema.act_start in sample_text and from_schema.act_start not in sample_text:
        print(f"verified: assistant text contains {to_schema.act_start!r}, no {from_schema.act_start!r}", flush=True)
    else:
        print(f"WARNING: post-translation sample doesn't look right", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
