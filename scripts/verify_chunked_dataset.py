"""Quick verifier for the chunked-action preprocessor output.

Reads parquet shards from --dir, reports shard sizes, schema, and one example
deeply (id + conversation roles + action-segment count) to confirm the chunks
have the expected structure.

Usage:
    python scripts/verify_chunked_dataset.py --dir /ephemeral/datasets/jarvisvla-chunk4-smoke
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq


_ACT_SEG = re.compile(r"<\|act_start\|.*?<\|act_end\|>")
_RESERVED_178 = re.compile(r"<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>")


def _count_action_segments(text: str) -> int:
    """Count segments bracketed by <|act_start|>..<|act_end|> OR Qwen2-VL's reserved-178/179."""
    n = len(_ACT_SEG.findall(text)) + len(_RESERVED_178.findall(text))
    return n


def _content_to_str(content) -> str:
    """Flatten a conversation-content list (list of {type, text} dicts) to plain text."""
    if isinstance(content, str):
        return content
    out = []
    for c in content:
        if isinstance(c, dict):
            if c.get("type") == "text":
                out.append(c.get("text", ""))
            else:
                out.append(f"<{c.get('type', 'unknown')}>")
        else:
            out.append(str(c))
    return "".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", required=True)
    p.add_argument("--expected-chunk-len", type=int, default=4)
    args = p.parse_args()

    root = Path(args.dir)
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        print(f"no parquet shards in {root}")
        return 1

    print(f"found {len(shards)} shard(s) in {root}")
    total_rows = 0
    seg_counts: dict[int, int] = {}
    sample_done = False
    for shard in shards:
        table = pq.read_table(shard)
        rows = table.num_rows
        total_rows += rows
        print(f"  {shard.name}  rows={rows}  cols={table.column_names}")
        # Sample first row of first shard for deep inspection.
        if not sample_done and rows > 0:
            first = table.slice(0, 1).to_pylist()[0]
            print("\nfirst chunk:")
            print(f"  id:    {first['id']}")
            print(f"  label: {first.get('label')}")
            print(f"  image: {first.get('image')}")
            print(f"  image_bytes len: {[len(b) if isinstance(b, (bytes, bytearray)) else None for b in first.get('image_bytes', [])]}")
            print(f"  conversations turns: {len(first['conversations'])}")
            for t in first["conversations"]:
                role = t.get("role")
                text = _content_to_str(t.get("content", []))
                n_seg = _count_action_segments(text) if role == "assistant" else 0
                preview = text[:160] + ("..." if len(text) > 160 else "")
                print(f"    {role:9s}  segments={n_seg}  preview={preview!r}")
            sample_done = True
        # Count action segments per row to confirm chunk_len.
        for row in table.to_pylist():
            assistant = next((t for t in row["conversations"] if t.get("role") == "assistant"), None)
            if assistant is None:
                continue
            n = _count_action_segments(_content_to_str(assistant.get("content", [])))
            seg_counts[n] = seg_counts.get(n, 0) + 1

    print(f"\ntotal chunks: {total_rows}")
    print(f"action-segments-per-chunk distribution: {dict(sorted(seg_counts.items()))}")
    expected = args.expected_chunk_len
    on_target = seg_counts.get(expected, 0)
    pct = 100 * on_target / max(total_rows, 1)
    print(f"chunks with exactly {expected} segments: {on_target}/{total_rows} ({pct:.1f}%)")
    return 0 if pct >= 99 else 2


if __name__ == "__main__":
    raise SystemExit(main())
