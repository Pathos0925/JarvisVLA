"""Re-render the JarvisVLA SFT dataset with N-action assistant turns.

The original CraftJarvis/minecraft-vla-sft dataset has one (observation, action) pair
per example and is shuffled across trajectories. Inference with action_chunk_len > 1
requires the model to have seen `<|act_end|><|act_start|>` boundaries mid-generation
during training — otherwise the chunked-inference benefit is lost.

This script:
  1. Downloads the dataset (load_dataset without streaming).
  2. Adds trajectory_key + frame_offset columns parsed from the id.
  3. Sorts by (trajectory_key, frame_offset) so frames from the same trajectory are adjacent.
  4. Walks the sorted dataset and emits non-overlapping windows of N consecutive frames
     as chunked examples. Each chunked example has the keyframe's image + a concatenated
     assistant turn containing N action segments.
  5. Writes output as parquet shards under --output-dir.

Disk requirement: hundreds of GB (full dataset download). Run on a machine with the
/ephemeral mount, not a small laptop.

Usage:
    python scripts/preprocess_chunked_actions.py --chunk-len 4 \\
        --output-dir /ephemeral/datasets/jarvisvla-chunk4

    # Smoke run on first 5k examples per split (skip the full sort):
    python scripts/preprocess_chunked_actions.py --chunk-len 4 \\
        --output-dir /ephemeral/datasets/jarvisvla-chunk4-smoke \\
        --max-examples 5000 --splits train
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

# Parses an id like "jumpy-denim-lion-5304af046010-20220105-224029-1603-1615_2" into
# (trajectory_key, frame_offset) = ("jumpy-denim-lion-5304af046010-20220105-224029-1603-1615", 2).
_ID_RE = re.compile(r"^(.*?)_(\d+)$")


def _parse_id(example_id: str) -> tuple[str, int]:
    m = _ID_RE.match(example_id)
    if not m:
        return ("", -1)
    return m.group(1), int(m.group(2))


def _trajectory_in_holdout(traj_key: str, fraction: float) -> bool:
    """Stable trajectory-level holdout. fraction=0.01 → ~1% of trajectories go to valid.

    Uses md5 so the split is deterministic across runs / machines. Hashing the trajectory
    key (not the per-frame id) ensures all frames from a trajectory stay in the same split
    — critical for chunked data, otherwise chunks could straddle the train/valid boundary.
    """
    if fraction <= 0:
        return False
    h = int(hashlib.md5(traj_key.encode("utf-8")).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < fraction


def _add_sort_keys(ex: dict) -> dict:
    traj_key, frame_offset = _parse_id(ex["id"])
    ex["_traj_key"] = traj_key
    ex["_frame_offset"] = frame_offset
    return ex


def _chunk_grouped_examples(examples_iter, chunk_len: int):
    """Walk a sorted iterable of examples and yield chunked examples.

    Examples are assumed sorted by (_traj_key, _frame_offset). Within each trajectory,
    emits non-overlapping windows of chunk_len consecutive frames. Trailing partial
    windows are dropped.
    """
    current_traj = None
    buffer: list[dict] = []

    def emit_buffer():
        for start in range(0, len(buffer) - chunk_len + 1, chunk_len):
            window = buffer[start : start + chunk_len]
            yield _build_chunk(window, chunk_len)

    for ex in examples_iter:
        traj = ex.get("_traj_key", "")
        if not traj:
            continue
        if traj != current_traj:
            yield from emit_buffer()
            buffer = []
            current_traj = traj
        buffer.append(ex)

    yield from emit_buffer()


def _build_chunk(window: list[dict], chunk_len: int) -> dict:
    """Combine N consecutive single-action examples into one N-action example."""
    keyframe = window[0]
    user_turn = next(t for t in keyframe["conversations"] if t.get("role") == "user")
    combined_assistant_content: list[dict] = []
    for ex in window:
        for t in ex["conversations"]:
            if t.get("role") == "assistant":
                combined_assistant_content.extend(t.get("content", []))
                break
    return {
        "id": f"{keyframe['id']}__chunk{chunk_len}",
        "label": keyframe.get("label", []),
        "image": keyframe.get("image", []),
        "image_bytes": keyframe.get("image_bytes", []),
        "conversations": [
            user_turn,
            {"role": "assistant", "content": combined_assistant_content},
        ],
    }


def _process_split(
    dataset_name: str, split: str, chunk_len: int, output_dir: Path,
    shard_size: int, max_examples: int, holdout_fraction: float = 0.0,
    holdout_split_name: str = "valid",
) -> None:
    """Process one source split into chunked parquet shards.

    If holdout_fraction > 0, the chunked output is split into two destinations by
    trajectory hash: chunks from trajectories in the holdout go to {holdout_split_name}-*,
    everything else to {split}-*.
    """
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"\n=== split: {split} ===", flush=True)
    if holdout_fraction > 0:
        print(f"  trajectory-level holdout: {holdout_fraction:.1%} → {holdout_split_name}-*", flush=True)
    print(f"  load_dataset({dataset_name!r}, split={split!r}) — downloads to local cache", flush=True)
    t0 = time.time()
    ds = load_dataset(dataset_name, split=split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    print(f"  loaded {len(ds):,} rows in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    print("  adding _traj_key + _frame_offset columns...", flush=True)
    ds = ds.map(_add_sort_keys, desc=f"key {split}")
    print(f"  done in {time.time()-t1:.0f}s", flush=True)

    t2 = time.time()
    print("  sorting by (_traj_key, _frame_offset)...", flush=True)
    ds = ds.sort(["_traj_key", "_frame_offset"])
    print(f"  done in {time.time()-t2:.0f}s", flush=True)

    t3 = time.time()
    print(f"  chunking with chunk_len={chunk_len}...", flush=True)

    # Two output buckets: primary (split name) and optional holdout.
    buffers: dict[str, list[dict]] = {split: [], holdout_split_name: []}
    shard_idx: dict[str, int] = {split: 0, holdout_split_name: 0}
    counts: dict[str, int] = {split: 0, holdout_split_name: 0}

    def flush(bucket: str) -> None:
        if not buffers[bucket]:
            return
        shard_path = output_dir / f"{bucket}-{shard_idx[bucket]:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buffers[bucket]), shard_path)
        print(f"    wrote {shard_path.name}  rows={len(buffers[bucket])}  total_chunks[{bucket}]={counts[bucket]}",
              flush=True)
        buffers[bucket].clear()
        shard_idx[bucket] += 1

    for chunk in _chunk_grouped_examples(iter(ds), chunk_len):
        # Look up the trajectory_key from the chunk id: format is "<traj_key>_<keyframe_offset>__chunk<N>".
        chunk_id = chunk["id"]
        # Strip the __chunkN suffix, then parse off the keyframe offset.
        keyframe_id = chunk_id.rsplit("__chunk", 1)[0]
        traj_key, _ = _parse_id(keyframe_id)
        bucket = holdout_split_name if (holdout_fraction > 0 and _trajectory_in_holdout(traj_key, holdout_fraction)) else split
        buffers[bucket].append(chunk)
        counts[bucket] += 1
        if len(buffers[bucket]) >= shard_size:
            flush(bucket)

    flush(split)
    if holdout_fraction > 0:
        flush(holdout_split_name)

    total_chunks = counts[split] + counts[holdout_split_name]
    print(f"  chunked in {time.time()-t3:.0f}s; {total_chunks:,} chunks "
          f"({counts[split]:,} {split} + {counts[holdout_split_name]:,} {holdout_split_name}) "
          f"from {len(ds):,} input rows "
          f"(yield {100*total_chunks*chunk_len/max(len(ds),1):.1f}% of input frames consumed)",
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="CraftJarvis/minecraft-vla-sft")
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    p.add_argument("--chunk-len", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=50_000)
    p.add_argument("--max-examples", type=int, default=0,
                   help="Cap input examples per split (0 = unlimited). Smoke testing only.")
    p.add_argument("--holdout-fraction", type=float, default=0.0,
                   help="If > 0, hash trajectory_key and route this fraction of trajectories "
                        "from --splits[0] into a separate {--holdout-split-name}-* output. "
                        "Used to carve a chunked valid set from train (the source dataset's "
                        "own valid split is too sparse to chunk).")
    p.add_argument("--holdout-split-name", type=str, default="valid",
                   help="Output bucket name for the holdout. Default 'valid'.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        # Holdout only applies to the first (assumed = train) split. The script's other
        # splits (e.g. the source valid) are processed unchanged.
        holdout = args.holdout_fraction if split == args.splits[0] else 0.0
        _process_split(args.dataset, split, args.chunk_len, output_dir,
                       args.shard_size, args.max_examples,
                       holdout_fraction=holdout, holdout_split_name=args.holdout_split_name)

    print(f"\nall done. output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
