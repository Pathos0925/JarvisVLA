"""Download a FULL resumable checkpoint dir from R2 (incl optimizer/scheduler/rng).

Unlike scripts/inference/r2_fetch_and_serve.sh (which skips resume-only files for
inference), this pulls everything so HF Trainer can auto-resume. Idempotent: skips
files already present at the right size.
"""
import os, sys, boto3
from pathlib import Path
from botocore.config import Config

PREFIX = "mc-vla-qwen3-5-9b-h200-full-e1-e1-b4-a1-p1.0"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoint-35000"
DEST = Path(sys.argv[2] if len(sys.argv) > 2
            else f"/workspace/checkpoints/{PREFIX}/{CKPT}")

ep = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
ak = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")
s3 = boto3.client("s3", endpoint_url=ep, aws_access_key_id=ak,
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  config=Config(signature_version="s3v4",
                                max_pool_connections=16,
                                retries={"max_attempts": 5, "mode": "standard"}))
b = os.environ["R2_BUCKET"]
remote_prefix = f"{PREFIX}/{CKPT}/"
DEST.mkdir(parents=True, exist_ok=True)

paginator = s3.get_paginator("list_objects_v2")
objs = []
for page in paginator.paginate(Bucket=b, Prefix=remote_prefix):
    objs.extend(page.get("Contents", []))
total = sum(o["Size"] for o in objs)
print(f"[fetch] {len(objs)} files, {total/1e9:.2f} GB -> {DEST}", flush=True)

for o in objs:
    name = o["Key"][len(remote_prefix):]
    if not name:
        continue
    out = DEST / name
    if out.exists() and out.stat().st_size == o["Size"]:
        print(f"[skip] {name} ({o['Size']/1e9:.3f} GB, already complete)", flush=True)
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {name} ({o['Size']/1e9:.3f} GB) ...", flush=True)
    s3.download_file(b, o["Key"], str(out))
print("[fetch] DONE", flush=True)
