"""Async R2 (Cloudflare S3-compatible) upload of HF Trainer checkpoints.

Wired into train.py via R2UploadCallback.maybe_create(...). Enabled by setting
required env vars; absent → callback returns None and training proceeds unchanged.

Required env vars:
    R2_BUCKET, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY

Optional:
    R2_PREFIX           — remote key prefix (default: basename of output_dir)
    R2_MAX_WORKERS      — upload thread pool size (default: 4)

Behavior:
  - Rank-0 only under DDP; other ranks no-op.
  - on_save fires after each periodic checkpoint write → submits async upload.
  - upload_path() is the public entrypoint for the post-train.py final save_model().
  - wait_all() must be called before process exit to flush in-flight uploads.
  - Idempotent per-file: head_object skip-if-same-size means resumed runs don't
    re-upload, and a second invocation on the same dir is cheap.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

from transformers import TrainerCallback

_REQUIRED_ENV = ["R2_BUCKET", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]


class R2UploadCallback(TrainerCallback):
    """See module docstring."""

    @classmethod
    def maybe_create(cls, output_dir: str) -> "R2UploadCallback | None":
        """Construct only if env + boto3 are ready; otherwise print why and return None."""
        missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
        if missing:
            print(f"[r2-upload] disabled — missing env vars: {missing}", flush=True)
            return None
        try:
            import boto3  # noqa: F401
        except ImportError:
            print("[r2-upload] disabled — boto3 not installed (`pip install boto3`)", flush=True)
            return None
        return cls(output_dir)

    def __init__(self, output_dir: str):
        import boto3

        self.bucket = os.environ["R2_BUCKET"]
        self.prefix = os.environ.get("R2_PREFIX", Path(output_dir).name)
        max_workers = int(os.environ.get("R2_MAX_WORKERS", "4"))
        endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="r2-upload")
        self._inflight: list[Future] = []
        self._lock = threading.Lock()
        print(
            f"[r2-upload] enabled  bucket={self.bucket}  prefix={self.prefix}  workers={max_workers}",
            flush=True,
        )

    # ---------- public ----------

    def upload_path(self, local_dir, remote_subprefix: str | None = None) -> None:
        """Submit an async upload of `local_dir`. Skips any first-level subdir matching
        `checkpoint-*` since those are handled by on_save."""
        local_dir = Path(local_dir)
        remote_prefix = f"{self.prefix}/{remote_subprefix}" if remote_subprefix else self.prefix
        self._submit(local_dir, remote_prefix, skip_first_level_glob="checkpoint-*")

    def wait_all(self, timeout: float | None = None) -> None:
        """Block until all in-flight uploads finish. Call before process exit."""
        with self._lock:
            futs = list(self._inflight)
        if not futs:
            self._executor.shutdown(wait=True)
            return
        print(f"[r2-upload] waiting for {len(futs)} in-flight upload(s)...", flush=True)
        for f in futs:
            try:
                f.result(timeout=timeout)
            except Exception as e:
                print(f"[r2-upload] ERROR during wait_all: {e!r}", flush=True)
        self._executor.shutdown(wait=True)
        print("[r2-upload] all uploads complete", flush=True)

    # ---------- TrainerCallback hooks ----------

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            # HF Trainer also fires on_save for empty/best-model-only paths; just skip.
            return
        remote_prefix = f"{self.prefix}/checkpoint-{state.global_step}"
        self._submit(ckpt_dir, remote_prefix)

    # ---------- internals ----------

    def _submit(self, local_dir: Path, remote_prefix: str, skip_first_level_glob: str | None = None):
        with self._lock:
            future = self._executor.submit(
                self._upload_impl, local_dir, remote_prefix, skip_first_level_glob
            )
            self._inflight.append(future)
            self._reap_done_locked()

    def _reap_done_locked(self):
        """Surface errors from completed uploads without blocking. Caller holds the lock."""
        done = [f for f in self._inflight if f.done()]
        for f in done:
            try:
                f.result()
            except Exception as e:
                print(f"[r2-upload] ERROR in completed upload: {e!r}", flush=True)
            self._inflight.remove(f)

    def _upload_impl(self, local_dir: Path, remote_prefix: str, skip_first_level_glob: str | None):
        files = []
        for f in sorted(local_dir.rglob("*")):
            if not f.is_file():
                continue
            if skip_first_level_glob is not None:
                top = f.relative_to(local_dir).parts[0]
                if fnmatch(top, skip_first_level_glob):
                    continue
            files.append(f)
        if not files:
            return
        total = sum(f.stat().st_size for f in files)
        print(
            f"[r2-upload] start  {local_dir.name} → s3://{self.bucket}/{remote_prefix}  "
            f"({len(files)} files, {total/2**30:.2f} GiB)",
            flush=True,
        )
        t0 = time.time()
        uploaded_bytes = 0
        skipped = 0
        for f in files:
            key = f"{remote_prefix}/{f.relative_to(local_dir).as_posix()}"
            size = f.stat().st_size
            if self._already_uploaded(key, size):
                skipped += 1
                continue
            self._s3.upload_file(str(f), self.bucket, key)
            uploaded_bytes += size
        dt = time.time() - t0
        rate = uploaded_bytes / dt / 2**30 if dt > 0 else 0
        print(
            f"[r2-upload] done   {local_dir.name}  in {dt:.1f}s  "
            f"({uploaded_bytes/2**30:.2f} GiB uploaded, {skipped} files skipped, {rate:.2f} GiB/s)",
            flush=True,
        )

    def _already_uploaded(self, key: str, size: int) -> bool:
        try:
            head = self._s3.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return False
        return head.get("ContentLength") == size
