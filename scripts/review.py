"""Multi-model code/plan review via OpenRouter.

Sends the plan + (optionally) selected source files to multiple frontier models in
parallel and saves their critiques. The goal is to surface issues we may have missed
before committing to the next phase of implementation.

Usage:
    # Verify model slugs available on OpenRouter (defaults below may need updating).
    python scripts/review.py --list-models opus
    python scripts/review.py --list-models gpt-5

    # Review the plan only.
    python scripts/review.py --plan

    # Review the plan plus specific files, with a focused question.
    python scripts/review.py --plan \\
        --include tml/streaming/format.py \\
        --include tml/model/audio_lm.py \\
        --question "Phase 4 streaming format design — what's missing or wrong?"

    # Override default models.
    python scripts/review.py --plan --models anthropic/claude-opus-4.7,openai/gpt-5-pro

Outputs go to `runs/reviews/<timestamp>/` as one .md per model plus a `prompt.md`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

OPENROUTER_API = "https://openrouter.ai/api/v1"

# Best-guess defaults — verify with `--list-models` if you get a 404.
DEFAULT_MODELS = [
    "anthropic/claude-opus-4.7",
    "openai/gpt-5.5-pro",
    "google/gemini-3.1-pro-preview",
]

DEFAULT_SYSTEM = """You are a senior ML engineer reviewing an in-progress research project.
Your job is to find issues. Be specific and concrete:

- Identify correctness bugs, missing pieces, architectural problems, and risks.
- Cite specific file paths, line numbers, function names, or plan sections.
- Prioritize: lead with the most important issues; mention smaller nits last.
- If you have an opinion on a deferred design decision, state it with reasoning.
- Do not just praise. If the work looks fine, say so briefly and move to what's missing.
- Be terse. Lists and short paragraphs beat essays.

If you have no real concerns, say so explicitly rather than padding.""".strip()


# ---------- .env loader (no python-dotenv dependency) ----------

def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# ---------- OpenRouter client ----------

def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("error: OPENROUTER_API_KEY not set. Put it in .env or export it.", file=sys.stderr)
        sys.exit(2)
    return key


def list_models(filter_substr: str | None = None) -> list[dict]:
    r = requests.get(f"{OPENROUTER_API}/models", timeout=30,
                     headers={"Authorization": f"Bearer {_api_key()}"})
    r.raise_for_status()
    models = r.json().get("data", [])
    if filter_substr:
        f = filter_substr.lower()
        models = [m for m in models if f in m.get("id", "").lower()]
    return models


@dataclass
class ReviewResult:
    model: str
    elapsed_s: float
    ok: bool
    text: str
    error: str | None = None
    raw: dict | None = None


def call_model(model: str, prompt: str, system: str, timeout: int = 600) -> ReviewResult:
    """Call one OpenRouter chat-completions endpoint."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{OPENROUTER_API}/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tml-research",
                "X-Title": "TML-Interaction prototype review",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # Leave temperature unset — let each provider use its default for review tasks.
            },
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            return ReviewResult(model=model, elapsed_s=elapsed, ok=False, text="",
                                error=f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        return ReviewResult(model=model, elapsed_s=elapsed, ok=True, text=content, raw=data)
    except requests.RequestException as e:
        return ReviewResult(model=model, elapsed_s=time.time() - t0, ok=False, text="",
                            error=f"{type(e).__name__}: {e}")


# ---------- Prompt assembly ----------

def build_prompt(
    plan_text: str | None,
    files: dict[str, str],
    question: str | None,
) -> str:
    parts: list[str] = []
    parts.append("# Review request")
    parts.append("")
    if question:
        parts.append("## Specific question / focus")
        parts.append(question.strip())
        parts.append("")
    else:
        parts.append(
            "Please review the plan and any included code. Identify issues, missing "
            "pieces, risks, or things I should change before moving forward."
        )
        parts.append("")

    if plan_text:
        parts.append("## Plan document (PLAN.md)")
        parts.append("")
        parts.append("```markdown")
        parts.append(plan_text)
        parts.append("```")
        parts.append("")

    for path, body in files.items():
        parts.append(f"## File: `{path}`")
        parts.append("")
        ext = Path(path).suffix.lstrip(".") or "text"
        parts.append(f"```{ext}")
        parts.append(body)
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def read_files(paths: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            print(f"warning: {p} not found; skipping", file=sys.stderr)
            continue
        out[str(pp)] = pp.read_text()
    return out


# ---------- Per-result persistence ----------

def save_review(out_dir: Path, r: ReviewResult) -> Path:
    """Write a single review to disk. Called as each model completes."""
    safe = r.model.replace("/", "__")
    path = out_dir / f"{safe}.md"
    if r.ok:
        header = f"# Review by `{r.model}`\n\n_elapsed: {r.elapsed_s:.1f}s_\n\n---\n\n"
        path.write_text(header + r.text)
    else:
        path.write_text(f"# Review by `{r.model}` — FAILED\n\n```\n{r.error}\n```\n")
    return path


def write_index(out_dir: Path, stamp: str, results: list[ReviewResult],
                pending: list[str]) -> None:
    """Rewrite the index after every completion so it always reflects the
    current state. Lists pending models too, so you can tell what's still
    outstanding by reading the index alone."""
    lines = ["# Review session", "", f"_timestamp: {stamp}_", "", "## Completed", ""]
    for r in results:
        safe = r.model.replace("/", "__")
        status = "OK" if r.ok else "FAIL"
        lines.append(f"- [{r.model}](./{safe}.md) — {status}, {r.elapsed_s:.1f}s")
    if pending:
        lines.extend(["", "## Pending", ""])
        for m in pending:
            lines.append(f"- {m}")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", action="store_true", help="Include PLAN.md")
    p.add_argument("--plan_path", default="PLAN.md")
    p.add_argument("--include", action="append", default=[],
                   help="Path to a file to include (repeatable).")
    p.add_argument("--question", default=None, help="Specific question or focus area.")
    p.add_argument("--models", default=None,
                   help="Comma-separated model IDs; overrides defaults.")
    p.add_argument("--system", default=None,
                   help="Override the system prompt (default: critical-review prompt).")
    p.add_argument("--out_dir", default="runs/reviews",
                   help="Base directory for review outputs.")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--list-models", default=None, metavar="FILTER",
                   help="List available OpenRouter models matching FILTER and exit.")
    args = p.parse_args()

    load_dotenv()

    if args.list_models is not None:
        models = list_models(args.list_models or None)
        if not models:
            print(f"(no models match {args.list_models!r})")
            return
        for m in models:
            cl = m.get("context_length", "?")
            pricing = m.get("pricing", {})
            print(f"{m['id']:55s}  ctx={cl:>7}  prompt=${pricing.get('prompt','?')}/Mtok  completion=${pricing.get('completion','?')}/Mtok")
        return

    if not args.plan and not args.include and not args.question:
        p.error("nothing to review — pass --plan, --include FILE, or --question TEXT")

    plan_text = None
    if args.plan:
        pp = Path(args.plan_path)
        if not pp.exists():
            print(f"error: plan file {pp} not found", file=sys.stderr)
            sys.exit(2)
        plan_text = pp.read_text()

    files = read_files(args.include)
    prompt = build_prompt(plan_text, files, args.question)
    system = args.system or DEFAULT_SYSTEM
    models = [m.strip() for m in (args.models.split(",") if args.models else DEFAULT_MODELS) if m.strip()]

    # Output dir
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.md").write_text(prompt)
    (out_dir / "system.md").write_text(system)

    print(f"reviewing with {len(models)} models: {', '.join(models)}", flush=True)
    print(f"prompt: {len(prompt):,} chars  ({len(prompt)//4:,} ≈ tokens)", flush=True)
    print(f"output dir: {out_dir}\n", flush=True)

    # Initial index so the directory is browsable while reviews are in-flight.
    write_index(out_dir, stamp, [], list(models))

    # Parallel calls — each result is persisted (and previewed) the instant it
    # lands; index is rewritten each time so partial state is always on disk.
    results: list[ReviewResult] = []
    pending = list(models)
    with cf.ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        futures = {pool.submit(call_model, m, prompt, system, args.timeout): m for m in models}
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            if r.model in pending:
                pending.remove(r.model)
            path = save_review(out_dir, r)
            write_index(out_dir, stamp, results, pending)

            status = "OK" if r.ok else "FAIL"
            print(f"  [{status}] {r.model:45s}  {r.elapsed_s:6.1f}s  "
                  f"{(len(r.text) if r.ok else 0):>6} chars"
                  + (f"  ({r.error})" if r.error else "")
                  + f"  -> {path}", flush=True)

            # Inline preview the moment the file lands.
            if r.ok:
                lines = r.text.strip().split("\n")
                for line in lines[:15]:
                    print(f"     {line}", flush=True)
                if len(lines) > 15:
                    print(f"     ... ({len(lines) - 15} more lines — see {path})", flush=True)
            print(flush=True)

    print(f"saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
