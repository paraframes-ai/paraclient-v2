#!/usr/bin/env python3
"""
push_adapters_hf.py — back the trained LoRA adapters up to a PRIVATE HF repo.

Only the base models (Gemma-4 12B, Qwen 2.5) live on the Hub already; these
subject adapters are the irreplaceable, hand-trained part, and they currently
exist on exactly one box. This mirrors the FINAL adapters (not the mid-training
checkpoints) into one private repo, organised by generation:

    v4/<subject>/   from adaptersG4/    (Gemma-4 12B  — live)
    v3/<subject>/   from adapters14b/   (Qwen2.5-14B)
    v2/<subject>/   from adapters/      (Qwen2.5-7B)

WHAT IS EXCLUDED, AND WHY
    Everything under a `checkpoint-*/` directory. Those hold optimizer.pt (~308
    MB each) and other resumable-training state — reproducible by re-running the
    trainer, and not needed to load or serve the adapter. Excluding them takes
    the payload from ~7 GB to ~2.5 GB.

AUTH
    Write token from HF_TOKEN, or --token-file (default ~/.hf_token). The token
    is never printed. Namespace is derived from whoami() unless --namespace is
    given.

Usage:
    python push_adapters_hf.py [--namespace NS] [--repo NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# generation -> (local dir, human note for the README)
GENERATIONS = {
    "v4": ("adaptersG4", "Gemma-4 12B (live GPU tier)"),
    "v3": ("adapters14b", "Qwen2.5-14B (AWQ)"),
    "v2": ("adapters", "Qwen2.5-7B"),
}

# Mid-training state — reproducible, and the bulk of the bytes. Never uploaded.
IGNORE = ["**/checkpoint-*/**", "**/checkpoint-*",
          "**/optimizer.pt", "**/scheduler.pt", "**/rng_state*.pth",
          "**/trainer_state.json", "**/*.tmp"]


def _read_token(token_file: str) -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    p = Path(token_file).expanduser()
    if p.exists():
        tok = p.read_text().strip()
    if not tok:
        sys.exit(f"no token: set HF_TOKEN or write one to {token_file} "
                 "(a WRITE-scoped Hugging Face token)")
    return tok


def _subjects(local: Path) -> list[str]:
    return sorted(d.name for d in local.iterdir()
                  if d.is_dir() and not d.name.startswith("checkpoint-"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default=None, help="HF user or org (default: whoami)")
    ap.add_argument("--repo", default="paraclient-adapters")
    ap.add_argument("--token-file", default="~/.hf_token")
    ap.add_argument("--dry-run", action="store_true",
                    help="show exactly what would be uploaded, touch nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = _read_token(args.token_file)
    api = HfApi(token=token)
    who = api.whoami()
    ns = args.namespace or who.get("name")
    repo_id = f"{ns}/{args.repo}"
    print(f"[*] authed as {who.get('name')} | target private repo: {repo_id}")

    # Plan first, so a dry-run is a real preview and a typo is caught before any
    # network write.
    plan = []
    for gen, (dirname, note) in GENERATIONS.items():
        local = HERE / dirname
        if not local.is_dir():
            print(f"[!] skip {gen}: {local} not found")
            continue
        subs = _subjects(local)
        if subs:
            plan.append((gen, local, subs, note))
            print(f"    {gen}/  <- {dirname}  ({len(subs)} adapters: {', '.join(subs)})")
    if not plan:
        sys.exit("nothing to upload")

    if args.dry_run:
        print("[dry-run] no repo created, nothing uploaded.")
        return

    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"[*] private repo ready: https://huggingface.co/{repo_id}")

    for gen, local, subs, note in plan:
        print(f"[*] uploading {gen}/ ({note}) ...")
        api.upload_folder(
            repo_id=repo_id, folder_path=str(local), path_in_repo=gen,
            ignore_patterns=IGNORE,
            commit_message=f"Add {gen} adapters ({note}) — final weights only",
        )
        print(f"    done: {gen}/ ({len(subs)} adapters)")

    readme = _readme(repo_id, plan)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo_id, commit_message="Add repo README")
    print(f"[✓] complete: https://huggingface.co/{repo_id}  (PRIVATE)")


def _readme(repo_id: str, plan) -> str:
    lines = [
        "---", "library_name: peft", "license: other", "---",
        f"# ParaClient / Kalvi — subject LoRA adapters (PRIVATE)", "",
        "Final trained LoRA adapters for the ParaFrames tutor, one per subject, "
        "organised by model generation. Training checkpoints are intentionally "
        "excluded — these are the served weights only.", "",
        "| Folder | Base model | Adapters |", "|---|---|---|",
    ]
    for gen, _local, subs, note in plan:
        lines.append(f"| `{gen}/` | {note} | {', '.join(subs)} |")
    lines += [
        "", "## Serving", "",
        "Each `<gen>/<subject>/` is a self-contained PEFT adapter "
        "(`adapter_config.json` + `adapter_model.safetensors`). vLLM can load one "
        "with `--lora-modules <name>=<path-or-repo-revision>`.", "",
        "Served-model naming: `v4` → Kalvi/ParaClient 4, `v3` → 3, `v2` → 2.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
