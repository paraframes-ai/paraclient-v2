#!/usr/bin/env python3
"""
Gateway-compatible MLX server for Apple Silicon.

WHY THIS EXISTS RATHER THAN JUST `mlx_lm.server`
------------------------------------------------
auth_gateway.py routes by model NAME: model_for(mode, subject) returns
"ParaFrames/ParaClient-math-v2.2" and friends. Under vLLM those names are
--served-model-name aliases. mlx_lm.server has NO equivalent flag.

Verified against mlx-lm 0.31.3 source: ModelProvider builds

    self._model_map["default_model"]        = cli_args.model
    self._adapter_map["default_model"]      = cli_args.adapter_path
    self._draft_model_map["default_model"]  = cli_args.draft_model

and resolves a request with

    model_path   = self._model_map.get(model_path, model_path)
    adapter_path = self._adapter_map.get(model_path, adapter_path)

So an unknown name falls through and is treated as a filesystem/HF path. A
request for "ParaFrames/ParaClient-math-v2.2" would therefore try to LOAD that
string as a model and fail -- the gateway cannot talk to a stock mlx_lm.server.

The maps are the documented extension point (the upstream comment reads "map
'default_model' to the provided model by cli argument but could" ...), so this
script populates them instead of forking mlx-lm. That also buys per-subject LoRA
adapters natively: one base in memory, adapter selected by the requested name,
which is exactly the ParaFrames architecture.

NOT VALIDATED ON HARDWARE. Written and API-checked against mlx-lm 0.31.3 on
x86_64 Linux, where the `mlx` backend cannot be installed, so nothing here has
been executed. Run mlx/bench_mlx.sh on the target Mac before trusting any of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ALIASES = "mlx/aliases.json"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aliases", default=DEFAULT_ALIASES,
                   help="JSON mapping gateway model names -> {model, adapter, draft}")
    p.add_argument("--host", default="127.0.0.1",
                   help="keep localhost-only; the gateway is the public surface")
    p.add_argument("--port", type=int, default=8080)
    # Perf knobs worth sweeping on the target hardware (see bench_mlx.sh).
    p.add_argument("--decode-concurrency", type=int, default=None,
                   help="batched decode width. Mutually exclusive with speculative "
                        "decoding: mlx-lm sets is_batchable = (draft_model is None)")
    p.add_argument("--prompt-concurrency", type=int, default=None)
    p.add_argument("--prefill-step-size", type=int, default=None)
    p.add_argument("--prompt-cache-size", type=int, default=None,
                   help="number of cached prompt prefixes. The gateway prepends one "
                        "of only 12 distinct (mode, subject) system prompts, so a "
                        "cache of >=12 should make system-prompt prefill ~free")
    p.add_argument("--prompt-cache-bytes", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    alias_path = Path(args.aliases)
    if not alias_path.exists():
        print(f"FATAL: alias map not found: {alias_path}", file=sys.stderr)
        return 1
    aliases = json.loads(alias_path.read_text())
    if "default" not in aliases:
        print("FATAL: alias map needs a 'default' entry (used when a request "
              "names an unknown model)", file=sys.stderr)
        return 1

    try:
        from mlx_lm.server import ModelProvider, main as mlx_main  # noqa: F401
    except ImportError as e:
        print(f"FATAL: mlx-lm unavailable ({e}). This runs on Apple Silicon only.",
              file=sys.stderr)
        return 1

    # Translate our alias map into the CLI shape mlx_lm.server expects, then
    # inject the remaining names into the provider maps.
    default = aliases["default"]
    argv = ["mlx_lm.server",
            "--model", default["model"],
            "--host", args.host,
            "--port", str(args.port),
            "--log-level", args.log_level]
    if default.get("adapter"):
        argv += ["--adapter-path", default["adapter"]]
    if default.get("draft"):
        argv += ["--draft-model", default["draft"]]
    for flag, val in (("--decode-concurrency", args.decode_concurrency),
                      ("--prompt-concurrency", args.prompt_concurrency),
                      ("--prefill-step-size", args.prefill_step_size),
                      ("--prompt-cache-size", args.prompt_cache_size),
                      ("--prompt-cache-bytes", args.prompt_cache_bytes)):
        if val is not None:
            argv += [flag, str(val)]

    original_init = ModelProvider.__init__

    def patched_init(self, cli_args, *a, **kw):
        original_init(self, cli_args, *a, **kw)
        for name, spec in aliases.items():
            if name == "default":
                continue
            self._model_map[name] = spec["model"]
            self._adapter_map[spec["model"]] = spec.get("adapter")
            if spec.get("draft"):
                self._draft_model_map[name] = spec["draft"]
        print(f"[mlx] registered {len(aliases) - 1} gateway model aliases: "
              f"{sorted(k for k in aliases if k != 'default')}", file=sys.stderr)

    ModelProvider.__init__ = patched_init

    sys.argv = argv
    print(f"[mlx] launching: {' '.join(argv)}", file=sys.stderr)
    if default.get("draft"):
        print("[mlx] NOTE: a draft model is set, so mlx-lm will disable batching "
              "(is_batchable = draft_model is None). Speculative decoding and "
              "batched decode cannot both be active.", file=sys.stderr)
    return mlx_main() or 0


if __name__ == "__main__":
    sys.exit(main())
