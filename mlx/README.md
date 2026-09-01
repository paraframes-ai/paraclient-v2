# MLX serving path (Apple Silicon)

## Status: UNVALIDATED. No number in this directory has been measured.

Written and API-checked against **mlx-lm 0.31.3**, but never executed. There is
no Apple hardware in the environment this was authored in — x86_64 Linux, where
the `mlx` backend cannot be installed at all (`mlx-lm` resolves, `import
mlx.core` fails). So every function name, flag, and signature here was read out
of the installed mlx-lm source rather than guessed — but nothing was run.

**Do not treat this as optimized.** It is a correct-by-inspection starting point
plus a harness for doing the optimization on real hardware. Run
`mlx/bench_mlx.sh` first; choose settings from what it reports.

## Why `serve_mlx.py` exists instead of just `mlx_lm.server`

`auth_gateway.py` routes by model **name**: `model_for(mode, subject)` returns
`ParaFrames/ParaClient-math-v2.2` and similar. Under vLLM those are
`--served-model-name` aliases. `mlx_lm.server` has no equivalent flag.

Reading mlx-lm 0.31.3, `ModelProvider` populates only one entry:

```python
self._model_map["default_model"]       = cli_args.model
self._adapter_map["default_model"]     = cli_args.adapter_path
self._draft_model_map["default_model"] = cli_args.draft_model
```

and resolves a request with `self._model_map.get(model_path, model_path)`. An
unknown name therefore **falls through and is treated as a model path**, so a
request for `ParaFrames/ParaClient-math-v2.2` tries to load that string as a
model and fails.

**Stock `mlx_lm.server` cannot serve this gateway.** `serve_mlx.py` populates
those maps from `aliases.json` — the upstream comment marks them as the intended
extension point (*"could be extended"*) — which avoids forking mlx-lm and also
gets per-subject LoRA adapters natively: one base resident, adapter chosen by the
requested name. That matches the ParaFrames architecture exactly, and is better
than vLLM's `--max-loras 3` ceiling.

**The gateway needs no code change** — point it at this server:

```bash
python auth_gateway.py --tutor-url http://127.0.0.1:8080/v1 --host 127.0.0.1 --port 8081
```

## Usage

```bash
pip install mlx-lm                      # Apple Silicon only

# 1. convert + quantize
python mlx/convert_mlx.py --hf-path Qwen/Qwen2.5-7B-Instruct \
    --mlx-path models/mlx/qwen2.5-7b-instruct-4bit --q-bits 4 --q-group-size 64

# 2. measure BEFORE choosing settings
./mlx/bench_mlx.sh models/mlx/qwen2.5-7b-instruct-4bit

# 3. serve (edit mlx/aliases.json to point at your converted dirs)
python mlx/serve_mlx.py --port 8080 --prompt-cache-size 16
```

## The levers worth sweeping, and why

| Lever | Flag | Why it matters here |
|---|---|---|
| Quantization bits | `--q-bits 4/6/8` | Generation is bandwidth-bound, so bits ≈ directly proportional to tokens/sec. Biggest single lever. |
| Group size | `--q-group-size 32/64/128` | Smaller groups cost size but recover accuracy. |
| Quantizer | `mlx_lm.quant.{awq,dwq,gptq}` | Stronger than plain `affine`; usually recovers 4-bit accuracy. Try before accepting a quality loss. |
| Batched decode | `--decode-concurrency` | On the Linux CPU baseline, batching gave **6.2×** aggregate generation throughput (14 → 87 t/s at batch 16). Expect a similar shape. |
| Prompt cache | `--prompt-cache-size` | **Product-specific win.** The gateway prepends one of only **12** distinct `(mode, subject)` system prompts. A cache of ≥12 should make system-prompt prefill essentially free. |
| Speculative decoding | `--draft-model`, `--num-draft-tokens` | Helps single-stream latency. **Mutually exclusive with batching** — mlx-lm sets `is_batchable = (draft_model is None)`. Pick per deployment: spec decode for few users, batching for many. |
| Prefill step size | `--prefill-step-size` | Prefill was already fast on CPU (~148 t/s, flat across batch); likely minor. |

## Linux CPU baseline to beat

Measured, Xeon Platinum 8488C (Sapphire Rapids, 8 physical cores, AMX +
AVX512-VNNI/BF16), Qwen2.5-7B-Instruct **Q4_K_M**, llama.cpp `c845263`,
prompt 64 / generation 64 tokens:

| batch | prefill t/s | generation t/s | wall for all |
|---|---|---|---|
| 1 | 131 | 14.05 | 5.04 s |
| 2 | 141 | 17.93 | 8.04 s |
| 4 | 147 | 32.70 | 9.57 s |
| 8 | 149 | 56.84 | 12.45 s |
| 16 | 148 | 87.09 | 18.66 s |

Single-stream thread scaling: 3.72 t/s (2 threads) → 7.08 (4) → 13.08 (8) →
14.12 (15). Near-linear to the physical core count, then +8% from hyperthreads.

**Read for the product:** at 16 concurrent students each waits ~19 s for a
64-token reply, and real tutoring turns of 150–300 tokens would be 3–5× that.
CPU-only is a dev-scale story, which is the conclusion that motivated this
directory.

## Expected mechanism, not a predicted number

Generation reads the entire quantized weight set per token, so single-stream
throughput is bounded by memory bandwidth:

```
tokens/sec  ≈  effective_memory_bandwidth_GB_s / model_size_GB
```

A 4-bit 7B is ~4.0–4.4 GB. Divide the machine's achievable bandwidth by ~4.2 for
a first-order estimate. Apple Silicon Ultra parts have substantially higher
unified-memory bandwidth than a server CPU's DRAM path, which is the structural
reason to expect a large win — but **no specific figure is asserted here**, because
the authoring environment had no way to measure one and quoting a spec sheet is
not a measurement.

## What still needs doing

1. Run `bench_mlx.sh`; pick quantization and concurrency from the results.
2. Verify output quality after quantization. There is **no** tutoring-quality
   metric in this repo — `eval_adapter.py` reports no score and always exits 0 —
   so a quality gate has to be built before quantization settings can be chosen
   responsibly. For the circuit model, `circuit_schema.erc()` gives an objective
   signal and is the better starting point.
3. Adapters: none exist yet (`adapters/` is gitignored and empty). Until they do,
   every alias in `aliases.json` should point at the base model with
   `"adapter": null`. `mlx_lm.fuse` can bake an adapter in if you prefer a single
   merged artifact over runtime adapter selection.
4. Wire `circuit_schema.grammar()` — it returns a complete GBNF grammar and has
   **zero callers** anywhere in the repo. Constrained decoding removes
   malformed-JSON retries, which matters more when tokens are expensive.
