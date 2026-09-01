#!/usr/bin/env bash
# Sweep the MLX perf levers on Apple Silicon and print a table directly
# comparable to the Linux CPU baseline in README.md.
#
# Uses mlx_lm.benchmark, whose real flags (mlx-lm 0.31.3) are:
#   --model --prompt-tokens --generation-tokens --batch-size --num-trials
#   --prefill-step-size --quantize-activations --pipeline --delay
#
# Dimensions match the CPU baseline exactly: 64-token prompt, 64-token
# generation, batch 1/2/4/8/16 -- so the numbers line up row for row.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: bench_mlx.sh <mlx-model-dir> [trials]}"
TRIALS="${2:-3}"
PP=64
TG=64

echo "model:  $MODEL"
echo "trials: $TRIALS   prompt_tokens=$PP   generation_tokens=$TG"
python3 -c "import platform;print('host:  ',platform.machine(),platform.processor() or '')" || true
echo

for B in 1 2 4 8 16; do
  echo "=== batch-size $B ==="
  python3 -m mlx_lm.benchmark \
    --model "$MODEL" \
    --prompt-tokens "$PP" \
    --generation-tokens "$TG" \
    --batch-size "$B" \
    --num-trials "$TRIALS"
  echo
done

cat <<'NOTE'
--- what to compare against ---
Linux CPU baseline, Xeon Platinum 8488C (Sapphire Rapids, 8 physical cores,
AMX + AVX512-VNNI), Qwen2.5-7B-Instruct Q4_K_M, llama.cpp c845263,
prompt 64 / generation 64:

  batch   prefill t/s   generation t/s   wall for all
      1         131          14.05          5.04 s
      2         141          17.93          8.04 s
      4         147          32.70          9.57 s
      8         149          56.84         12.45 s
     16         148          87.09         18.66 s

Single-stream ceiling was 14.12 t/s at 15 threads (8 threads gave 13.08, so
hyperthreads added ~8%).

Generation is memory-bandwidth bound: each token reads the whole quantized
weight set. Expected single-stream tokens/sec is roughly

    effective_bandwidth_GB_s / model_size_GB

A 4-bit 7B is about 4.0-4.4 GB, so divide the machine's achievable memory
bandwidth by ~4.2 to get a first-order estimate, then measure. Do NOT trust the
estimate over the measurement.
NOTE
