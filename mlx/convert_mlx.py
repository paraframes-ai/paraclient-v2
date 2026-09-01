#!/usr/bin/env python3
"""
Convert + quantize a HF model to MLX format.

Thin wrapper over mlx_lm.convert, whose signature was read from mlx-lm 0.31.3:

    convert(hf_path, mlx_path="mlx_model", quantize=False, q_group_size=None,
            q_bits=None, q_mode="affine", dtype=None, upload_repo=None,
            revision=None, dequantize=False, quant_predicate=None,
            trust_remote_code=False)

q_mode choices in that version: affine | mxfp4 | nvfp4 | mxfp8.

The quantization settings are the main quality/speed lever and MUST be chosen by
measuring on the target hardware -- see bench_mlx.sh. Defaults here are a
starting point, not a recommendation.

mlx-lm also ships stronger quantizers under mlx_lm.quant (awq, dwq, gptq,
dynamic_quant). Those typically recover accuracy at 4 bits versus plain affine
and are worth trying before accepting a quality drop.

NOT VALIDATED ON HARDWARE -- see mlx/README.md.
"""
import argparse, sys

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mlx-path", required=True, help="e.g. models/mlx/qwen2.5-7b-instruct-4bit")
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--q-bits", type=int, default=4, help="4 | 6 | 8")
    p.add_argument("--q-group-size", type=int, default=64, help="32 | 64 | 128")
    p.add_argument("--q-mode", default="affine",
                   choices=["affine", "mxfp4", "nvfp4", "mxfp8"])
    p.add_argument("--dtype", default=None, help="e.g. bfloat16 (unquantized only)")
    a = p.parse_args()

    try:
        from mlx_lm.convert import convert
    except ImportError as e:
        print(f"FATAL: mlx-lm unavailable ({e}). Apple Silicon only.", file=sys.stderr)
        return 1

    convert(hf_path=a.hf_path, mlx_path=a.mlx_path,
            quantize=not a.no_quantize,
            q_bits=None if a.no_quantize else a.q_bits,
            q_group_size=None if a.no_quantize else a.q_group_size,
            q_mode=a.q_mode, dtype=a.dtype)
    print(f"[mlx] wrote {a.mlx_path}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
