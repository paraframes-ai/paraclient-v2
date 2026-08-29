#!/usr/bin/env python3
"""
train_adapter.py — QLoRA fine-tune one subject adapter on Qwen2.5-7B-Instruct.

Tuned to fit a SINGLE NVIDIA L4 (24 GB). One adapter per subject; same command
for every subject, just change --subject / --data.

    python scripts/train_adapter.py --subject math --data data/math.jsonl

Expectations on an L4:
- 4-bit base (QLoRA) + gradient checkpointing + capped seq len keeps you in 24GB.
- This is SLOW (real hours per epoch on a few-thousand-example set). That is
  expected on an L4 — it fits, it isn't fast. Adapters trained here load
  identically to ones trained on an A100, so you can move training later.

Produces a LoRA adapter (~tens of MB) at adapters/<subject>/ — NOT a merged
full model. vLLM serves it with --enable-lora.
"""
import argparse
import glob
import hashlib
import inspect
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _loss_curve(log_history: list[dict]) -> str:
    points = [(float(row["step"]), float(row["loss"])) for row in log_history
              if "step" in row and "loss" in row]
    width, height, margin = 800, 420, 55
    if not points:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}"><text x="{margin}" y="{margin}" '
                'font-family="sans-serif">No logged loss points</text></svg>')
    x_max = max(x for x, _ in points) or 1.0
    y_min = min(y for _, y in points)
    y_max = max(y for _, y in points)
    y_span = y_max - y_min or 1.0
    coords = " ".join(
        f"{margin + x / x_max * (width - 2 * margin):.1f},"
        f"{height - margin - (y - y_min) / y_span * (height - 2 * margin):.1f}"
        for x, y in points)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444"/>
<polyline fill="none" stroke="#1677ff" stroke-width="3" points="{coords}"/>
<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif">Training step</text>
<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif">Loss</text>
<text x="{margin}" y="30" font-family="sans-serif">Training loss</text>
</svg>'''


def _write_training_artifacts(out: Path, args, trainer, metrics: dict) -> None:
    data_path = Path(args.data)
    manifest = {
        "subject": args.subject,
        "base_model": args.base,
        "requested_revision": args.revision,
        "resolved_revision": getattr(trainer.model.config, "_commit_hash", None),
        "dataset": str(data_path),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "max_sequence_length": args.max_seq_len,
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "lora_dropout": 0.05,
        },
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "trl": _package_version("trl"),
            "peft": _package_version("peft"),
            "datasets": _package_version("datasets"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    (out / "loss_curve.svg").write_text(_loss_curve(trainer.state.log_history))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--data", required=True, help="path to <subject>.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out-dir", default="adapters")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048,
                    help="lower to 1024 if you hit OOM on the L4")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-device; keep at 1 on L4, scale with grad-accum")
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="effective batch = batch-size * grad-accum")
    ap.add_argument("--no-flash-attn", action="store_true",
                    help="set if flash-attn 2 isn't installed")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="if >0, checkpoint every N steps instead of every epoch "
                         "(use for long jobs on a preemptable partition)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="if >0, stop after N steps (quick pipeline validation)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # ---- 4-bit quantization (QLoRA) — the thing that makes 7B fit on L4 ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Our JSONL rows carry subject/mode alongside `messages`. Newer TRL applies
    # the chat template to `messages` automatically; the pinned trl 0.11.x does
    # not, so render each conversation to text explicitly via a formatting_func
    # (batched form: returns one string per example).
    ds = load_dataset("json", data_files=args.data, split="train")

    def formatting_func(ex):
        # trl 0.11.x calls this batched: ex["messages"] is a list of conversations,
        # so return one rendered string per conversation. Newer trl calls it
        # per-example: ex["messages"] is a single conversation (list of role/content
        # dicts), so return one string. Detect by whether the first element is a list.
        msgs = ex["messages"]
        if msgs and isinstance(msgs[0], list):
            return [tok.apply_chat_template(m, tokenize=False) for m in msgs]
        return tok.apply_chat_template(msgs, tokenize=False)

    cfg_kwargs = dict(
        output_dir=f"{args.out_dir}/{args.subject}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "epoch",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,           # keep off: preserves turn boundaries in chats
        report_to="none",
    )
    if args.save_steps:
        cfg_kwargs["save_steps"] = args.save_steps
    if args.max_steps:
        cfg_kwargs["max_steps"] = args.max_steps
    # trl 0.11.x names this max_seq_length; newer trl renamed it to max_length.
    _sft_params = inspect.signature(SFTConfig.__init__).parameters
    cfg_kwargs["max_length" if "max_length" in _sft_params else "max_seq_length"] = \
        args.max_seq_len
    cfg = SFTConfig(**cfg_kwargs)

    # trl 0.11.x takes tokenizer=; newer trl renamed it to processing_class=.
    _trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    _tok_kw = "processing_class" if "processing_class" in _trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        formatting_func=formatting_func,
        **{_tok_kw: tok},
    )

    print(f"[*] Training {args.subject} adapter on {len(ds)} examples "
          f"(effective batch {args.batch_size * args.grad_accum})...")
    # Resume from the latest checkpoint if one exists (so a preempted/requeued
    # job on the preemptable partition continues instead of restarting at step 0).
    _ckpts = glob.glob(f"{args.out_dir}/{args.subject}/checkpoint-*")
    result = trainer.train(resume_from_checkpoint=bool(_ckpts))
    out = Path(args.out_dir) / args.subject
    trainer.save_model(out)
    _write_training_artifacts(out, args, trainer, result.metrics)
    print(f"[✓] Saved LoRA adapter -> {out}")


if __name__ == "__main__":
    main()
