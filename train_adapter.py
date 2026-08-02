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
import inspect
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--data", required=True, help="path to <subject>.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
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

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
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
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,           # keep off: preserves turn boundaries in chats
        report_to="none",
    )
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
    trainer.train(resume_from_checkpoint=bool(_ckpts))
    trainer.save_model(f"{args.out_dir}/{args.subject}")
    print(f"[✓] Saved LoRA adapter -> {args.out_dir}/{args.subject}")


if __name__ == "__main__":
    main()
