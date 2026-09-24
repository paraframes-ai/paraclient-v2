#!/usr/bin/env python3
"""Train one subject adapter: legacy Qwen QLoRA or Muse BF16 language-only LoRA."""
import argparse
import inspect
import json
import os
from pathlib import Path
import signal

from model_support import MODELS, model_id, render_dialogue


def latest_checkpoint(output_dir):
    """Ignore interrupted saves; our completion marker is written last."""
    candidates = []
    for path in Path(output_dir).glob('checkpoint-*'):
        suffix = path.name.removeprefix('checkpoint-')
        if suffix.isdigit() and (path / 'checkpoint_complete.json').exists():
            required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json', 'adapter_model.safetensors', 'rng_state.pth')
            if all((path / name).exists() for name in required):
                candidates.append((int(suffix), path))
    return str(max(candidates)[1]) if candidates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--base', default=MODELS['qwen'], help='qwen, muse, or a legacy Qwen HF ID/path')
    ap.add_argument('--revision')
    ap.add_argument('--out-dir', default='adapters')
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--max-seq-len', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=16)
    ap.add_argument('--no-flash-attn', action='store_true', help='Legacy flag; SDPA is now the portable default')
    ap.add_argument('--flash-attn', action='store_true')
    ap.add_argument('--quantization', choices=('auto', 'none', '4bit'), default='auto', help='auto = Qwen NF4, Muse BF16')
    ap.add_argument('--save-steps', type=int, default=50)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--resume', default='auto', help='auto, never, or an explicit complete checkpoint')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if min(args.rank, args.alpha, args.max_seq_len, args.batch_size, args.grad_accum, args.save_steps, args.save_total_limit) <= 0:
        ap.error('Training sizes, rank/alpha, and checkpoint limits must be positive')
    if args.flash_attn and args.no_flash_attn:
        ap.error('--flash-attn and --no-flash-attn are mutually exclusive')
    requested = {'signal': None}

    def handle_signal(signum, frame):
        requested['signal'] = signal.Signals(signum).name

    signal.signal(signal.SIGUSR1, handle_signal)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    from model_support import assert_language_only_adapter, language_lora_targets
    set_seed(args.seed)
    base = model_id(args.base)
    is_muse = base == MODELS['muse'] or args.base == 'muse'
    if is_muse and not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run Muse training through the GH200 Slurm scripts')
    revision = args.revision
    report_path = Path(__file__).resolve().parent / 'artifacts/muse_inspection.json'
    if is_muse and revision is None and report_path.exists():
        revision = json.loads(report_path.read_text())['revision']
    config = AutoConfig.from_pretrained(base, revision=revision, token=False)
    is_muse = config.model_type == 'muse_glimmer'
    if is_muse and args.quantization == '4bit':
        ap.error('The verified Muse path uses BF16; quantized Muse training is not validated')
    quantized = args.quantization == '4bit' or (args.quantization == 'auto' and not is_muse)
    family = 'muse' if is_muse else 'qwen'
    tok = AutoTokenizer.from_pretrained(base, revision=revision, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    output = Path(args.out_dir) / args.subject
    resume = latest_checkpoint(output) if args.resume == 'auto' else (None if args.resume == 'never' else args.resume)
    if resume and not Path(resume).is_dir():
        raise ValueError(f'Checkpoint not found: {resume}')
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'No complete checkpoint in nonempty {output}; use a new --out-dir')
    with open(args.data) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError('Training data is empty')
    texts = []
    for row in rows:
        if row.get('subject') != args.subject:
            raise ValueError('Data subject does not match --subject')
        if row.get('base_model') not in (None, base):
            raise ValueError('Data was formatted for a different base model')
        if revision and row.get('base_revision') not in (None, revision):
            raise ValueError('Data and training model revisions differ')
        # Render from messages again rather than trusting stale cached text.
        text = render_dialogue(tok, row, family)
        if row.get('text') is not None and row['text'] != text:
            raise ValueError('Stored text differs from the selected native chat template; rebuild data')
        texts.append({'text': text})
    ds = Dataset.from_list(texts)
    ds = ds.map(lambda batch: tok(batch['text'], truncation=True, max_length=args.max_seq_len, add_special_tokens=False), batched=True, remove_columns=['text'])
    model_kwargs = dict(revision=revision, token=False, torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='flash_attention_2' if args.flash_attn else 'sdpa')
    if quantized:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    if is_muse:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base, **model_kwargs)
        targets = language_lora_targets(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
        targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias='none', task_type='CAUSAL_LM', target_modules=targets))
    if is_muse:
        assert_language_only_adapter(model)
        # PEFT minimizes long target lists during injection. Preserve the verified
        # full paths in exported metadata for safe reloads and serving preflight.
        model.peft_config['default'].target_modules = set(targets)
        model.peft_config['default'].revision = revision or getattr(config, '_commit_hash', None)
    model.print_trainable_parameters()

    class SaveOnSignal(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if requested['signal']:
                print(f"Received {requested['signal']}; saving a resumable checkpoint at step {state.global_step}", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            (checkpoint / 'checkpoint_complete.json').write_text(json.dumps({'step': state.global_step, 'signal': requested['signal']}) + '\n')
            # Small progress marker lets the one smoke job exercise a real signal.
            (Path(args.output_dir) / 'progress.json').write_text(json.dumps({'step': state.global_step}) + '\n')
            return control

    class Collator:
        def __call__(self, features):
            batch = tok.pad(features, padding=True, return_tensors='pt')
            labels = batch['input_ids'].clone()
            labels[batch['attention_mask'] == 0] = -100
            batch['labels'] = labels
            return batch

    cfg_kwargs = dict(output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
                      per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
                      learning_rate=args.lr, lr_scheduler_type='cosine',
                      logging_steps=args.logging_steps, save_strategy='steps', save_steps=args.save_steps,
                      save_total_limit=args.save_total_limit, bf16=True,
                      gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                      optim='adamw_torch', report_to='none', seed=args.seed, data_seed=args.seed,
                      dataloader_num_workers=0, remove_unused_columns=False)
    # Transformers 5 folds warmup_ratio into fractional warmup_steps; keep 4.x.
    training_parameters = inspect.signature(TrainingArguments).parameters
    cfg_kwargs['warmup_ratio' if 'warmup_ratio' in training_parameters else 'warmup_steps'] = 0.03
    if 'save_safetensors' in training_parameters:
        cfg_kwargs['save_safetensors'] = True
    cfg = TrainingArguments(**cfg_kwargs)
    trainer_kwargs = dict(model=model, args=cfg, train_dataset=ds, data_collator=Collator(), callbacks=[SaveOnSignal()])
    trainer_kwargs['processing_class' if 'processing_class' in inspect.signature(Trainer).parameters else 'tokenizer'] = tok
    trainer = Trainer(**trainer_kwargs)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'training_pid').write_text(str(os.getpid()))
    manifest = dict(base_model=base, revision=revision or getattr(config, '_commit_hash', None), subject=args.subject,
                    family=family, rows=len(rows), rank=args.rank, alpha=args.alpha, dropout=args.dropout,
                    quantized=quantized, lora_targets=targets, resumed_from=resume,
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output))
    tok.save_pretrained(str(output))
    trainer.save_state()
    metrics = dict(result.metrics, global_step=trainer.state.global_step, stopped_by=requested['signal'], resumed_from=resume, cuda_peak_bytes=torch.cuda.max_memory_allocated())
    (output / 'train_results.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    if requested['signal']:
        raise SystemExit(75)  # Slurm wrapper requeues only after a complete save.


if __name__ == '__main__':
    main()
