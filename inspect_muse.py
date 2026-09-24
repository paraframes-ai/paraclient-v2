#!/usr/bin/env python3
"""Run only in a GH200 allocation; load BF16 weights and record runtime facts."""
import json
import os
from pathlib import Path
import platform
import sys


def main():
    if not os.environ.get('SLURM_JOB_ID') or platform.machine() != 'aarch64':
        raise SystemExit('Run inspection through Slurm on GH200')
    import torch
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from model_support import MODELS, language_lora_targets
    base = MODELS['muse']
    try:
        revision = HfApi().model_info(base, token=False).sha
        config = AutoConfig.from_pretrained(base, revision=revision, token=False)
        tok = AutoTokenizer.from_pretrained(base, revision=revision, token=False)
        model = AutoModelForImageTextToText.from_pretrained(
            base, revision=revision, token=False, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa')
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403):
            raise SystemExit('HF_AUTH_REQUIRED: stop and ask the user; do not write a token') from exc
        raise
    tools = [{'type': 'function', 'function': {'name': 'tutor.hint', 'description': 'Return a guiding hint, never a final answer.', 'parameters': {'type': 'object', 'properties': {'topic': {'type': 'string'}}, 'required': ['topic'], 'additionalProperties': False}}}]
    dialogue = [
        {'role': 'system', 'content': 'You are a Socratic math tutor. Never give the final answer.'},
        {'role': 'user', 'content': 'I am stuck on multiplication. Can you find a hint?'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'PRIVATE_INSPECTION_CANARY', 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'tutor.hint', 'arguments': {'topic': 'multiplication'}}}]},
        {'role': 'tool', 'name': 'tutor.hint', 'tool_call_id': 'call_1', 'content': '{"hint":"Think about equal groups."}'},
        {'role': 'assistant', 'content': 'What does one group contain?'},
    ]
    rendered = tok.apply_chat_template(dialogue, tools=tools, reasoning_strength='low', tokenize=False, add_generation_prompt=False)
    encoded = tok.apply_chat_template(dialogue, tools=tools, reasoning_strength='low', tokenize=True, add_generation_prompt=False)
    prompt = tok.apply_chat_template(dialogue[:2], tools=tools, reasoning_strength='low', tokenize=False, add_generation_prompt=True)
    assert 'assistant to=self<|message|>PRIVATE_INSPECTION_CANARY<|eom|>' in rendered
    assert 'assistant to=tutor.hint<|message|><atem:function_calls>' in rendered
    assert 'assistant to=user<|message|>What does one group contain?<|eot|>' in rendered
    assert '<tool_output name="tutor.hint">' in rendered
    assert prompt.endswith('<|start|>assistant')
    rejected_json_string = False
    dialogue[2]['tool_calls'][0]['function']['arguments'] = '{"topic":"multiplication"}'
    try:
        tok.apply_chat_template(dialogue, tools=tools, tokenize=False)
    except Exception as exc:
        if 'dict' not in str(exc) and 'mapping' not in str(exc):
            raise
        rejected_json_string = True
    assert rejected_json_string
    targets = language_lora_targets(model)
    inputs = tok('A student asks for a hint.', return_tensors='pt').to('cuda')
    with torch.inference_mode():
        result = model(**inputs, use_cache=False)
    assert torch.isfinite(result.logits).all().item()
    report = dict(
        job_id=os.environ['SLURM_JOB_ID'], host=platform.node(), machine=platform.machine(),
        model_id=base, revision=revision, architectures=config.architectures,
        auto_class='AutoModelForImageTextToText', runtime_class=type(model).__name__,
        model_type=config.model_type, parameter_count=sum(p.numel() for p in model.parameters()),
        vision_parameter_count=sum(p.numel() for name, p in model.named_parameters() if name.startswith('model.vision_tower.')),
        vision_module_paths=['model.vision_tower', 'model.vision_adapter', 'model.vision_projection'],
        language_linear_targets=targets, target_count=len(targets),
        rendered_tool_dialogue=rendered, generation_prompt_suffix=prompt[-100:],
        token_count=len(encoded['input_ids']) if isinstance(encoded, dict) or hasattr(encoded, 'keys') else len(encoded), string_tool_arguments_rejected=rejected_json_string,
        forward_logits_shape=list(result.logits.shape), cuda_peak_bytes=torch.cuda.max_memory_allocated(),
    )
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/muse_inspection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('language_linear_targets', 'rendered_tool_dialogue')}, indent=2), flush=True)
    print('NATIVE_CHAT_TEMPLATE_EXAMPLE\n' + rendered, flush=True)


def adapter_preflight():
    """Inspect PEFT/Trainer integration on a meta model; never perform training."""
    if not os.environ.get('SLURM_JOB_ID') or platform.machine() != 'aarch64':
        raise SystemExit('Run inspection through Slurm on GH200')
    import inspect
    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, TrainingArguments
    from model_support import MODELS, language_lora_targets, assert_language_only_adapter, render_dialogue
    report_path = Path('artifacts/muse_inspection.json')
    report = json.loads(report_path.read_text())
    base, revision = report['model_id'], report['revision']
    tok = AutoTokenizer.from_pretrained(base, revision=revision, token=False)
    config = AutoConfig.from_pretrained(base, revision=revision, token=False)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16, attn_implementation='sdpa')
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM', target_modules=language_lora_targets(model)))
    assert_language_only_adapter(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = inspect.signature(TrainingArguments).parameters
    warmup = {'warmup_ratio' if 'warmup_ratio' in parameters else 'warmup_steps': 0.03}
    TrainingArguments(output_dir=str(Path(os.environ['TMPDIR']) / 'inspection-only'), bf16=True,
                      gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                      save_strategy='steps', save_steps=1, save_total_limit=2, report_to='none', optim='adamw_torch', **warmup)
    qwen = AutoTokenizer.from_pretrained(MODELS['qwen'], token=False)
    row = {'messages': [{'role': 'system', 'content': 'Socratic tutor.'}, {'role': 'user', 'content': 'Help?'}, {'role': 'assistant', 'content': 'What have you tried?'}]}
    assert render_dialogue(qwen, row, 'qwen') == qwen.apply_chat_template(row['messages'], tokenize=False)
    report['token_count'] = len(tok(report['rendered_tool_dialogue'], add_special_tokens=False)['input_ids'])
    report['peft_runtime_class'] = type(model).__name__
    report['lora_trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['adapter_preflight_job_id'] = os.environ['SLURM_JOB_ID']
    report['qwen_template_regression_passed'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['token_count', 'peft_runtime_class', 'lora_trainable_parameter_count', 'adapter_preflight_job_id', 'qwen_template_regression_passed']}, indent=2))


if __name__ == '__main__':
    if '--adapter-only' not in sys.argv:
        main()
    adapter_preflight()
