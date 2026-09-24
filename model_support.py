"""Shared model IDs, native chat rendering, and language-only LoRA selection."""
from __future__ import annotations

MODELS = {'qwen': 'Qwen/Qwen2.5-7B-Instruct', 'muse': 'meta-models/Muse-Glimmer-30B'}
MUSE_LANGUAGE_PREFIX = 'model.language_model.layers.'
MUSE_VISION_PREFIXES = ('model.vision_tower', 'model.vision_adapter', 'model.vision_projection')


def model_id(base: str) -> str:
    return MODELS.get(base, base)


def render_dialogue(tokenizer, row: dict, base: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    if row.get('tools'):
        if base != 'muse':
            raise ValueError('Native Muse tool examples require --base muse')
        kwargs['tools'] = row['tools']
    if base == 'muse':
        kwargs['reasoning_strength'] = row.get('reasoning_strength', 'low')
    return tokenizer.apply_chat_template(row['messages'], **kwargs)


def language_lora_targets(model) -> list[str]:
    """Use full nn.Linear paths: vision uses several of the same suffixes."""
    import torch
    allowed = {
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'self_attn.gate_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    }
    targets = [name for name, module in model.named_modules()
               if name.startswith(MUSE_LANGUAGE_PREFIX)
               and isinstance(module, torch.nn.Linear)
               and '.'.join(name.split('.')[-2:]) in allowed]
    layers = model.config.text_config.num_hidden_layers
    if len(targets) != layers * len(allowed):
        raise ValueError(f'Unexpected Muse linear layout: {len(targets)} targets for {layers} layers')
    if not any(name.startswith('model.vision_tower.') for name, _ in model.named_modules()):
        raise ValueError('Expected Muse vision tower is absent; inspect the architecture again')
    return sorted(targets)


def assert_language_only_adapter(model):
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('model.language_model.layers.' not in name or '.lora_' not in name
                            for name in trainable):
        raise ValueError(f'Unexpected trainable parameters outside language LoRA: {trainable[:10]}')
    for name, p in model.named_parameters():
        if any(prefix in name for prefix in MUSE_VISION_PREFIXES) and p.requires_grad:
            raise ValueError(f'Vision parameter was unfrozen: {name}')
