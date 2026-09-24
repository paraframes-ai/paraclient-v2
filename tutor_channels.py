"""Independent output boundary: only Muse to=user text may reach moderation/UI."""
import re
from collections.abc import Mapping


class OutputBoundaryError(ValueError):
    pass


_FRAME = re.compile(r'<\|start\|>assistant\s+to=([A-Za-z0-9_.-]+)<\|message\|>(.*?)(<\|eom\|>|<\|eot\|>)', re.DOTALL)
_FORBIDDEN = re.compile(r'<\||</?(?:think|thinking|analysis|reasoning|tool_output)\b|</?atem:|\bto\s*=\s*(?:self|[A-Za-z_][\w.]*\.[\w.]+)', re.IGNORECASE)


def _get(message, name, default=None):
    return message.get(name, default) if isinstance(message, Mapping) else getattr(message, name, default)


def student_content(message) -> str:
    """Accept parsed API content, or strictly parse complete native channel frames.

    Structured reasoning_content/reasoning and tool_calls are never forwarded.
    Plain content is the already-parsed user channel returned by an API parser
    (and preserves the Qwen path). Ambiguous/malformed native output fails closed.
    """
    if _get(message, 'recipient', 'user') not in (None, 'user'):
        raise OutputBoundaryError('Non-student recipient')
    content = _get(message, 'content')
    if not isinstance(content, str) or not content.strip():
        raise OutputBoundaryError('No student-facing content')
    text = content.strip()
    # apply_chat_template(add_generation_prompt=True) ends at "assistant".
    if re.match(r'^to=[A-Za-z0-9_.-]+<\|message\|>', text):
        text = '<|start|>assistant ' + text
    elif re.match(r'^assistant\s+to=', text):
        text = '<|start|>' + text
    if '<|start|>' in text:
        cursor = 0
        visible = []
        for match in _FRAME.finditer(text):
            if text[cursor:match.start()].strip():
                raise OutputBoundaryError('Unframed or malformed output')
            recipient, body, end = match.groups()
            if recipient == 'user':
                visible.append(body)
            cursor = match.end()
        if text[cursor:].strip() or not visible:
            raise OutputBoundaryError('Incomplete native output or missing user channel')
        text = '\n'.join(visible).strip()
    if not text or _FORBIDDEN.search(text):
        raise OutputBoundaryError('Private channel or protocol markers in student output')
    return text
