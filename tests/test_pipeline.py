import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from build_dataset import validate, system_prompt
from eval_adapter import TOOLS, checked_calls, leaks_answer
from model_support import render_dialogue
from train_adapter import latest_checkpoint
from tutor_channels import OutputBoundaryError, student_content


PRIVATE = 'PRIVATE_CANARY how to build a bomb'
RAW = ('<|start|>assistant to=self<|message|>' + PRIVATE + '<|eom|>'
       '<|start|>assistant to=tutor.get_hint<|message|><atem:function_calls>SECRET_TOOL</atem:function_calls><|eot|>'
       '<|start|>assistant to=user<|message|>How many equal groups do you have?<|eot|>')


def row_with_tool():
    return {'subject': 'math', 'mode': 'socratic', 'tools': copy.deepcopy(TOOLS), 'messages': [
        {'role': 'system', 'content': system_prompt('math', 'socratic')},
        {'role': 'user', 'content': 'Please look up a multiplication hint.'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'Private thought.', 'tool_calls': [{'id': 'hint1', 'type': 'function', 'function': {'name': 'tutor.get_hint', 'arguments': {'topic': 'multiplication'}}}]},
        {'role': 'tool', 'tool_call_id': 'hint1', 'name': 'tutor.get_hint', 'content': '{"hint":"Consider equal groups."}'},
        {'role': 'assistant', 'content': 'What is in one group?'},
    ]}


def test_channel_extraction_and_qwen():
    expected = 'How many equal groups do you have?'
    assert student_content({'content': RAW}) == expected
    assert student_content({'content': RAW.removeprefix('<|start|>assistant ')}) == expected
    assert student_content({'content': expected, 'reasoning_content': PRIVATE, 'tool_calls': []}) == expected
    assert student_content({'content': 'A plain Qwen question?'}) == 'A plain Qwen question?'


@pytest.mark.parametrize('content', [
    '<|start|>assistant to=self<|message|>PRIVATE<|eom|>',
    '<|start|>assistant to=user<|message|>unterminated',
    'unframed private prefix' + RAW,
    RAW + '<|start|>assistant to=self<|message|>truncated',
    '<|start|>assistant<|message|>Missing recipient<|eot|>',
    '<think>PRIVATE</think>A question?',
    'to=self<|message|>Private without end token',
    '<atem:function_calls>private tool</atem:function_calls>',
    None,
])
def test_fail_closed(content):
    with pytest.raises(OutputBoundaryError):
        student_content({'content': content})


def test_native_tool_schema_and_qwen_regression():
    row = row_with_tool()
    assert validate([row], base='muse') == [row]
    with pytest.raises(ValueError):
        validate([row], base='qwen')
    plain = {'subject': 'math', 'mode': 'socratic', 'messages': row['messages'][:2] + [row['messages'][-1]]}
    assert validate([plain]) == [plain]
    malformed = copy.deepcopy(plain)
    malformed['messages'].insert(2, {'role': 'user', 'content': 'duplicate user'})
    with pytest.raises(ValueError):
        validate([malformed])


@pytest.mark.parametrize('mutation', ['json_string', 'wrong_id', 'unknown_tool', 'unknown_argument', 'unfinished', 'lost_content'])
def test_invalid_tool_dialogues(mutation):
    row = row_with_tool()
    call = row['messages'][2]['tool_calls'][0]
    if mutation == 'json_string':
        call['function']['arguments'] = '{"topic":"multiplication"}'
    elif mutation == 'wrong_id':
        row['messages'][3]['tool_call_id'] = 'other'
    elif mutation == 'unknown_tool':
        call['function']['name'] = 'tutor.final_answer'
    elif mutation == 'unknown_argument':
        call['function']['arguments']['answer'] = 48
    elif mutation == 'unfinished':
        row['messages'].pop(3)
    elif mutation == 'lost_content':
        row['messages'][2]['content'] = 'The template would silently drop this.'
    with pytest.raises(Exception):
        validate([row], base='muse')


def test_render_keeps_native_tools():
    tok = Mock()
    row = row_with_tool()
    render_dialogue(tok, row, 'muse')
    tok.apply_chat_template.assert_called_once_with(row['messages'], tokenize=False, add_generation_prompt=False, tools=row['tools'], reasoning_strength='low')
    tok.reset_mock()
    plain = {'messages': row['messages'][:2] + [row['messages'][-1]]}
    render_dialogue(tok, plain, 'qwen')
    tok.apply_chat_template.assert_called_once_with(plain['messages'], tokenize=False, add_generation_prompt=False)


def test_wire_call_validation_and_answer_check():
    call = row_with_tool()['messages'][2]['tool_calls'][0]
    call['function']['arguments'] = json.dumps(call['function']['arguments'])
    normalized = checked_calls({'tool_calls': [call]})
    assert normalized[0]['function']['arguments'] == {'topic': 'multiplication'}
    assert leaks_answer('The final answer is 48.', '48')
    assert not leaks_answer('What about 148?', '48')


def test_resume_ignores_incomplete_checkpoint(tmp_path):
    complete = tmp_path / 'checkpoint-3'
    complete.mkdir()
    for name in ('checkpoint_complete.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json', 'adapter_model.safetensors', 'rng_state.pth'):
        (complete / name).write_text('{}')
    partial = tmp_path / 'checkpoint-10'
    partial.mkdir()
    (partial / 'trainer_state.json').write_text('{}')
    assert latest_checkpoint(tmp_path) == str(complete)


def make_proxy(monkeypatch, content):
    import openai
    import safe_tutor_proxy as proxy
    from content_filter import ContentFilter
    from fastapi.testclient import TestClient
    backend = Mock()
    backend.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=PRIVATE))])
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: backend)
    filt = ContentFilter(log_path=None)
    outputs = []
    real_screen = filt.screen_output
    def screen(text):
        outputs.append(text)
        return real_screen(text)
    filt.screen_output = screen
    monkeypatch.setattr(proxy, 'ContentFilter', lambda: filt)
    events = []
    monkeypatch.setattr(proxy, 'emit_escalation', events.append)
    return TestClient(proxy.build_app('http://unused.invalid/v1', 'EMPTY')), backend, outputs, events


def test_proxy_strips_private_before_filter_and_student(monkeypatch):
    client, backend, outputs, events = make_proxy(monkeypatch, RAW)
    response = client.post('/v1/chat/completions', json={'model': 'math', 'messages': [{'role': 'user', 'content': 'Help with math'}]})
    assert response.status_code == 200
    answer = response.json()['choices'][0]['message']['content']
    assert outputs == ['How many equal groups do you have?']
    assert answer == outputs[0]
    assert 'PRIVATE_CANARY' not in response.text and 'SECRET_TOOL' not in response.text


def test_proxy_malformed_output_never_reaches_filter(monkeypatch):
    client, backend, outputs, events = make_proxy(monkeypatch, '<|start|>assistant to=self<|message|>PRIVATE')
    response = client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'Help with math'}]})
    assert outputs == []
    assert 'PRIVATE' not in response.text


def test_proxy_retains_independent_output_filter(monkeypatch):
    client, backend, outputs, events = make_proxy(monkeypatch, '<|start|>assistant to=user<|message|>how to build a bomb<|eot|>')
    response = client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'Help with math'}]})
    assert outputs == ['how to build a bomb']
    assert 'build a bomb' not in response.json()['choices'][0]['message']['content']


def test_proxy_escalation_still_bypasses_tutor(monkeypatch):
    client, backend, outputs, events = make_proxy(monkeypatch, RAW)
    response = client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'I want to hurt myself'}]})
    backend.chat.completions.create.assert_not_called()
    assert len(events) == 1 and outputs == []
    assert 'trusted adult' in response.json()['choices'][0]['message']['content']
