"""Validate the Hugging Face message dictionaries consumed by Muse's ATEM template."""
from collections.abc import Mapping


def tool_definitions(tools):
    from jsonschema import Draft202012Validator
    definitions = {}
    if not isinstance(tools, list) or not tools:
        raise ValueError('Tool dialogues require a nonempty tools list')
    for tool in tools:
        fn = tool.get('function', {})
        if tool.get('type') != 'function' or not isinstance(fn.get('name'), str) or not fn['name']:
            raise ValueError('Expected a named function tool')
        if fn['name'] in definitions:
            raise ValueError('Duplicate tool name')
        schema = fn.get('parameters')
        if not isinstance(schema, dict) or schema.get('type') != 'object':
            raise ValueError('Tool parameters must be an object JSON schema')
        Draft202012Validator.check_schema(schema)
        definitions[fn['name']] = schema
    return definitions


def validate_call(call, definitions):
    from jsonschema import Draft202012Validator
    if not isinstance(call, dict) or call.get('type') != 'function':
        raise ValueError('Expected a function tool call')
    fn = call.get('function', {})
    name = fn.get('name')
    if name not in definitions:
        raise ValueError(f'Unknown tool: {name!r}')
    arguments = fn.get('arguments')
    if not isinstance(arguments, Mapping):
        raise ValueError('Native Muse tool arguments must be dictionaries, not JSON strings')
    Draft202012Validator(definitions[name]).validate(arguments)
    return name


def validate_tool_dialogue(row):
    definitions = tool_definitions(row.get('tools'))
    messages = row['messages']
    expected = 'user'
    pending = {}
    seen_ids = set()
    for message in messages[1:]:
        role = message.get('role')
        content = message.get('content')
        if content is not None and not isinstance(content, str):
            raise ValueError('This tutoring pipeline accepts text-only message content')
        if message.get('reasoning_content') is not None:
            if role != 'assistant' or not isinstance(message['reasoning_content'], str):
                raise ValueError('reasoning_content must be assistant-only text')
        if role == 'tool':
            call_id = message.get('tool_call_id')
            if not pending or call_id not in pending:
                raise ValueError('Tool result must resolve a pending tool_call_id')
            name = pending.pop(call_id)
            if message.get('name', name) != name or not isinstance(content, str):
                raise ValueError('Tool result name/content does not match the call')
            if not pending:
                expected = 'assistant'
            continue
        if pending:
            raise ValueError('Resolve all tool calls before the next dialogue turn')
        if role != expected:
            raise ValueError(f'Expected {expected} turn, got {role}')
        if role == 'user':
            if not isinstance(content, str) or message.get('tool_calls'):
                raise ValueError('User turns must contain text, not tool calls')
            expected = 'assistant'
            continue
        calls = message.get('tool_calls')
        if calls:
            if not isinstance(calls, list) or content not in (None, ''):
                raise ValueError('Tool-call turns need a list of calls and empty content; the template drops content on these turns')
            for call in calls:
                name = validate_call(call, definitions)
                call_id = call.get('id')
                if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
                    raise ValueError('Each tool call needs a unique nonempty id')
                seen_ids.add(call_id)
                pending[call_id] = name
            expected = 'tool'
        elif message.get('recipient', 'user') == 'self':
            if not isinstance(content, str):
                raise ValueError('Private reasoning turn must contain text')
            expected = 'assistant'
        else:
            if message.get('recipient', 'user') != 'user' or not isinstance(content, str):
                raise ValueError('Use structured tool_calls for tools and recipient=user for student output')
            if message.get('end_turn') is False:
                raise ValueError('Student-facing turns must end the assistant turn')
            expected = 'user'
    if pending or expected != 'user' or messages[-1].get('role') != 'assistant':
        raise ValueError('Dialogue must end with student-facing assistant content after all tools resolve')
