import datetime as dt

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import ToolDefinition

from knowledge.local_agent_model import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    history_to_messages,
    parse_tool_calls,
    to_tool_parts,
    tools_to_payload,
)

NOW = dt.datetime.now(dt.UTC)


def request(*parts) -> ModelRequest:
    return ModelRequest(parts=list(parts), timestamp=NOW, instructions=None)


def response(*parts) -> ModelResponse:
    return ModelResponse(parts=list(parts), timestamp=NOW)


def call(name: str, body: str) -> str:
    return TOOL_CALL_OPEN + body + TOOL_CALL_CLOSE


# --- parse_tool_calls ---


def test_parse_without_tool_calls_returns_text_unchanged():
    visible, calls = parse_tool_calls("hola mundo")
    assert visible == "hola mundo"
    assert calls == []


def test_parse_extracts_single_call():
    text = "voy a buscar" + call("search_web", '{"name": "search_web", "arguments": {"query": "rag"}}')
    visible, calls = parse_tool_calls(text)
    assert "voy a buscar" in visible
    assert TOOL_CALL_OPEN not in visible
    assert len(calls) == 1
    assert calls[0]["name"] == "search_web"
    assert calls[0]["arguments"] == {"query": "rag"}


def test_parse_extracts_multiple_calls():
    body = (
        call("a", '{"name": "a", "arguments": {"x": 1}}')
        + "y"
        + call("b", '{"name": "b", "arguments": {"y": 2}}')
    )
    visible, calls = parse_tool_calls(body)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert visible == "y"


def test_parse_tolerates_unclosed_block():
    text = 'intro' + TOOL_CALL_OPEN + '{"name": "go", "arguments": {"a": 1}}'
    visible, calls = parse_tool_calls(text)
    assert calls == []
    assert "intro" in visible


def test_parse_recovers_truncated_json():
    text = call("go", '{"name": "go", "arguments": {"a": 1, "b":')
    _, calls = parse_tool_calls(text)
    assert calls == [] or calls[0]["name"] == "go"


def test_parse_ignores_empty_block():
    assert parse_tool_calls(call("x", "")) == ("", [])


def test_parse_keeps_text_after_block():
    text = call("a", '{"name": "a", "arguments": {}}') + "fin"
    visible, calls = parse_tool_calls(text)
    assert visible == "fin"
    assert len(calls) == 1


# --- to_tool_parts ---


def test_to_tool_parts_builds_tool_call_parts():
    parts = to_tool_parts([{"name": "search_web", "arguments": {"query": "x"}}], "ignored")
    assert len(parts) == 1
    assert isinstance(parts[0], ToolCallPart)
    assert parts[0].tool_name == "search_web"
    assert parts[0].args == {"query": "x"}
    assert parts[0].tool_call_id


def test_to_tool_parts_accepts_function_wrapper():
    parts = to_tool_parts([{"function": {"name": "f", "arguments": {"a": 1}}}], "")
    assert parts[0].tool_name == "f"
    assert parts[0].args == {"a": 1}


def test_to_tool_parts_coerces_string_arguments():
    parts = to_tool_parts([{"name": "f", "arguments": '{"a": 2}'}], "")
    assert parts[0].args == {"a": 2}


def test_to_tool_parts_wraps_bare_string_arguments():
    parts = to_tool_parts([{"name": "f", "arguments": "texto plano"}], "")
    assert parts[0].args == {"input": "texto plano"}


def test_to_tool_parts_handles_missing_arguments():
    parts = to_tool_parts([{"name": "f"}], "")
    assert parts[0].args == {}


def test_to_tool_parts_skips_call_without_name():
    assert to_tool_parts([{"arguments": {}}], "") == []


# --- tools_to_payload ---


def test_tools_to_payload_shape():
    tool = ToolDefinition(
        name="search_knowledge_base",
        parameters_json_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        description="Busca en la base",
    )
    payload = tools_to_payload([tool])
    assert payload[0]["type"] == "function"
    assert payload[0]["function"]["name"] == "search_knowledge_base"
    assert payload[0]["function"]["description"] == "Busca en la base"
    assert payload[0]["function"]["parameters"]["type"] == "object"


def test_tools_to_payload_skips_non_function_kinds():
    tool = ToolDefinition(name="out", parameters_json_schema={}, kind="output")
    assert tools_to_payload([tool]) == []


def test_tools_to_payload_defaults_missing_schema():
    tool = ToolDefinition(name="f", parameters_json_schema=None, description=None)
    payload = tools_to_payload([tool])
    assert payload[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# --- history_to_messages ---


def test_history_maps_system_and_user():
    history = [
        request(
            SystemPromptPart(content="sistema"),
            UserPromptPart(content="pregunta"),
        )
    ]
    messages = history_to_messages(history)
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == "sistema"
    assert messages[1].content == "pregunta"


def test_history_maps_assistant_text():
    history = [response(TextPart(content="respuesta"))]
    messages = history_to_messages(history)
    assert messages[0].role == "assistant"
    assert messages[0].content == "respuesta"


def test_history_round_trips_tool_call():
    history = [response(ToolCallPart(tool_name="search_web", args={"query": "x"}, tool_call_id="id1"))]
    messages = history_to_messages(history)
    assert messages[0].role == "assistant"
    assert TOOL_CALL_OPEN in messages[0].content
    assert "search_web" in messages[0].content


def test_history_maps_tool_return():
    history = [
        request(
            ToolReturnPart(tool_name="search_web", content="resultado", tool_call_id="id1"),
        )
    ]
    messages = history_to_messages(history)
    assert messages[0].role == "tool"
    assert messages[0].tool_call_id == "id1"
    assert messages[0].name == "search_web"
    assert "resultado" in messages[0].content


def test_history_extracts_binary_evidence_as_media():
    history = [
        request(
            UserPromptPart(
                content=[
                    "mira esto",
                    BinaryContent(data=b"\x89PNG", media_type="image/png"),
                ]
            )
        )
    ]
    messages = history_to_messages(history)
    assert messages[0].role == "user"
    assert messages[0].content == "mira esto"
    assert len(messages[0].media) == 1
    assert messages[0].media[0].is_image
    assert messages[0].media[0].data == b"\x89PNG"


def test_history_collects_media_from_every_turn():
    history = [
        request(UserPromptPart(content=[BinaryContent(data=b"a", media_type="audio/wav")])),
        response(TextPart(content="ok")),
        request(UserPromptPart(content=[BinaryContent(data=b"b", media_type="image/png")])),
    ]
    messages = history_to_messages(history)
    assert sum(len(m.media) for m in messages) == 2


# --- bare (unwrapped) tool calls ---


def test_parse_accepts_bare_json_call():
    text = '{"name": "search_knowledge_base", "arguments": {"query": "rag"}}'
    visible, calls = parse_tool_calls(text)
    assert visible == ""
    assert len(calls) == 1
    assert calls[0]["name"] == "search_knowledge_base"
    assert calls[0]["arguments"] == {"query": "rag"}


def test_parse_bare_call_survives_surrounding_whitespace():
    text = '\n\n  {"name": "go", "arguments": {"a": 1}}\n'
    _, calls = parse_tool_calls(text)
    assert len(calls) == 1


def test_parse_bare_call_rejects_plain_prose_starting_with_brace():
    text = '{"esto": "no es una llamada porque no tiene name ni arguments"}'
    visible, calls = parse_tool_calls(text)
    assert calls == []
    assert visible


def test_parse_bare_call_rejects_ordinary_text():
    visible, calls = parse_tool_calls("La similitud coseno compara dos vectores.")
    assert calls == []
    assert "vectores" in visible


def test_wrapped_call_still_wins_over_bare():
    text = 'hola' + call("a", '{"name": "a", "arguments": {}}')
    _, calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a"]


def test_bare_call_becomes_tool_call_part():
    _, calls = parse_tool_calls('{"name": "f", "arguments": {"x": 1}}')
    parts = to_tool_parts(calls, "")
    assert parts[0].tool_name == "f"
    assert parts[0].args == {"x": 1}
