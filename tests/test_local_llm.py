from pathlib import Path

import pytest

from ragpoc.budget import MemoryBudget
from ragpoc.local_llm import (
    ChatMessage,
    LocalLlmEngine,
    LocalLlmError,
    LocalLlmNotLoaded,
    MediaInput,
    close_thinking,
    read_context_length,
    resolve_model_path,
    split_thinking,
    strip_thinking,
)

OPEN = "<" + "think" + ">"
CLOSE = "</" + "think" + ">"
ASSISTANT_PROMPT = "<" + "|im_start|>assistant\n"


def think(body: str) -> str:
    return OPEN + body + CLOSE


# --- thinking split ---


def test_split_without_thinking_returns_whole_text():
    visible, reasoning = split_thinking("hola mundo")
    assert visible == "hola mundo"
    assert reasoning == ""


def test_split_extracts_reasoning_and_visible():
    visible, reasoning = split_thinking("hola" + think("pensando") + "respuesta")
    assert "respuesta" in visible
    assert "hola" in visible
    assert reasoning == "pensando"


def test_split_keeps_unterminated_thinking_as_reasoning():
    visible, reasoning = split_thinking(OPEN + "se corto el pensamiento")
    assert visible == ""
    assert "se corto el pensamiento" in reasoning


def test_strip_falls_back_to_reasoning_when_no_visible():
    assert strip_thinking(OPEN + "solo pensamiento sin respuesta") == "solo pensamiento sin respuesta"


def test_strip_prefers_visible_over_reasoning():
    assert strip_thinking(think("ruido") + "la respuesta") == "la respuesta"


# --- close_thinking ---


def test_close_thinking_injects_empty_closed_block():
    result = close_thinking(ASSISTANT_PROMPT, template_thinks=True)
    assert result.startswith(ASSISTANT_PROMPT.rstrip("\n"))
    assert result.endswith(OPEN + "\n" + CLOSE + "\n\n")


def test_close_thinking_is_noop_without_template_thinking():
    assert close_thinking(ASSISTANT_PROMPT, template_thinks=False) == ASSISTANT_PROMPT


def test_close_thinking_is_noop_when_already_closed():
    rendered = ASSISTANT_PROMPT + think("t") + "\n"
    assert close_thinking(rendered, template_thinks=True) == rendered


def test_close_thinking_is_noop_when_not_at_assistant_prompt():
    assert close_thinking("hola", template_thinks=True) == "hola"


def test_close_thinking_round_trips_through_split():
    rendered = close_thinking(ASSISTANT_PROMPT, template_thinks=True)
    visible, reasoning = split_thinking(rendered.split("assistant\n")[-1])
    assert visible == ""
    assert reasoning == ""


# --- media input ---


def test_media_input_classification():
    image = MediaInput(data=b"x", media_type="image/png")
    audio = MediaInput(data=b"x", media_type="audio/wav")
    assert image.is_image and not image.is_audio
    assert audio.is_audio and not audio.is_image
    assert image.extension == ".png"
    assert audio.extension == ".wav"


def test_media_input_unknown_extension():
    assert MediaInput(data=b"x", media_type="application/octet-stream").extension == ""


def test_chat_message_defaults():
    message = ChatMessage(role="user", content="hola")
    assert message.media == []
    assert message.tool_call_id is None
    assert message.name is None


# --- model path resolution ---


def test_resolve_model_path_missing_dir(tmp_path: Path):
    with pytest.raises(LocalLlmError, match="model directory not found"):
        resolve_model_path(tmp_path / "nope")


def test_resolve_model_path_finds_genai_tree(tmp_path: Path):
    target = tmp_path / "onnxruntime" / "cpu_and_mobile" / "cpu-int4-rtn-block-32"
    target.mkdir(parents=True)
    (target / "genai_config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_path(tmp_path) == target


def test_resolve_model_path_falls_back_to_any_genai_config(tmp_path: Path):
    target = tmp_path / "deep" / "nest"
    target.mkdir(parents=True)
    (target / "genai_config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_path(tmp_path) == target


def test_resolve_model_path_rejects_transformers_js_only_layout(tmp_path: Path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_q4f16.onnx").write_bytes(b"x")
    with pytest.raises(LocalLlmError, match="genai_config.json"):
        resolve_model_path(tmp_path)


def test_read_context_length(tmp_path: Path):
    (tmp_path / "genai_config.json").write_text(
        '{"model": {"context_length": 2048}}', encoding="utf-8"
    )
    assert read_context_length(tmp_path) == 2048
    assert read_context_length(tmp_path / "nope") is None


def test_read_context_length_survives_bad_json(tmp_path: Path):
    (tmp_path / "genai_config.json").write_text("{not json", encoding="utf-8")
    assert read_context_length(tmp_path) is None


# --- engine guards ---


def test_engine_rejects_unknown_dtype(tmp_path: Path):
    with pytest.raises(LocalLlmError, match="unsupported dtype"):
        LocalLlmEngine(tmp_path, dtype="int3")


def test_generate_before_load_raises(tmp_path: Path):
    engine = LocalLlmEngine(tmp_path)
    with pytest.raises(LocalLlmNotLoaded):
        engine.generate([ChatMessage(role="user", content="hola")])
    with pytest.raises(LocalLlmNotLoaded):
        engine.format_chat([ChatMessage(role="user", content="hola")])
    with pytest.raises(LocalLlmNotLoaded):
        list(engine.generate_stream([ChatMessage(role="user", content="hola")]))


def test_processor_before_load_raises(tmp_path: Path):
    with pytest.raises(LocalLlmNotLoaded):
        _ = LocalLlmEngine(tmp_path).processor


def test_count_tokens_before_load_raises(tmp_path: Path):
    with pytest.raises(LocalLlmNotLoaded):
        LocalLlmEngine(tmp_path).count_tokens("hola")


def test_budget_exceeded_refuses_load(tmp_path: Path):
    from ragpoc.local_llm import LocalLlmBudgetExceeded

    budget = MemoryBudget(hard_limit_bytes=1024)
    engine = LocalLlmEngine(tmp_path, budget=budget)
    with pytest.raises(LocalLlmBudgetExceeded):
        engine.load(budget_key="qwen3-vl-2b-q4f16")
