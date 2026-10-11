"""Whisper must be resident only while audio is actually being transcribed.

These tests exercise the load/unload lifecycle without touching ONNX: the
point under test is the memory policy, not inference.
"""

from pathlib import Path

from knowledge.local_agent_model import LocalAgentModel
from ragpoc.local_llm import ChatMessage, MediaInput
from ragpoc.whisper_transcribe import WhisperTranscriber, _clean_transcript


class FakeTranscriber(WhisperTranscriber):
    """Records lifecycle calls instead of loading a model."""

    def __init__(self):
        super().__init__(model_dir=Path("unused"))
        self.events: list[str] = []
        self.loaded = False

    def load(self) -> None:
        self.events.append("load")
        self.loaded = True

    def unload(self) -> None:
        self.events.append("unload")
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def transcribe(self, media: MediaInput) -> str:
        self.events.append("transcribe")
        return "hola"


def test_session_loads_and_unloads():
    tr = FakeTranscriber()
    with tr.session():
        assert tr.is_loaded is True
    assert tr.events == ["load", "unload"]
    assert tr.is_loaded is False


def test_session_returns_transcriber_for_chaining():
    tr = FakeTranscriber()
    with tr.session() as inner:
        assert inner is tr


def test_session_unloads_even_when_body_raises():
    tr = FakeTranscriber()
    try:
        with tr.session():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tr.events == ["load", "unload"]


def test_session_keeps_an_already_loaded_model_resident():
    """A caller that loaded deliberately must not have it yanked away."""
    tr = FakeTranscriber()
    tr.load()
    with tr.session():
        assert tr.is_loaded is True
    assert tr.events == ["load", "load"]
    assert tr.is_loaded is True


def test_session_transcribes_during_the_window():
    tr = FakeTranscriber()
    with tr.session() as inner:
        assert inner.transcribe(MediaInput(data=b"x", media_type="audio/wav")) == "hola"
    assert tr.events == ["load", "transcribe", "unload"]


# --- transcript cleanup ---


def test_clean_transcript_strips_control_markers():
    assert _clean_transcript("<|startoftranscript|><|es|><|transcribe|><|notimestamps|>hola mundo") == "hola mundo"


def test_clean_transcript_keeps_text_around_timestamps():
    """Whisper interleaves ``<|0.96|>`` with speech; the text must survive."""
    assert _clean_transcript("hola<|0.00|>adios<|1.20|>fin") == "hola adios fin"


def test_clean_transcript_collapses_whitespace():
    assert _clean_transcript("  hola    mundo  ") == "hola mundo"


def test_clean_transcript_empty_is_safe():
    assert _clean_transcript("") == ""


# --- audio detection feeds the lazy path ---

def test_local_agent_model_detects_audio():
    message = ChatMessage(
        role="user", content="x", media=[MediaInput(data=b"a", media_type="audio/wav")]
    )
    assert LocalAgentModel._has_audio([message]) is True


def test_local_agent_model_ignores_images():
    message = ChatMessage(role="user", content="x", media=[MediaInput(data=b"a", media_type="image/png")])
    assert LocalAgentModel._has_audio([message]) is False


def test_local_agent_model_handles_no_media():
    assert LocalAgentModel._has_audio([ChatMessage(role="user", content="x")]) is False
