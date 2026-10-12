import threading
from pathlib import Path

from ragpoc.chat_models import CHAT_MODELS, ChatModelDownloader, get_chat_downloader
from ragpoc.config import Settings


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_upload_dir=tmp_path / "data",
        local_chat_cache_dir=tmp_path / "models",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _install(tmp_path: Path, key: str) -> Path:
    """Create the minimum file layout a manifest considers installed."""
    manifest = CHAT_MODELS[key]
    root = tmp_path / "models" / key
    if manifest.subdir:
        root = root / manifest.subdir
    root.mkdir(parents=True, exist_ok=True)
    for name in manifest.required:
        (root / name).write_text("{}", encoding="utf-8")
    return root


def test_manifest_keys_are_stable():
    assert {"qwen3-vl-2b", "qwen3-0.6b", "whisper-tiny"} <= set(CHAT_MODELS)


def test_whisper_manifest_points_at_a_genai_export():
    """onnx-community/whisper-tiny has no genai_config; this must not be used."""
    assert CHAT_MODELS["whisper-tiny"].repo_id != "onnx-community/whisper-tiny"
    assert "genai_config.json" in CHAT_MODELS["whisper-tiny"].required


def test_unknown_model_key_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unknown chat model"):
        ChatModelDownloader(make_settings(tmp_path), "nope")


def test_is_installed_false_when_absent(tmp_path):
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-vl-2b")
    assert downloader.is_installed() is False


def test_is_installed_true_when_files_present(tmp_path):
    _install(tmp_path, "qwen3-vl-2b")
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-vl-2b")
    assert downloader.is_installed() is True


def test_get_status_reports_ready_for_model_already_on_disk(tmp_path):
    """Regression: a previously downloaded model read as ``not_downloaded``."""
    _install(tmp_path, "qwen3-vl-2b")
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-vl-2b")
    status = downloader.get_status()
    assert status.state == "ready"
    assert status.downloaded_bytes == status.total_bytes > 0


def test_get_status_reports_not_downloaded_when_absent(tmp_path):
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-0.6b")
    assert downloader.get_status().state == "not_downloaded"


def test_status_to_dict_shape(tmp_path):
    _install(tmp_path, "qwen3-0.6b")
    payload = ChatModelDownloader(make_settings(tmp_path), "qwen3-0.6b").get_status().to_dict()
    assert set(payload) == {
        "state",
        "model_key",
        "downloaded_bytes",
        "total_bytes",
        "percent",
        "message",
        "error",
    }
    assert payload["model_key"] == "qwen3-0.6b"
    assert payload["percent"] == 100.0


def test_install_dir_uses_manifest_subdir(tmp_path):
    settings = make_settings(tmp_path)
    vl = ChatModelDownloader(settings, "qwen3-vl-2b")
    whisper = ChatModelDownloader(settings, "whisper-tiny")
    assert vl.install_dir.name.endswith("cpu-int4-rtn-block-32")
    # whisper's manifest has no subdir, so it installs at the model root
    assert whisper.install_dir == whisper.model_dir


def test_download_blocking_short_circuits_when_installed(tmp_path):
    _install(tmp_path, "qwen3-vl-2b")
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-vl-2b")
    status = downloader.download_blocking()
    assert status.state == "ready"
    assert status.message == "already installed"


def test_cancel_and_reset_are_safe(tmp_path):
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-vl-2b")
    downloader.cancel()
    assert downloader.is_cancelled is True
    downloader.reset_cancel()
    assert downloader.is_cancelled is False


def test_get_chat_downloader_returns_a_shared_instance(tmp_path):
    """Regression: views built a fresh downloader per request.

    The download thread ran on instance A while ``/api/chat-model/status``
    polled instance B, whose status was always ``not_downloaded`` -- the
    progress bar vanished and the UI said "Modelo no descargado" while the
    model was still downloading. Cancel was equally lost.
    """
    settings = make_settings(tmp_path)
    assert get_chat_downloader(settings, "qwen3-vl-2b") is get_chat_downloader(settings, "qwen3-vl-2b")


def test_download_state_is_visible_from_another_call(tmp_path):
    settings = make_settings(tmp_path)
    downloader = get_chat_downloader(settings, "qwen3-0.6b")
    downloader._set(state="downloading", downloaded_bytes=10, total_bytes=100)
    status = get_chat_downloader(settings, "qwen3-0.6b").get_status()
    assert status.state == "downloading"
    assert status.downloaded_bytes == 10


def test_cancel_reaches_the_downloading_instance(tmp_path):
    settings = make_settings(tmp_path)
    downloader = get_chat_downloader(settings, "qwen3-vl-2b")
    get_chat_downloader(settings, "qwen3-vl-2b").cancel()
    assert downloader.is_cancelled is True
    downloader.reset_cancel()


def test_a_new_download_clears_a_stale_cancel(tmp_path):
    """A cancelled download must not poison every later attempt."""
    downloader = ChatModelDownloader(make_settings(tmp_path), "qwen3-vl-2b")
    downloader.cancel()
    started = threading.Event()
    release = threading.Event()
    original = downloader._download_files

    def fake_download():
        started.set()
        release.wait(timeout=5)

    downloader._download_files = fake_download
    thread = threading.Thread(target=downloader.download_blocking, daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    release.set()
    thread.join(timeout=5)
    downloader._download_files = original
    assert downloader.get_status().message != "cancelled"
