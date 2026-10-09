from pathlib import Path

from ragpoc.config import Settings
from ragpoc.model_downloader import get_downloader


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_upload_dir=tmp_path / "data" / "uploads",
        local_embedding_cache_dir=tmp_path / "models",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_model_status_not_downloaded(tmp_path):
    settings = make_settings(tmp_path)
    status = get_downloader(settings).get_status()
    assert status.state == "not_downloaded"
    assert status.downloaded_bytes == 0
    assert status.total_bytes == 850_000_000  # expected q8 size for UI progress


def test_model_status_downloaded(tmp_path):
    settings = make_settings(tmp_path)
    model_dir = settings.local_model_dir
    (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
    for f in [
        "config.json", "preprocessor_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json",
        "onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data",
        "onnx/vision_encoder_quantized.onnx", "onnx/vision_encoder_quantized.onnx_data",
        "onnx/audio_encoder_quantized.onnx", "onnx/audio_encoder_quantized.onnx_data",
    ]:
        p = model_dir / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 100)
    status = get_downloader(settings).get_status()
    assert status.state == "ready"


def test_model_status_partial(tmp_path):
    settings = make_settings(tmp_path)
    model_dir = settings.local_model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tokenizer.json").write_bytes(b"\x00" * 100)
    status = get_downloader(settings).get_status()
    assert status.state == "not_downloaded"


def test_downloader_is_singleton_per_settings(tmp_path):
    settings = make_settings(tmp_path)
    d1 = get_downloader(settings)
    d2 = get_downloader(settings)
    assert d1 is d2


def test_downloader_progress_callback(tmp_path):
    settings = make_settings(tmp_path)
    downloader = get_downloader(settings)
    events = []
    downloader.add_progress_listener(lambda e: events.append(e))
    # Simulate progress
    downloader._emit_progress(state="downloading", downloaded_bytes=50, total_bytes=100)
    assert len(events) == 1
    assert events[0]["state"] == "downloading"
    assert events[0]["percent"] == 50.0


def test_downloader_cancel(tmp_path):
    settings = make_settings(tmp_path)
    downloader = get_downloader(settings)
    downloader._cancel_flag = True
    assert downloader.is_cancelled


def test_model_status_bytes(tmp_path):
    settings = make_settings(tmp_path)
    model_dir = settings.local_model_dir
    (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
    for f in ["tokenizer.json", "onnx/model_quantized.onnx_data"]:
        p = model_dir / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 200)
    status = get_downloader(settings).get_status()
    assert status.downloaded_bytes == 400


def test_model_downloader_no_mark_file(tmp_path):
    settings = make_settings(tmp_path)
    downloader = get_downloader(settings)
    assert downloader.model_dir == settings.local_model_dir
