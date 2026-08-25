import json
from base64 import b64encode
import pytest
from django.test import Client

from knowledge.models import AppConfig
from knowledge.settings_store import get_current_language, set_app_settings, get_api_key_status
from knowledge.pydantic_agent import get_system_prompt, SYSTEM_PROMPT_ES, SYSTEM_PROMPT_EN
from knowledge.views import get_artifact_title, get_artifact_system_prompt
from knowledge.artifact_media import describe_artifact_settings
from ragpoc.config import Settings


def auth_header() -> dict[str, str]:
    return {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}


@pytest.fixture(autouse=True)
def setup_django_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        data_dir=data_dir,
        allowed_upload_dir=uploads_dir,
        ui_password="test-password",
    )
    from ragpoc import config
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    yield


@pytest.mark.django_db
def test_language_db_persistence():
    # Initial state default is "es"
    assert get_current_language() == "es"

    # Set to English
    set_app_settings(language="en")
    assert get_current_language() == "en"
    config = AppConfig.load()
    assert config.language == "en"

    # Status dictionary includes language
    status = get_api_key_status()
    assert status["language"] == "en"

    # Switch back to Spanish
    set_app_settings(language="es")
    assert get_current_language() == "es"


@pytest.mark.django_db
def test_language_settings_api():
    client = Client()

    # GET /api/settings
    resp = client.get("/api/settings", **auth_header())
    assert resp.status_code == 200
    data = resp.json()
    assert "language" in data
    assert data["language"] == "es"

    # POST /api/settings with language = "en"
    post_resp = client.post(
        "/api/settings",
        data=json.dumps({"language": "en"}),
        content_type="application/json",
        **auth_header(),
    )
    assert post_resp.status_code == 200
    assert post_resp.json()["language"] == "en"
    assert get_current_language() == "en"

    # Verify GET now returns "en"
    resp2 = client.get("/api/settings", **auth_header())
    assert resp2.json()["language"] == "en"

    # Reset back to "es"
    client.post(
        "/api/settings",
        data=json.dumps({"language": "es"}),
        content_type="application/json",
        **auth_header(),
    )
    assert get_current_language() == "es"


def test_system_prompts_bilingual():
    prompt_es = get_system_prompt("es")
    prompt_en = get_system_prompt("en")

    assert "asistente de conocimiento" in prompt_es
    assert "knowledge assistant" in prompt_en
    assert prompt_es == SYSTEM_PROMPT_ES
    assert prompt_en == SYSTEM_PROMPT_EN
    assert prompt_es != prompt_en

    # Defaults to Spanish if invalid language passed
    assert get_system_prompt("unknown") == SYSTEM_PROMPT_ES


def test_artifact_titles_and_prompts_bilingual():
    types = ["podcast", "study_guide", "flashcards", "quiz", "diagram", "mindmap", "summary", "infographic", "timeline", "table"]

    for t in types:
        title_es = get_artifact_title(t, language="es")
        title_en = get_artifact_title(t, language="en")
        assert title_es, f"Missing Spanish title for {t}"
        assert title_en, f"Missing English title for {t}"

    # Specifically check titles that differ between ES and EN
    assert get_artifact_title("study_guide", language="es") == "Guía de Estudio"
    assert get_artifact_title("study_guide", language="en") == "Study Guide"
    assert get_artifact_title("mindmap", language="es") == "Mapa Mental"
    assert get_artifact_title("mindmap", language="en") == "Mind Map"
    assert get_artifact_title("timeline", language="es") == "Línea de Tiempo"
    assert get_artifact_title("timeline", language="en") == "Timeline"
    assert get_artifact_title("summary", language="es") == "Resumen Ejecutivo"
    assert get_artifact_title("summary", language="en") == "Executive Summary"
    assert get_artifact_title("table", language="es") == "Tabla de Datos"
    assert get_artifact_title("table", language="en") == "Data Table"
    assert get_artifact_title("diagram", language="es") == "Diagrama de Arquitectura"
    assert get_artifact_title("diagram", language="en") == "Architecture Diagram"

    prompt_types = ["study_guide", "flashcards", "quiz", "diagram", "mindmap", "summary"]
    for pt in prompt_types:
        prompt_es = get_artifact_system_prompt(pt, language="es")
        prompt_en = get_artifact_system_prompt(pt, language="en")
        assert prompt_es, f"Missing Spanish prompt for {pt}"
        assert prompt_en, f"Missing English prompt for {pt}"
        assert prompt_es != prompt_en, f"Spanish and English prompts should differ for {pt}"


def test_describe_artifact_settings_bilingual():
    es_desc = describe_artifact_settings(
        "podcast",
        {"duration": "corto", "tone": "informal"},
        language="es"
    )
    en_desc = describe_artifact_settings(
        "podcast",
        {"duration": "corto", "tone": "informal"},
        language="en"
    )

    assert "corto" in es_desc.lower()
    assert "short" in en_desc.lower()
    assert "informal" in es_desc.lower()
    assert "informal" in en_desc.lower()
