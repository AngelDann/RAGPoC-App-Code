"""The Ajustes dialog must stay comfortable to scan.

These tests pin the layout decisions from the better-ui pass so a later edit
cannot silently undo them: the dialog width, the two-column grid, and above all
the concentric border radii -- card radius minus padding equals the nested
radius. Mismatched radii on nested elements are the most common reason an
interface reads as "off" without anyone being able to say why.
"""

from base64 import b64encode

import pytest
from django.test import Client

from ragpoc.config import Settings

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def isolate_env_settings(tmp_path, monkeypatch):
    test_settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_upload_dir=tmp_path / "data",
        ui_password="test-password",
    )
    import knowledge.settings_store as settings_store
    from ragpoc import config

    monkeypatch.setattr(settings_store, "get_env_settings", lambda: test_settings)
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    yield


@pytest.fixture()
def console_html() -> str:
    session = Client()
    session.defaults.update(
        {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}
    )
    response = session.get("/")
    assert response.status_code == 200
    return response.content.decode("utf-8")


def test_settings_dialog_has_its_own_class(console_html):
    assert 'craft-dialog craft-dialog-settings' in console_html


SETTINGS_BLOCK = "Ajustes: layout de confort (better-ui)"


def _settings_css(console_html: str) -> str:
    start = console_html.index(SETTINGS_BLOCK)
    end = console_html.index("@keyframes dialogFade", start)
    return console_html[start:end]


def test_settings_dialog_is_not_pinned_to_a_narrow_width(console_html):
    """560px forced six cards into one long scrollable column."""
    block = _settings_css(console_html)
    assert "max-width: 880px" in block
    assert "max-width: 560px" not in block


def test_settings_body_is_a_grid(console_html):
    assert 'class="settings-body' in console_html
    assert "grid-template-columns: repeat(auto-fit, minmax(330px, 1fr))" in console_html


def test_grid_does_not_stretch_cards_to_the_tallest(console_html):
    """Stretching left a large void next to the tall provider card."""
    assert "align-items: start;" in console_html


def test_nested_radius_is_computed_not_picked(console_html):
    assert "border-radius: calc(var(--radius-xl) - 16px);" in console_html


def test_no_inline_radius_left_in_the_settings_dialog(console_html):
    """Inline border-radius beats the stylesheet and defeats the rule above."""
    start = console_html.index('craft-dialog craft-dialog-settings')
    end = console_html.index('modal-backdrop-custom', start)
    region = console_html[start:end]
    assert "border-radius: var(--radius-sm)" not in region


def test_settings_cards_stagger_their_entrance(console_html):
    block = _settings_css(console_html)
    for delay in ("0ms", "70ms", "140ms", "210ms", "280ms", "350ms"):
        assert f"animation-delay: {delay}" in block
    assert "@keyframes settingsCardIn" in block


def test_reduced_motion_disables_the_stagger(console_html):
    block = _settings_css(console_html)
    assert "prefers-reduced-motion: reduce" in block
    reduced = block.split("prefers-reduced-motion: reduce")[1]
    # The rule targets the cards rather than naming the keyframe: what matters
    # is that the entrance animation is switched off.
    assert "settings-body > .craft-card" in reduced
    assert "animation: none" in reduced


def test_narrow_viewport_collapses_to_one_column(console_html):
    block = _settings_css(console_html)
    assert "@media (max-width: 720px)" in block
    narrow = block.split("@media (max-width: 720px)")[1]
    assert "grid-template-columns: 1fr" in narrow
    # On one column the stagger would read as latency, not hierarchy.
    assert "animation: none" in narrow


def test_icon_column_is_fixed_for_optical_alignment(console_html):
    """Glyphs have different advance widths; titles must still line up."""
    assert "flex: 0 0 18px" in console_html