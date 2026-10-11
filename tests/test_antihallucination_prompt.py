"""The system prompt must tell the agent when to refuse.

Motivated by a measured defect: asked for a fact absent from the knowledge
base, the local model answered with a fabricated price instead of abstaining.
These tests pin the wording that prevents it in both shipped languages.
"""

from knowledge.pydantic_agent import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ES, get_system_prompt

ABSTENTION_MARKERS_ES = [
    "Abstenci",
    "no lo s",
    "NO lo inventes",
    "no est",
    "disponible en la base de",
    "citar el fragmento",
]

ABSTENTION_MARKERS_EN = [
    "Abstention",
    "I don't know",
    "do NOT invent it",
    "not available in the knowledge base",
    "cite the specific",
    "unverified",
]


def test_es_prompt_tells_the_agent_to_abstain():
    for marker in ABSTENTION_MARKERS_ES:
        assert marker in SYSTEM_PROMPT_ES, f"falta {marker!r} en el prompt ES"


def test_en_prompt_tells_the_agent_to_abstain():
    for marker in ABSTENTION_MARKERS_EN:
        assert marker in SYSTEM_PROMPT_EN, f"falta {marker!r} en el prompt EN"


def test_es_prompt_requires_citing_before_asserting():
    assert "Antes de afirmar un dato num" in SYSTEM_PROMPT_ES
    assert "Si no puedes citarlo, no lo afirmes" in SYSTEM_PROMPT_ES


def test_en_prompt_requires_citing_before_asserting():
    assert "Before asserting a number" in SYSTEM_PROMPT_EN
    assert "If you cannot cite it, do not assert it" in SYSTEM_PROMPT_EN


def test_prompts_prefer_admitting_ignorance():
    """The core rule: an admitted gap beats a fabricated value."""
    assert "preferible decir" in SYSTEM_PROMPT_ES
    assert "is better than a fabricated figure" in SYSTEM_PROMPT_EN


def test_get_system_prompt_returns_language_variant():
    assert get_system_prompt("es") is SYSTEM_PROMPT_ES
    assert get_system_prompt("en") is SYSTEM_PROMPT_EN
    assert get_system_prompt("") is SYSTEM_PROMPT_ES
    assert get_system_prompt("fr") is SYSTEM_PROMPT_ES


def test_auto_language_prompt_still_has_the_directive():
    auto = get_system_prompt("auto")
    assert "Abstention" in auto
    assert "Language Matching Directive" in auto


def test_directive_lives_in_the_operating_directives():
    """Not in the tool catalogue, but with the behavioural rules."""
    es = SYSTEM_PROMPT_ES
    assert es.index("Abstenci") > es.index("DIRECTIVAS")
    en = SYSTEM_PROMPT_EN
    assert en.index("Abstention") > en.index("OPERATIONAL DIRECTIVES")


def test_directive_is_absent_from_the_tool_catalogue():
    """It is guidance, not a tool description."""
    es_tools = SYSTEM_PROMPT_ES[: SYSTEM_PROMPT_ES.index("DIRECTIVAS")]
    assert "Abstenci" not in es_tools
    en_tools = SYSTEM_PROMPT_EN[: SYSTEM_PROMPT_EN.index("OPERATIONAL DIRECTIVES")]
    assert "Abstention" not in en_tools
