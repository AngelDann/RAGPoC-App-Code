from __future__ import annotations

from pathlib import Path

from ragpoc.extractors.office import OFFICE_SUFFIXES, extract_office
from ragpoc.extractors.structured import STRUCTURED_SUFFIXES, extract_structured

CODE_AND_CONFIG_SUFFIXES = {
    # Plain text / docs
    ".txt",
    ".md",
    ".markdown",
    ".rtf",
    ".rst",
    ".log",
    # Scripting and configs
    ".toml",
    ".ini",
    ".env",
    ".cfg",
    ".conf",
    ".sql",
    ".properties",
    # Programming languages
    ".py",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".cs",
    ".java",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".sh",
    ".bash",
    ".zsh",
    ".bat",
    ".cmd",
    ".ps1",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".graphql",
    ".proto",
    ".dockerfile",
}

ALL_TEXT_MEDIA_SUFFIXES = CODE_AND_CONFIG_SUFFIXES | STRUCTURED_SUFFIXES | OFFICE_SUFFIXES
TEXT_SUFFIXES = ALL_TEXT_MEDIA_SUFFIXES


def extract_text(path: Path) -> str:
    """Extract and format text from code, structured data, office files or plain text."""
    suffix = path.suffix.lower()
    if suffix in OFFICE_SUFFIXES:
        return extract_office(path)
    if suffix in STRUCTURED_SUFFIXES:
        return extract_structured(path)
    if suffix in CODE_AND_CONFIG_SUFFIXES:
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported text file: {suffix}")
