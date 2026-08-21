from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

STRUCTURED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".ipynb",
}


def _read_file_text(path: Path) -> str:
    """Read a text file trying several common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_csv(path: Path, max_rows: int = 3000) -> str:
    """Extract CSV or TSV file to a Markdown table."""
    raw_text = _read_file_text(path)
    if not raw_text.strip():
        return ""

    delimiter = "\t" if path.suffix.lower() == ".tsv" else None
    if not delimiter:
        sample = raw_text[:2048]
        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=",;\t|").delimiter
        except Exception:
            delimiter = ","

    reader = csv.reader(raw_text.splitlines(), delimiter=delimiter)
    rows: list[list[str]] = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        cleaned = [cell.strip().replace("\n", " ") for cell in row]
        if any(cleaned):
            rows.append(cleaned)

    if not rows:
        return ""

    headers = rows[0]
    num_cols = len(headers)
    output: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]
    for row in rows[1:]:
        padded = (row + [""] * num_cols)[:num_cols]
        output.append("| " + " | ".join(padded) + " |")

    if len(rows) >= max_rows:
        output.append(f"\n*(Truncado a {max_rows} filas)*")

    return "\n".join(output)


def extract_json(path: Path) -> str:
    """Extract JSON or JSONL file into formatted, readable text."""
    raw_text = _read_file_text(path)
    if not raw_text.strip():
        return ""

    if path.suffix.lower() == ".jsonl":
        return extract_jsonl(path)

    try:
        data = json.loads(raw_text)
    except Exception:
        return raw_text

    # If it's a list of dicts (like database exports / catalogs), format cleanly
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data[:10]):
        lines: list[str] = []
        for index, item in enumerate(data, start=1):
            lines.append(f"### Registro {index}\n" + json.dumps(item, indent=2, ensure_ascii=False))
        return "\n\n".join(lines)

    return json.dumps(data, indent=2, ensure_ascii=False)


def extract_jsonl(path: Path, max_records: int = 3000) -> str:
    """Extract JSON Lines (.jsonl) file into clean records."""
    raw_text = _read_file_text(path)
    records: list[str] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        if index > max_records:
            records.append(f"*(Truncado a {max_records} registros)*")
            break
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            records.append(f"### Registro {index}\n" + json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            records.append(f"### Línea {index}\n{line}")

    return "\n\n".join(records)


def extract_yaml(path: Path) -> str:
    """Extract and normalize YAML file."""
    import yaml

    raw_text = _read_file_text(path)
    if not raw_text.strip():
        return ""
    try:
        data = yaml.safe_load(raw_text)
        if isinstance(data, (dict, list)):
            return yaml.dump(data, sort_keys=False, allow_unicode=True)
    except Exception:
        pass
    return raw_text


def extract_html(path: Path) -> str:
    """Extract readable text and structure from HTML file using lxml or regex fallback."""
    import lxml.html

    raw_text = _read_file_text(path)
    if not raw_text.strip():
        return ""

    try:
        doc = lxml.html.fromstring(raw_text)
        # Remove scripts, styles, comments, meta, head
        for element in doc.xpath("//script | //style | //noscript | //head | //svg | //iframe"):
            element.drop_tree()
        text = doc.text_content()
        # Clean up excessive blank lines
        lines = [line.strip() for line in text.splitlines()]
        cleaned = [line for line in lines if line]
        return "\n\n".join(cleaned)
    except Exception:
        # Fallback to simple regex/strip
        import re
        no_script = re.sub(r"<(script|style).*?>.*?</\1>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
        no_tags = re.sub(r"<[^>]+>", " ", no_script)
        lines = [line.strip() for line in no_tags.splitlines() if line.strip()]
        return "\n\n".join(lines)


def extract_xml(path: Path) -> str:
    """Extract and format XML content."""
    import lxml.etree

    raw_text = _read_file_text(path)
    if not raw_text.strip():
        return ""
    try:
        tree = lxml.etree.fromstring(raw_text.encode("utf-8"))
        return lxml.etree.tostring(tree, encoding="unicode", pretty_print=True)
    except Exception:
        return raw_text


def extract_ipynb(path: Path) -> str:
    """Extract Jupyter Notebook (.ipynb) markdown, code cells and outputs into Markdown."""
    raw_text = _read_file_text(path)
    if not raw_text.strip():
        return ""

    try:
        nb = json.loads(raw_text)
    except Exception:
        return raw_text

    cells = nb.get("cells", [])
    output_parts: list[str] = []

    for index, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue

        if cell_type == "markdown":
            output_parts.append(source)
        elif cell_type == "code":
            code_block = f"```python\n{source}\n```"
            output_lines = [code_block]
            outputs = cell.get("outputs", [])
            output_texts: list[str] = []
            for out in outputs:
                if "text" in out:
                    out_text = "".join(out["text"]).strip()
                    if out_text:
                        output_texts.append(out_text)
                elif "data" in out and "text/plain" in out["data"]:
                    out_text = "".join(out["data"]["text/plain"]).strip()
                    if out_text:
                        output_texts.append(out_text)
            if output_texts:
                combined_out = "\n".join(output_texts)
                output_lines.append(f"> **Salida:**\n```\n{combined_out[:1000]}\n```")
            output_parts.append("\n\n".join(output_lines))
        elif cell_type == "raw":
            output_parts.append(f"```\n{source}\n```")

    return "\n\n---\n\n".join(output_parts).strip()


def extract_structured(path: Path) -> str:
    """Route structured document formats to appropriate extractors."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return extract_csv(path)
    if suffix in {".json", ".jsonl"}:
        return extract_json(path)
    if suffix in {".yaml", ".yml"}:
        return extract_yaml(path)
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    if suffix == ".xml":
        return extract_xml(path)
    if suffix == ".ipynb":
        return extract_ipynb(path)
    raise ValueError(f"Unsupported structured file type: {suffix}")
