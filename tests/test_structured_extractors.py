import json
from pathlib import Path

from ragpoc.extractors.structured import (
    extract_csv,
    extract_html,
    extract_ipynb,
    extract_json,
    extract_jsonl,
    extract_structured,
    extract_xml,
    extract_yaml,
)
from ragpoc.extractors.text import extract_text


def test_extract_csv_and_tsv(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,name,role\n1,Alice,Admin\n2,Bob,User\n", encoding="utf-8")
    
    extracted = extract_csv(csv_path)
    assert "| id | name | role |" in extracted
    assert "| 1 | Alice | Admin |" in extracted
    assert "| 2 | Bob | User |" in extracted
    assert extract_text(csv_path) == extracted

    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("key\tvalue\napi_key\tsecret123\n", encoding="utf-8")
    extracted_tsv = extract_csv(tsv_path)
    assert "| key | value |" in extracted_tsv
    assert "| api_key | secret123 |" in extracted_tsv


def test_extract_json_and_jsonl(tmp_path: Path):
    json_path = tmp_path / "users.json"
    data = [
        {"id": 1, "username": "admin", "active": True},
        {"id": 2, "username": "tester", "active": False},
    ]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    extracted = extract_json(json_path)
    assert "### Registro 1" in extracted
    assert '"username": "admin"' in extracted
    assert extract_text(json_path) == extracted

    jsonl_path = tmp_path / "events.jsonl"
    jsonl_path.write_text('{"event": "login", "user": "alice"}\n{"event": "logout", "user": "alice"}\n', encoding="utf-8")
    extracted_jsonl = extract_jsonl(jsonl_path)
    assert "### Registro 1" in extracted_jsonl
    assert '"event": "login"' in extracted_jsonl
    assert "### Registro 2" in extracted_jsonl
    assert '"event": "logout"' in extracted_jsonl


def test_extract_yaml(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("database:\n  host: localhost\n  port: 5432\n", encoding="utf-8")
    extracted = extract_yaml(yaml_path)
    assert "database:" in extracted
    assert "port: 5432" in extracted
    assert extract_text(yaml_path) == extracted


def test_extract_html_and_xml(tmp_path: Path):
    html_path = tmp_path / "page.html"
    html_path.write_text("""
    <!DOCTYPE html>
    <html>
      <head><title>Ignored Head</title><script>alert('malicious')</script></head>
      <body>
        <h1>Guía de Arquitectura</h1>
        <p>Este es el contenido principal del sistema.</p>
      </body>
    </html>
    """, encoding="utf-8")
    extracted_html = extract_html(html_path)
    assert "Guía de Arquitectura" in extracted_html
    assert "Este es el contenido principal del sistema." in extracted_html
    assert "alert" not in extracted_html

    xml_path = tmp_path / "data.xml"
    xml_path.write_text("<root><user><name>Charlie</name></user></root>", encoding="utf-8")
    extracted_xml = extract_xml(xml_path)
    assert "Charlie" in extracted_xml


def test_extract_ipynb(tmp_path: Path):
    ipynb_path = tmp_path / "analysis.ipynb"
    nb_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Análisis Exploratorio\n", "Este notebook procesa los datos."]
            },
            {
                "cell_type": "code",
                "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')\n", "print(df.head())"],
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["   id   name\n0   1  Alice\n"]
                    }
                ]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    ipynb_path.write_text(json.dumps(nb_content), encoding="utf-8")

    extracted = extract_ipynb(ipynb_path)
    assert "# Análisis Exploratorio" in extracted
    assert "```python\nimport pandas as pd" in extracted
    assert "Salida:" in extracted
    assert "Alice" in extracted
    assert extract_text(ipynb_path) == extracted
