"""Unit tests for the real-retrieval gold builder and evaluator scripts.

Pure functions and a tmp_path sandbox only: no database, no network, no model.
The scripts live in scripts/ and are imported by path, mirroring the pattern in
tests/integration_local_embedding.py.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_retrieval_gold as gold  # noqa: E402
import run_retrieval_eval_real as eval_real  # noqa: E402

from ragpoc.embeddings import EmbeddingProvider  # noqa: E402

# --- fixtures and helpers ---


def make_rows() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "La arquitectura limpia separa las capas del sistema.",
            "filename": "manual_arquitectura.txt",
        },
        {
            "chunk_id": "c2",
            "document_id": "d2",
            "text": "La arquitectura limpia separa las capas del sistema.",
            "filename": "nota.txt",
        },
        {
            "chunk_id": "c3",
            "document_id": "d3",
            "text": "Protocolo de seguridad\n\nEl codigo de acceso confidencial del servidor nunca se comparte.",
            "filename": "protocolo_seguridad.txt",
        },
    ]


def make_gold_cases() -> list[dict]:
    return [
        {
            "query": "q",
            "relevant_chunk_ids": ["c1"],
            "relevant_document_ids": ["d1"],
            "strategy": "first_sentence",
            "case_id": "g000",
        }
    ]


def make_eval_output() -> dict:
    return {
        "summary": {
            "cases": 2,
            "mode": "chunk",
            "recall@1": 0.5,
            "recall@3": 1.0,
            "precision@1": 0.5,
            "hit_rate@1": 0.5,
            "ndcg@3": 0.8,
            "mrr": 0.75,
            "latency_p50_s": 0.01,
            "latency_p95_s": 0.02,
            "latency_max_s": 0.03,
        },
        "per_case": [
            {"case_id": "g000", "query": "q", "retrieved": ["c1"], "scores": [0.9], "latency_s": 0.01}
        ],
    }


# --- gold query derivation ---


def test_first_sentence_query_stops_at_sentence_end():
    assert gold.first_sentence_query("Primera frase. Segunda frase.") == "Primera frase."


def test_first_sentence_query_skips_slug_style_headers():
    text = "nota_nivel_cuaderno\n\nContenido de prueba a nivel cuaderno."
    assert gold.first_sentence_query(text) == "Contenido de prueba a nivel cuaderno."


def test_first_sentence_query_rejects_tiny_fragments():
    assert gold.first_sentence_query("Hola.") is None


def test_mid_phrase_query_takes_words_from_the_middle():
    text = "uno dos tres cuatro cinco seis siete ocho nueve diez"
    assert gold.mid_phrase_query(text) == "tres cuatro cinco seis siete"


def test_mid_phrase_query_needs_a_long_chunk():
    assert gold.mid_phrase_query("uno dos tres cuatro cinco") is None


def test_title_query_strips_hashes_and_extension():
    assert gold.title_query("210b2b290f2b2527-Nota agente.txt") == "Nota agente"


def test_title_query_rejects_uninformative_filenames():
    assert gold.title_query("210b2b290f2b2527.txt") is None
    assert gold.title_query("doc1.txt") is None


# --- gold case building ---


def test_build_cases_groups_duplicate_texts_into_one_case():
    cases, stats = gold.build_cases(make_rows(), min_chars=20)
    sentence = next(case for case in cases if case["query"].startswith("La arquitectura"))
    assert sentence["relevant_chunk_ids"] == ["c1", "c2"]
    assert sentence["relevant_document_ids"] == ["d1", "d2"]
    assert stats == {"chunks_considered": 3, "unique_texts": 2, "total_cases": 6}


def test_build_cases_skips_short_chunks():
    rows = [{"chunk_id": "c1", "document_id": "d1", "text": "corto.", "filename": "doc1.txt"}]
    cases, stats = gold.build_cases(rows, min_chars=50)
    assert cases == []
    assert stats == {"chunks_considered": 0, "unique_texts": 0, "total_cases": 0}


def test_build_cases_merges_queries_across_strategies():
    rows = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "Guía de Caché\n\nLa caché acelera las consultas recurrentes del sistema.",
            "filename": "Guía de Caché.txt",
        }
    ]
    cases, _ = gold.build_cases(rows, min_chars=10)
    merged = [case for case in cases if case["query"] == "Guía de Caché"]
    assert len(merged) == 1
    assert merged[0]["strategy"] == "first_sentence+title_filename"
    assert merged[0]["relevant_chunk_ids"] == ["c1"]
    assert merged[0]["relevant_document_ids"] == ["d1"]


def test_build_cases_limit_is_deterministic_per_seed():
    rows = [
        {
            "chunk_id": f"c{index}",
            "document_id": f"d{index}",
            "text": f"Tema {index}: "
            + " ".join(f"palabra{index}x{step}" for step in range(12))
            + f" final {index}.",
            "filename": f"archivo_{index}.txt",
        }
        for index in range(6)
    ]
    full, _ = gold.build_cases(rows, min_chars=10)
    assert len(full) == 12
    first, _ = gold.build_cases(rows, min_chars=10, limit=3, seed=7)
    second, _ = gold.build_cases(rows, min_chars=10, limit=3, seed=7)
    assert first == second
    assert len(first) == 3
    assert [case["case_id"] for case in first] == ["g000", "g001", "g002"]
    assert {case["query"] for case in first} <= {case["query"] for case in full}


def test_build_cases_with_no_rows_is_empty():
    cases, stats = gold.build_cases([], min_chars=10)
    assert cases == []
    assert stats["total_cases"] == 0


# --- gold schema validation ---


def test_validate_gold_set_accepts_built_cases():
    cases, _ = gold.build_cases(make_rows(), min_chars=20)
    assert gold.validate_gold_set({"meta": {"total_cases": len(cases)}, "cases": cases}) == []


def test_validate_gold_set_reports_schema_problems():
    payload = {
        "meta": {},
        "cases": [
            {"query": " ", "relevant_chunk_ids": ["c"], "relevant_document_ids": ["d"]},
            {"query": "q", "relevant_chunk_ids": [], "relevant_document_ids": ["d"]},
        ],
    }
    problems = gold.validate_gold_set(payload)
    assert any("query" in problem for problem in problems)
    assert any("relevant_chunk_ids" in problem for problem in problems)


def test_validate_gold_set_flags_duplicate_queries():
    case = {"query": "hola", "relevant_chunk_ids": ["c"], "relevant_document_ids": ["d"]}
    problems = gold.validate_gold_set({"meta": {}, "cases": [case, dict(case)]})
    assert any("duplicate" in problem for problem in problems)


def test_validate_gold_set_requires_meta_and_cases():
    assert gold.validate_gold_set({}) != []
    assert gold.validate_gold_set({"meta": {}, "cases": "x"}) != []
    assert gold.validate_gold_set({"meta": {}, "cases": []}) != []
    assert gold.validate_gold_set("nope") == ["gold set must be a JSON object"]


def test_gold_round_trip_preserves_cases(tmp_path):
    path = tmp_path / "nested" / "gold.json"
    meta = {"total_cases": 1}
    gold.write_gold(path, make_gold_cases(), meta)
    loaded_cases, loaded_meta = gold.load_gold(path)
    assert loaded_cases == make_gold_cases()
    assert loaded_meta == meta


def test_load_gold_rejects_invalid_payload(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"meta": {}, "cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid gold set"):
        gold.load_gold(path)


# --- gold CLI ---


def test_gold_parse_args_defaults():
    args = gold.parse_args([])
    assert args.out == gold.DEFAULT_OUT
    assert args.min_chars == gold.DEFAULT_MIN_CHARS
    assert args.limit is None
    assert args.seed == gold.DEFAULT_SEED


def test_gold_main_with_missing_db_exits_gracefully(tmp_path, capsys):
    out = tmp_path / "gold.json"
    assert gold.main(["--db", str(tmp_path / "missing.db"), "--out", str(out)]) == 0
    assert not out.exists()
    assert "no database" in capsys.readouterr().out


def test_help_exits_cleanly_for_both_clis():
    for module in (gold, eval_real):
        with pytest.raises(SystemExit) as excinfo:
            module.parse_args(["--help"])
        assert excinfo.value.code == 0


# --- hashing embedder ---


async def test_hashing_embedder_is_deterministic():
    embedder = eval_real.HashingEmbedder(dimension=64)
    first = await embedder.embed_query("consulta de prueba")
    second = await embedder.embed_query("consulta de prueba")
    assert first == second


async def test_hashing_embedder_vectors_have_unit_norm():
    vector = await eval_real.HashingEmbedder(dimension=64).embed_query("consulta de prueba")
    assert abs(math.sqrt(sum(value * value for value in vector)) - 1.0) < 1e-9


async def test_hashing_embedder_separates_different_texts():
    embedder = eval_real.HashingEmbedder(dimension=256)
    left = await embedder.embed_query("finanzas y presupuestos anuales")
    right = await embedder.embed_query("arquitectura limpia desacoplada")
    assert left != right


async def test_hashing_embedder_respects_requested_dimension():
    vectors = await eval_real.HashingEmbedder(dimension=32).embed_texts(["hola"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 32


async def test_hashing_embedder_empty_text_yields_zero_vector():
    vector = await eval_real.HashingEmbedder(dimension=32).embed_query("")
    assert vector == [0.0] * 32


def test_hashing_embedder_satisfies_embedding_protocol():
    assert isinstance(eval_real.HashingEmbedder(), EmbeddingProvider)


# --- eval gold loading and report aggregation ---


def test_to_retrieval_case_maps_gold_rows():
    case = eval_real.to_retrieval_case(
        {"query": "q", "relevant_chunk_ids": ["c1", "c2"], "relevant_document_ids": ["d1"], "case_id": "g007"}
    )
    assert case.query == "q"
    assert case.relevant == {"c1", "c2"}
    assert case.relevant_documents == {"d1"}
    assert case.case_id == "g007"


def test_load_gold_cases_reads_valid_gold(tmp_path):
    path = tmp_path / "gold.json"
    gold.write_gold(path, make_gold_cases(), {"total_cases": 1})
    loaded, meta = eval_real.load_gold_cases(path)
    assert meta == {"total_cases": 1}
    assert loaded[0].relevant == {"c1"}


def test_load_gold_cases_rejects_missing_file(tmp_path):
    with pytest.raises(eval_real.GoldSetError, match="build_retrieval_gold"):
        eval_real.load_gold_cases(tmp_path / "nope.json")


def test_load_gold_cases_rejects_invalid_payload(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"meta": {}, "cases": []}), encoding="utf-8")
    with pytest.raises(eval_real.GoldSetError, match="not usable"):
        eval_real.load_gold_cases(path)


def test_build_report_folds_summary_and_metadata():
    report = eval_real.build_report(
        make_eval_output(),
        gold_meta={"total_cases": 2},
        provider_name="hashing",
        database="data/ragpoc.db",
        dimension=3072,
        ks=(1, 3),
        top_k=3,
    )
    assert report["summary"]["recall@1"] == 0.5
    assert report["gold"] == {"total_cases": 2}
    assert report["ks"] == [1, 3]
    assert report["top_k"] == 3
    assert report["embedding_dimension"] == 3072
    assert report["notes"] == [eval_real.HASHING_NOTE]
    assert len(report["per_case"]) == 1


def test_build_report_skips_hashing_note_for_local_provider():
    report = eval_real.build_report(
        make_eval_output(),
        gold_meta={},
        provider_name="local",
        database="db",
        dimension=None,
        ks=(1,),
        top_k=1,
    )
    assert report["notes"] == []


def test_format_summary_reports_percentages_per_depth():
    report = eval_real.build_report(
        make_eval_output(),
        gold_meta={},
        provider_name="hashing",
        database="db",
        dimension=None,
        ks=(1, 3),
        top_k=3,
    )
    text = eval_real.format_summary(report)
    assert "K=1" in text and "K=3" in text
    assert "recall" in text and "50.00%" in text
    assert "mrr" in text and "75.00%" in text
    assert "p95" in text


# --- eval CLI ---


def test_eval_parse_args_defaults():
    args = eval_real.parse_args([])
    assert args.provider == "hashing"
    assert args.ks == (1, 3, 5, 8)
    assert args.top_k is None
    assert args.limit is None


def test_eval_parse_args_accepts_custom_depths():
    args = eval_real.parse_args(["--ks", "2,4", "--top-k", "4"])
    assert args.ks == (2, 4)
    assert args.top_k == 4


def test_eval_parse_args_rejects_bad_depths():
    with pytest.raises(SystemExit):
        eval_real.parse_args(["--ks", "1,x"])
    with pytest.raises(SystemExit):
        eval_real.parse_args(["--ks", "0,1"])


def test_eval_parse_args_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        eval_real.parse_args(["--provider", "gemini"])


def test_eval_main_missing_gold_reports_actionable_error(tmp_path, capsys):
    code = eval_real.main(["--gold", str(tmp_path / "nope.json"), "--out", str(tmp_path / "r.json")])
    assert code == 2
    assert "build_retrieval_gold" in capsys.readouterr().err


def test_eval_main_missing_database_reports_actionable_error(tmp_path, capsys):
    gold_path = tmp_path / "gold.json"
    gold.write_gold(gold_path, make_gold_cases(), {"total_cases": 1})
    code = eval_real.main(
        ["--gold", str(gold_path), "--db", str(tmp_path / "no.db"), "--out", str(tmp_path / "r.json")]
    )
    assert code == 2
    assert "database" in capsys.readouterr().err
