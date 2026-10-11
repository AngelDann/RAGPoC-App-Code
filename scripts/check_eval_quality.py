import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpoc.eval_quality import score_answer

PRICE = "$18.50"
HEDGED = (
    "No puedo determinar el precio de las acciones de ACME; "
    "ese dato no esta en la base de conocimiento."
)

rows = []
for label, answer in [("fabrication", PRICE), ("hedged", HEDGED)]:
    s = score_answer(answer, case_kind="unanswerable", expected_language="es")
    rows.append(
        {
            "label": label,
            "answer": answer,
            "abstained": s.abstained,
            "abstention_score": s.abstention_score,
            "fabrication_risk": s.fabrication_risk,
            "overall_field": getattr(s, "overall", None),
            "has_overall_for": hasattr(s, "overall_for"),
            "overall_for_none": s.overall_for(None) if hasattr(s, "overall_for") else None,
            "keyword_recall": s.keyword_recall,
            "matched_terms": s.matched_terms,
            "language_score": s.language_score,
            "detected_language": s.detected_language,
            "shape_score": s.shape_score,
            "empty": s.empty,
            "too_long": s.too_long,
        }
    )

for row in rows:
    print(row)
