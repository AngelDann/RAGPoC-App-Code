import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
agg = r["generation"]["aggregate"]

print("=== AGREGADO ===")
for key in (
    "n",
    "tokens_per_second_mean",
    "latency_p50_s",
    "latency_max_s",
    "term_recall_mean",
    "quality_overall_mean",
    "unanswerable_cases",
    "abstention_rate",
    "hallucination_rate",
    "peak_rss_mb",
):
    print(f"  {key:26s} {agg.get(key)}")

print()
print("=== CASOS SIN RESPUESTA (deben abstenerse) ===")
for c in r["generation"]["cases"]:
    if c["kind"] != "unanswerable":
        continue
    print(
        f"[{c['case_id']}] abstained={c['abstained']} "
        f"risk={c['fabrication_risk']} overall={c['quality_overall']}"
    )
    print(f"   > {c['answer_preview']}")

print()
print("=== CASOS CON RESPUESTA ESPERADA ===")
for c in r["generation"]["cases"]:
    if c["kind"] == "unanswerable":
        continue
    print(
        f"[{c['case_id']}] recall={c['term_recall']} overall={c['quality_overall']} "
        f"lang={c['detected_language']} tok/s={c['tokens_per_second']}"
    )
    print(f"   > {c['answer_preview']}")
