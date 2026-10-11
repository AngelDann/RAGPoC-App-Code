import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

src = Path(sys.argv[1])
r = json.loads(src.read_text(encoding="utf-8"))

print("AGGREGATE:", json.dumps(r["generation"]["aggregate"], ensure_ascii=False))
print()
for c in r["generation"]["cases"]:
    print(f"[{c['case_id']}] kind={c['kind']} recall={c['term_recall']} chars={c['answer_chars']}")
    print(f"   matched: {c['matched_terms']}")
    print(f"   > {c['answer_preview']}")
    print()
