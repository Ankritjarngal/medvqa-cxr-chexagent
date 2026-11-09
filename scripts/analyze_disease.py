"""Re-run the Stage 1 (disease naming) reports from disease_results.csv, no model needed.

  python3 scripts/analyze_disease.py results/run_5/disease_results.csv
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.disease_labels import (  # noqa: E402
    choose_method, class_table, disease_reading, disease_report, format_disease_report, parse_diseases,
)

METHODS = ("open", "forced", "perclass")
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
by_method = defaultdict(list)
scores = {}
for r in rows:
    labels = [x.strip() for x in r["gold"].split(",")]
    if r["method"] == "perclass":
        s = json.loads(r["raw"])
        scores[r["id"]] = (s, labels)
        pred = sorted(s, key=lambda c: -s[c])[:1]
    else:
        pred = parse_diseases(r["raw"])
    by_method[r["method"]].append({"id": r["id"], "labels": labels, "pred": pred, "split": r["split"]})

reports = {m: disease_report(by_method[m]) for m in METHODS if m in by_method}
print(format_disease_report(reports))
if scores:
    print("\nperclass one-vs-rest AUC:")
    table = class_table({i: s for i, (s, _) in scores.items()}, {i: l for i, (_, l) in scores.items()})
    gold_counts = defaultdict(int)
    for _, (_, l) in scores.items():
        for x in l:
            gold_counts[x] += 1
    for name, auc in table.items():
        if gold_counts.get(name):
            print(f"  {name:>14} n={gold_counts[name]:>2} AUC={auc:.2f}")
val = {m: disease_report([r for r in by_method[m] if r["split"] == "validation"], n_shuffles=200) for m in reports}
chosen = choose_method(val)
print("\nvalidation accuracy:", {m: f"{v['acc_top']:.0%}" for m, v in val.items()})
print("CHOSEN_METHOD =", repr(chosen))
for m, rep in reports.items():
    print(f"[{m}] {disease_reading(rep)}")
