"""Re-run the Phase 1c reports from diagnostics_results.csv (no model needed).

Usage: python3 scripts/analyze_diagnostics.py results/run_4/diagnostics_results.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.diagnostics import (  # noqa: E402
    SIDE_PROMPTS, HEART_PROMPTS, flip_report, format_flip_report, variant_report, format_variant_table,
    heart_report, format_heart_table,
)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(path):
    rows = load(path)
    by = {}
    for r in rows:
        by.setdefault((r["test"], r["variant"]), {})[r["id"]] = r
    orig, flipped = by[("A_flip", "original")], by[("A_flip", "flipped")]
    flip_rows = [{"gold_side": o["gold_side"], "explicit": o["explicit"] == "True",
                  "pred_original": o["prediction"], "pred_flipped": flipped[i]["prediction"]}
                 for i, o in orig.items()]
    print(format_flip_report(flip_report(flip_rows)))

    def side_rows(v, only_explicit=False):
        return [{"gold_side": r["gold_side"], "prediction": r["prediction"]}
                for r in by[("B_side_prompt", v)].values() if not only_explicit or r["explicit"] == "True"]

    print()
    print(format_variant_table({v: variant_report(side_rows(v)) for v in SIDE_PROMPTS}))
    print("\nExplicit questions only:")
    print(format_variant_table({v: variant_report(side_rows(v, True)) for v in SIDE_PROMPTS}))

    def heart_rows(v):
        return [{"gold_yes": r["gold"].strip().lower() == "yes", "prediction": r["prediction"]}
                for r in by[("C_heart", v)].values()]

    print()
    print(format_heart_table({v: heart_report(heart_rows(v), v) for v in HEART_PROMPTS}))


if __name__ == "__main__":
    main(sys.argv[1])
