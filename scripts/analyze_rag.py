"""Analyse the Phase 2 generation arms against the pre-registered rules.

  python3 scripts/analyze_rag.py results/run_5/rag_results.csv [--make-sheets DIR] [--ratings DIR]
                                 [--run2 results/run_2/baseline_results.csv]

--make-sheets writes the blind rating sheets (rate columns rating_1 / rating_2 with 0, 1 or 2; rubric in baseline/knowledge.py).
--ratings reads the completed sheets and unblinds them for the decision rule.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.compare import load_rows  # noqa: E402
from baseline.knowledge import RATING_RUBRIC  # noqa: E402
from baseline.rag_analysis import (  # noqa: E402
    arm_table, comparison_report, decision_rule, format_decision, guess_accuracy, load_arm_rows,
    make_rating_sheets, summarize_rating_files,
)

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--make-sheets", default=None)
ap.add_argument("--ratings", default=None)
ap.add_argument("--run2", default="results/run_2/baseline_results.csv")
args = ap.parse_args()

rows = load_arm_rows(args.results)
print(arm_table(rows))
g = guess_accuracy(rows)
print(f"\nThe model's disease guess equals the true disease on {g['correct']} / {g['n']} oracle-subset questions.\n")
print(comparison_report(rows))

if args.make_sheets:
    info = make_rating_sheets(rows, args.make_sheets)
    print("\n=== Blind rating sheets written to", args.make_sheets, "===")
    for k, v in info.items():
        print(f"  {v['pair']}: {v['to_rate']} pairs to rate ({v['identical']} identical pairs left off, tied by definition)")
    print("Rubric:\n" + RATING_RUBRIC)

ratings = {}
if args.ratings:
    for a, b in (("A1", "A6"), ("A5", "A6"), ("A3", "A4")):
        ratings[f"{a}_vs_{b}"] = summarize_rating_files(args.ratings, a, b)
        r = ratings[f"{a}_vs_{b}"]
        if r:
            print(f"\nRatings {a} vs {b}: rated {r['n_rated']}, mean {a}/{b} = {r['mean_baseline']:.2f}/{r[f'mean_{b}']:.2f}, "
                  f"{b} wins {r['wins']}, losses {r['losses']}, ties {r['ties']} (+{r['identical_pairs_tied']} identical), "
                  f"sign test p = {r['sign_test_p']:.3f}")

run2 = None
if Path(args.run2).exists():
    run2 = load_rows(args.run2)
print()
print(format_decision(decision_rule(rows, ratings, run2)))
