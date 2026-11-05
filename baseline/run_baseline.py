"""Phase 1 CLI: build the chest X-ray test set, run CheXagent-2-3b zero-shot,
score the predictions, and save a category breakdown.

Usage (GPU runtime, e.g. Colab):
    python -m baseline.run_baseline --n-per-category 20

Usage (no GPU, just to sanity-check the data pipeline):
    python -m baseline.run_baseline --n-per-category 5 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import build_baseline_test_set
from .evaluate import evaluate, format_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-category", type=int, default=20,
                         help="Max questions sampled per category (general/laterality/severity).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true",
                         help="Build and save the test set, but skip model inference "
                              "(use this on a machine without a GPU).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building chest X-ray test set (SLAKE metadata-filtered + VQA-RAD keyword-filtered)...")
    records = build_baseline_test_set(
        images_dir=args.images_dir, n_per_category=args.n_per_category, seed=args.seed
    )
    print(f"{len(records)} questions ready.")

    test_set_path = output_dir / "baseline_test_set.jsonl"
    with open(test_set_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved test set to {test_set_path}")

    if args.dry_run:
        print("\n--dry-run set: skipping model inference. "
              "Re-run without --dry-run on a GPU runtime (e.g. Colab T4) to get predictions.")
        return

    from .model import CheXagentRunner, prompt_for_category

    print("Loading StanfordAIMI/CheXagent-2-3b (first run downloads ~12.6GB of weights)...")
    runner = CheXagentRunner()

    for i, r in enumerate(records, start=1):
        prompt = prompt_for_category(r["question"], r["category"])
        r["prediction"] = runner.ask([r["image_path"]], prompt, max_new_tokens=args.max_new_tokens)
        print(f"[{i}/{len(records)}] ({r['category']}) Q: {r['question']!r}  "
              f"gold={r['gold_answer']!r}  pred={r['prediction']!r}")

    report = evaluate(records)
    print("\n" + format_report(report))

    results_csv_path = output_dir / "baseline_results.csv"
    fieldnames = ["id", "source_dataset", "source_split", "category", "image_path",
                  "question", "gold_answer", "prediction", "exact_match", "loose_match"]
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"\nSaved per-question results to {results_csv_path}")

    report_txt_path = output_dir / "baseline_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(format_report(report))
    print(f"Saved summary report to {report_txt_path}")


if __name__ == "__main__":
    main()
