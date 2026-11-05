"""Paired comparison of two runs on the same questions (baseline vs retrieval, prompt A vs B).

Rows are joined on the stable question `id`. For a yes/no-style outcome (answer
correct or not) it reports how many questions each run alone got right and an
exact McNemar test. For a continuous score (token F1, item F1) it reports the
mean paired difference with a bootstrap CI that resamples whole images, because
several questions are asked about the same image and are not independent.

Usage:
    python -m baseline.compare results/run_2/baseline_results.csv results/phase2/rag.csv \
        --metric exact_match --by category
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass

from .stats import cluster_bootstrap_mean_diff, mcnemar_exact

BINARY_METRICS = {"exact_match", "loose_match", "side_correct"}


def load_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "1.0")


def pair_by_id(rows_a: list[dict], rows_b: list[dict]) -> list[tuple[dict, dict]]:
    by_id = {r["id"]: r for r in rows_b}
    return [(a, by_id[a["id"]]) for a in rows_a if a["id"] in by_id]


@dataclass
class BinaryComparison:
    n: int
    a_correct: int
    b_correct: int
    only_a: int
    only_b: int
    p_value: float


@dataclass
class ContinuousComparison:
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float


def compare_binary(pairs: list[tuple[dict, dict]], metric: str) -> BinaryComparison:
    a_ok = [_truthy(a[metric]) for a, _ in pairs]
    b_ok = [_truthy(b[metric]) for _, b in pairs]
    only_a = sum(x and not y for x, y in zip(a_ok, b_ok))
    only_b = sum(y and not x for x, y in zip(a_ok, b_ok))
    return BinaryComparison(len(pairs), sum(a_ok), sum(b_ok), only_a, only_b, mcnemar_exact(only_a, only_b))


def compare_continuous(pairs: list[tuple[dict, dict]], metric: str, n_boot: int = 10000,
                       seed: int = 0) -> ContinuousComparison:
    a = [float(x[metric]) for x, _ in pairs]
    b = [float(y[metric]) for _, y in pairs]
    clusters = [str(x.get("image_path") or x["id"]) for x, _ in pairs]
    mean, lo, hi = cluster_bootstrap_mean_diff([y - x for x, y in zip(a, b)], clusters, n_boot, seed)
    n = len(pairs)
    return ContinuousComparison(n, sum(a) / n if n else float("nan"), sum(b) / n if n else float("nan"), mean, lo, hi)


def compare(rows_a: list[dict], rows_b: list[dict], metric: str, by: str | None = None) -> dict[str, object]:
    """Compare run B against run A overall and, optionally, per value of the
    row field `by` (e.g. category, source_split, knowledge_subtype)."""
    pairs = pair_by_id(rows_a, rows_b)
    fn = compare_binary if metric in BINARY_METRICS else compare_continuous
    result: dict[str, object] = {"all": fn(pairs, metric)}
    if by:
        groups: dict[str, list] = {}
        for a, b in pairs:
            groups.setdefault(str(a.get(by)), []).append((a, b))
        for key, ps in sorted(groups.items()):
            result[f"{by}={key}"] = fn(ps, metric)
    return result


def format_comparison(result: dict[str, object], metric: str, name_a: str = "A", name_b: str = "B") -> str:
    L = [f"Paired comparison on '{metric}' ({name_a} -> {name_b})"]
    for key, c in result.items():
        if isinstance(c, BinaryComparison):
            L.append(f"{key:>28}: n={c.n:<4} {name_a}={c.a_correct}  {name_b}={c.b_correct}  "
                     f"only-{name_a}-right={c.only_a}  only-{name_b}-right={c.only_b}  McNemar p={c.p_value:.3f}")
        else:
            L.append(f"{key:>28}: n={c.n:<4} {name_a}={c.mean_a:.3f}  {name_b}={c.mean_b:.3f}  "
                     f"diff={c.diff:+.3f} (95% CI {c.ci_low:+.3f} to {c.ci_high:+.3f}, resampled by image)")
    L.append("A difference is only credible if p < 0.05 (binary) or the CI excludes 0 (continuous), "
             "and the same direction holds in the train and test/validation rows separately.")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--metric", default="exact_match")
    parser.add_argument("--by", default=None)
    parser.add_argument("--exclude-ids", default=None,
                        help="text file of question ids to drop from both runs (lines starting with # ignored)")
    args = parser.parse_args()
    rows_a, rows_b = load_rows(args.run_a), load_rows(args.run_b)
    if args.exclude_ids:
        with open(args.exclude_ids) as f:
            drop = {line.strip() for line in f if line.strip() and not line.startswith("#")}
        rows_a = [r for r in rows_a if r["id"] not in drop]
        rows_b = [r for r in rows_b if r["id"] not in drop]
    print(format_comparison(compare(rows_a, rows_b, args.metric, args.by),
                            args.metric, "A", "B"))


if __name__ == "__main__":
    main()
