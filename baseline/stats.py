"""Small, dependency-free statistical tests used to compare buckets and runs.

Everything here is implemented with the standard library so it runs in the CI
job and in Colab without scipy. Each function is checked against known values
in `tests/test_stats.py`.
"""

from __future__ import annotations

import math
import random


def binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return float(k == 0)
    if p >= 1.0:
        return float(k == n)
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test: probability, under Binomial(n, p), of an
    outcome no more likely than the observed `k`."""
    if n == 0:
        return 1.0
    observed = binom_pmf(k, n, p)
    total = sum(binom_pmf(i, n, p) for i in range(n + 1) if binom_pmf(i, n, p) <= observed * (1 + 1e-9))
    return min(1.0, total)


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Exact McNemar test on discordant pairs: under 'no difference between the
    two systems', each discordant pair is equally likely to favour either."""
    return binom_two_sided(only_a_correct, only_a_correct + only_b_correct, 0.5)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    lo, hi = max(0, k - n2), min(n1, k)
    observed = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= observed * (1 + 1e-9)))


def cluster_bootstrap_mean_diff(diffs: list[float], clusters: list[str], n_boot: int = 10000,
                                seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of paired differences with a percentile bootstrap CI that resamples
    whole clusters (e.g. all questions about one image), because questions about
    the same image or from the same template are not independent. Returns
    (mean, ci_low, ci_high)."""
    if not diffs:
        return (float("nan"), float("nan"), float("nan"))
    groups: dict[str, list[float]] = {}
    for d, c in zip(diffs, clusters):
        groups.setdefault(c, []).append(d)
    keys = list(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total, count = 0.0, 0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            total += sum(g)
            count += len(g)
        means.append(total / count)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (sum(diffs) / len(diffs), lo, hi)
