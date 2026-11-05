import math

from baseline.stats import binom_two_sided, cluster_bootstrap_mean_diff, fisher_exact_2x2, mcnemar_exact


def test_binomial_known_values():
    assert math.isclose(binom_two_sided(7, 10, 0.5), 0.34375)
    assert math.isclose(binom_two_sided(0, 8, 0.5), 2 / 256)
    assert binom_two_sided(5, 10, 0.5) == 1.0
    assert binom_two_sided(0, 0) == 1.0


def test_mcnemar_exact_known_values():
    assert math.isclose(mcnemar_exact(0, 8), 2 / 256)
    assert mcnemar_exact(4, 4) == 1.0
    assert math.isclose(mcnemar_exact(10, 2), binom_two_sided(10, 12, 0.5))


def test_fisher_exact_known_value():
    assert math.isclose(fisher_exact_2x2(3, 1, 1, 3), 34 / 70)
    assert fisher_exact_2x2(5, 5, 5, 5) == 1.0
    assert fisher_exact_2x2(10, 0, 0, 10) < 0.001


def test_cluster_bootstrap_is_deterministic_and_brackets_mean():
    diffs = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    clusters = ["a", "a", "b", "c", "d", "e", "f", "g"]
    a = cluster_bootstrap_mean_diff(diffs, clusters, n_boot=2000, seed=3)
    b = cluster_bootstrap_mean_diff(diffs, clusters, n_boot=2000, seed=3)
    assert a == b
    mean, lo, hi = a
    assert math.isclose(mean, 0.625)
    assert lo <= mean <= hi


def test_cluster_bootstrap_no_difference_ci_contains_zero():
    diffs = [1.0, -1.0] * 20
    clusters = [str(i) for i in range(40)]
    _, lo, hi = cluster_bootstrap_mean_diff(diffs, clusters, n_boot=2000, seed=0)
    assert lo < 0 < hi
