import math

from baseline.compare import compare, compare_binary, compare_continuous, format_comparison, pair_by_id


def _rows(flags, key="exact_match"):
    return [{"id": str(i), key: f, "image_path": f"img{i}", "category": "c1" if i % 2 else "c2"}
            for i, f in enumerate(flags)]


def test_pairs_join_on_id_and_ignore_unmatched():
    a, b = _rows([True, False, True]), _rows([True, True])
    assert len(pair_by_id(a, b)) == 2


def test_binary_comparison_counts_and_mcnemar():
    a = _rows(["False"] * 10 + ["True"] * 5)
    b = _rows(["True"] * 10 + ["True"] * 5)
    c = compare_binary(pair_by_id(a, b), "exact_match")
    assert (c.a_correct, c.b_correct, c.only_a, c.only_b) == (5, 15, 0, 10)
    assert math.isclose(c.p_value, 2 / 1024)


def test_binary_comparison_handles_csv_strings():
    a, b = _rows(["True", "False"]), _rows(["False", "True"])
    c = compare_binary(pair_by_id(a, b), "exact_match")
    assert c.only_a == 1 and c.only_b == 1 and c.p_value == 1.0


def test_continuous_comparison_reports_difference_and_ci():
    a = _rows([0.2] * 20, key="token_f1")
    b = _rows([0.5] * 20, key="token_f1")
    c = compare_continuous(pair_by_id(a, b), "token_f1", n_boot=500)
    assert math.isclose(c.diff, 0.3) and c.ci_low > 0.29 and c.ci_high < 0.31


def test_compare_by_group_and_format():
    a, b = _rows([False, False, True, True]), _rows([True, True, True, True])
    res = compare(a, b, "exact_match", by="category")
    assert set(res) == {"all", "category=c1", "category=c2"}
    assert "McNemar" in format_comparison(res, "exact_match", "base", "rag")
