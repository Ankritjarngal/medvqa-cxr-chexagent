from baseline.evaluate import evaluate, exact_match, loose_match, normalize


def test_normalize_yes_no_synonyms():
    assert normalize("Yes.") == "yes"
    assert normalize("Affirmative") == "yes"
    assert normalize("No ") == "no"
    assert normalize("absent") == "no"


def test_exact_match():
    assert exact_match("Yes", "yes.")
    assert not exact_match("left", "right")


def test_loose_match_substring():
    assert loose_match("The effusion is on the right side.", "right")
    assert loose_match("right", "The effusion is on the right side.")
    assert not loose_match("a", "cat")  # too short to count as a real match


def test_evaluate_breakdown_by_category_and_source():
    records = [
        {"question": "q1", "gold_answer": "no", "prediction": "No",
         "category": "severity", "source_dataset": "VQA-RAD"},
        {"question": "q2", "gold_answer": "right", "prediction": "left",
         "category": "laterality", "source_dataset": "SLAKE"},
        {"question": "q3", "gold_answer": "x-ray", "prediction": "This is an x-ray.",
         "category": "general", "source_dataset": "SLAKE"},
    ]
    report = evaluate(records)

    assert report.overall.total == 3
    assert report.overall.exact_correct == 1  # only q1
    assert report.overall.loose_correct == 2  # q1 and q3

    assert report.per_category["laterality"].exact_accuracy == 0.0
    assert report.per_category["severity"].exact_accuracy == 1.0
    assert report.per_source["SLAKE"].total == 2
    assert report.per_source["VQA-RAD"].total == 1


from baseline.evaluate import token_f1, wilson_interval


def test_token_f1_gives_partial_credit():
    assert token_f1("Chest tightness, shortness of breath", "Chest tightness, fatigue") > 0.3
    assert token_f1("left", "right") == 0.0
    assert token_f1("Right", "right") == 1.0


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(9, 20)
    assert lo < 0.45 < hi
    assert 0.24 < lo < 0.28 and 0.64 < hi < 0.68  # ~26%-66%
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # NaN for empty bucket


def _rec(q, gold, pred, cat="general", fmt=None):
    r = {"question": q, "gold_answer": gold, "prediction": pred, "category": cat,
         "source_dataset": "SLAKE"}
    if fmt:
        r["answer_format"] = fmt
    return r


def test_yes_no_sensitivity_specificity_and_majority_baseline():
    # 4 gold-no (all answered no), 2 gold-yes (both missed): model always says no.
    records = [_rec("q", "no", "No") for _ in range(4)] + [_rec("q", "yes", "No") for _ in range(2)]
    report = evaluate(records)
    yn = report.yes_no
    assert (yn.tp, yn.fn, yn.tn, yn.fp) == (0, 2, 4, 0)
    assert yn.sensitivity == 0.0 and yn.specificity == 1.0
    assert abs(yn.accuracy - 4 / 6) < 1e-9
    assert abs(yn.majority_baseline - 4 / 6) < 1e-9  # no better than always saying "no"
    assert report.per_format["yes/no"].total == 6


def test_breakdowns_by_format_and_kind():
    records = [
        {**_rec("how to treat it?", "surgery", "surgery", cat="laterality"), "is_knowledge": True},
        _rec("which side?", "left", "right", cat="laterality"),
    ]
    report = evaluate(records)
    assert report.per_format["open"].total == 2
    assert report.per_kind["knowledge"].exact_accuracy == 1.0
    assert report.per_kind["perception"].exact_accuracy == 0.0
    assert report.per_category_format["laterality / open"].total == 2
