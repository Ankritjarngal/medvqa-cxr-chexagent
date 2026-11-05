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
