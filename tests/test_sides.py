from baseline.sides import format_side_report, score_side_rows, side_of, side_report


def test_side_of_extracts_sides():
    assert side_of("Right") == "right"
    assert side_of("Lower Left Lung") == "left"
    assert side_of("upper right lobe.") == "right"
    assert side_of("Both Lung") == "both"
    assert side_of("bilateral lungs") == "both"
    assert side_of("left and right lungs") == "both"
    assert side_of("No") is None
    assert side_of("Lung") is None


def test_side_of_does_not_match_inside_words():
    assert side_of("copyright") is None
    assert side_of("leftover") is None


def _row(gold, pred):
    return {"gold_answer": gold, "prediction": pred}


def test_score_side_rows_marks_region_answers_correct_on_side():
    rows = score_side_rows([_row("Lower Left Lung", "Left lung"), _row("Right", "Left"), _row("Left", "No")])
    assert [r["side_correct"] for r in rows] == [True, False, False]
    assert [r["side_given"] for r in rows] == [True, True, False]


def test_side_report_detects_right_default():
    rows = score_side_rows(
        [_row("Left", "Right")] * 8 + [_row("Left", "Left")] * 2
        + [_row("Right", "Right")] * 10 + [_row("Both", "Right")] * 2
    )
    rep = side_report(rows)
    assert rep["n"] == 22
    assert rep["confusion"]["left"]["right"] == 8
    assert rep["per_gold"]["right"]["accuracy"] == 1.0
    assert rep["left_to_right_rate"] == 0.8
    assert rep["right_to_left_rate"] == 0.0
    assert rep["asymmetry_p"] < 0.001
    assert rep["predicted_right_rate"] > rep["gold_right_rate"]
    assert "right" in format_side_report(rep)


def test_side_report_symmetric_errors_not_flagged():
    rows = score_side_rows([_row("Left", "Right")] * 3 + [_row("Left", "Left")] * 7
                           + [_row("Right", "Left")] * 3 + [_row("Right", "Right")] * 7)
    assert side_report(rows)["asymmetry_p"] == 1.0


def test_side_report_ignores_rows_without_gold_side():
    rows = score_side_rows([_row("Pneumonia", "Right"), _row("Left", "Left")])
    assert side_report(rows)["n"] == 1
