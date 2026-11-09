import math

import pytest

from baseline.diagnostics import (flip_image, flip_report, heart_report, is_explicit_side_question, parse_heart_answer,
                                  parse_normal_or_enlarged, parse_yes_no, select_heart_size_records,
                                  select_side_diagnostic_records, variant_report)


def test_explicit_side_question_detection():
    assert is_explicit_side_question("Which side of lung is abnormal in this image,left or right?")
    assert is_explicit_side_question("which side of the image is the thoracic mass located on?")
    assert not is_explicit_side_question("Where is the abnormality in this image?")


def test_side_diagnostic_selection_keeps_all_explicit_and_drops_both():
    recs = [{"question": f"Which side {i}?", "gold_side": "left" if i % 2 else "right"} for i in range(6)]
    recs += [{"question": f"Where {i}?", "gold_side": "left"} for i in range(10)]
    recs += [{"question": "Which side?", "gold_side": "both"}]
    out = select_side_diagnostic_records(recs, n_other=4, seed=1)
    assert sum(is_explicit_side_question(r["question"]) for r in out) == 6
    assert len(out) == 10 and all(r["gold_side"] in ("left", "right") for r in out)


def test_flip_image_mirrors_pixels(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (4, 2), "black")
    im.putpixel((0, 0), (255, 0, 0))
    src, dst = tmp_path / "a.png", tmp_path / "b.png"
    im.save(src)
    flip_image(str(src), str(dst))
    assert Image.open(dst).getpixel((3, 0)) == (255, 0, 0) and Image.open(dst).getpixel((0, 0)) == (0, 0, 0)


def _flip_rows(pairs):
    return [{"gold_side": g, "pred_original": a, "pred_flipped": b, "explicit": True} for g, a, b in pairs]


def test_flip_report_detects_constant_answer():
    rep = flip_report(_flip_rows([("left", "Right", "Right"), ("right", "Right", "Right")] * 10))["all"]
    assert rep["swap_rate"] == 0.0 and rep["same"] == 20 and rep["identical_text"] == 20


def test_flip_report_detects_image_following_model():
    rows = _flip_rows([("left", "Left", "Right"), ("right", "Right", "Left")] * 10)
    rep = flip_report(rows)["all"]
    assert rep["swap_rate"] == 1.0 and rep["acc_original"] == 1.0 and rep["acc_flipped_vs_flipped_gold"] == 1.0


def test_flip_report_ignores_unnamed_sides_in_swap_rate():
    rep = flip_report(_flip_rows([("left", "Upper lung", "Right"), ("left", "Left", "Right")]))["all"]
    assert rep["named_both"] == 1 and rep["swapped"] == 1


def test_variant_report_depends_on_truth():
    constant = [{"gold_side": g, "prediction": "Right"} for g in ["left"] * 15 + ["right"] * 15]
    follows = [{"gold_side": g, "prediction": g.title()} for g in ["left"] * 15 + ["right"] * 15]
    c, f = variant_report(constant), variant_report(follows)
    assert c["answered_right"] == 1.0 and c["depends_on_truth_p"] == 1.0 and c["acc_gold_left"] == 0.0
    assert f["depends_on_truth_p"] < 0.001 and f["acc_overall"] == 1.0


def test_heart_selection_polarity_and_exclusions():
    def rec(q, a="yes", uid=None):
        return {"question": q, "answer_format": "yes/no", "gold_answer": a, "source_dataset": "VQA-RAD",
                "uid": uid or q}
    pool = [rec("Is the heart enlarged?"), rec("Is there cardiomegaly?", "no"),
            rec("Can cardiomegaly be observed in this picture?"),
            rec("Is the heart a normal size?"), rec("Does the heart appear normal size?", "no"),
            rec("Can cardiomegaly be observed on the lower right lung in this picture?", "no"),
            rec("Is there bilateral hilar enlargement?"), rec("Are the pulmonary arteries enlarged?"),
            {**rec("Is the heart enlarged?"), "answer_format": "open"}]
    out = select_heart_size_records(pool)
    assert [r["question"] for r in out] == ["Is the heart enlarged?", "Is there cardiomegaly?",
                                             "Can cardiomegaly be observed in this picture?"]
    assert [r["gold_yes"] for r in out] == [True, False, True]


@pytest.mark.parametrize("text,expected", [
    ("Yes", "yes"), ("No.", "no"), ("The heart looks big. Answer: yes", "yes"),
    ("No cardiomegaly is seen. Answer: no", "no"),
    ("The heart is not clearly enlarged, so no", "no"), ("The cardiac silhouette is wide.", None),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("enlarged", "yes"), ("Normal", "no"), ("The heart is mildly enlarged.", "yes"),
    ("normal, not enlarged", "no"), ("No cardiomegaly", "no"), ("Cardiomegaly", "yes"), ("unclear", None),
])
def test_parse_normal_or_enlarged(text, expected):
    assert parse_normal_or_enlarged(text) == expected


def test_heart_report_counts_unparsed_as_wrong():
    rows = [{"gold_yes": True, "prediction": "No"}, {"gold_yes": True, "prediction": "Yes"},
            {"gold_yes": False, "prediction": "No"}, {"gold_yes": False, "prediction": "dunno"}]
    h = heart_report(rows, "plain")
    assert (h["tp"], h["fn"], h["tn"], h["fp"], h["unparsed"]) == (1, 1, 1, 1, 1)
    assert math.isclose(h["sensitivity"], 0.5) and math.isclose(h["always_no_accuracy"], 0.5)
    assert parse_heart_answer("normal_or_enlarged", "Enlarged") == "yes"


def test_youden_is_zero_for_say_yes_to_everything_and_for_say_no_to_everything():
    gold = [True] * 10 + [False] * 10
    assert heart_report([{"gold_yes": g, "prediction": "Yes"} for g in gold], "plain")["youden_j"] == 0.0
    assert heart_report([{"gold_yes": g, "prediction": "No"} for g in gold], "plain")["youden_j"] == 0.0
    perfect = [{"gold_yes": g, "prediction": "Yes" if g else "No"} for g in gold]
    assert heart_report(perfect, "plain")["youden_j"] == 1.0
