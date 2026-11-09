import pytest

from baseline.disease_labels import (
    attach_oracle_disease, auc_one_vs_rest, build_image_labels, canonical_labels, choose_method, class_table,
    disease_id_items, disease_report, format_disease_report, is_disease_naming_question, parse_diseases,
    perclass_predictions,
)


def rec(uid, key, q, a, split="train", ds="SLAKE", knowledge=False):
    return {"source_dataset": ds, "source_split": split, "uid": uid, "image_key": key, "question": q,
            "gold_answer": a, "is_knowledge": knowledge}


def test_disease_question_detection():
    assert is_disease_naming_question("What diseases are included in the picture?")
    assert is_disease_naming_question("What disease is/are shown on the left of lung?")
    assert not is_disease_naming_question("How to treat the disease located on the left of lung?")
    assert not is_disease_naming_question("Which disease can be seen?")


def test_spelling_variants_and_multilabel():
    assert canonical_labels("Pulmonary Mass") == ("Mass",)
    assert canonical_labels("Noudle") == ("Nodule",)
    assert canonical_labels("Cardiomega") == ("Cardiomegaly",)
    assert canonical_labels("Cardiomegaly, Infiltration, Effusion") == ("Cardiomegaly", "Effusion", "Infiltration")
    assert canonical_labels("Pleural Effusion") == ("Effusion",)
    assert canonical_labels("Something New") == ("Something New",)


def test_image_labels_merge_and_ignore_other_questions_and_datasets():
    pool = [
        rec("SLAKE_1", "xmlab1/source.jpg", "What diseases are included in the picture?", "Pneumonia"),
        rec("SLAKE_2", "xmlab1/source.jpg", "What disease is/are shown on the left of lung?", "Pneumonia"),
        rec("SLAKE_3", "xmlab2/source.jpg", "What diseases are included in the picture?", "Atelectasis, Mass", "test"),
        rec("SLAKE_4", "xmlab3/source.jpg", "Where is the lesion?", "Left"),
        rec("VQA-RAD_1", "x.jpg", "What disease is this?", "Pneumonia", ds="VQA-RAD"),
    ]
    labels = build_image_labels(pool)
    assert labels == {"xmlab1/source.jpg": {"labels": ("Pneumonia",), "split": "train"},
                      "xmlab2/source.jpg": {"labels": ("Atelectasis", "Mass"), "split": "test"}}
    items = disease_id_items(pool)
    assert [i["id"] for i in items] == ["DIS_xmlab1", "DIS_xmlab2"]
    assert items[1]["labels"] == ["Atelectasis", "Mass"]


def test_oracle_disease_only_for_single_label_slake_images():
    pool = [
        rec("SLAKE_1", "a/s.jpg", "What diseases are included in the picture?", "Pneumonia"),
        rec("SLAKE_2", "b/s.jpg", "What diseases are included in the picture?", "Mass, Atelectasis"),
    ]
    know = [rec("SLAKE_10", "a/s.jpg", "How to treat the disease?", "x", knowledge=True),
            rec("SLAKE_11", "b/s.jpg", "How to treat the disease?", "x", knowledge=True),
            rec("SLAKE_12", "c/s.jpg", "How to treat the disease?", "x", knowledge=True),
            rec("VQA-RAD_1", "a/s.jpg", "How to treat?", "x", ds="VQA-RAD", knowledge=True)]
    out = attach_oracle_disease(know, pool)
    assert [r["oracle_disease"] for r in out] == ["Pneumonia", None, None, None]


def test_parse_diseases_order_and_synonyms():
    assert parse_diseases("The image shows cardiomegaly and a small pleural effusion.") == ["Cardiomegaly", "Effusion"]
    assert parse_diseases("Pneumonia") == ["Pneumonia"]
    assert parse_diseases("A lung mass is present") == ["Mass"]
    assert parse_diseases("No acute findings") == ["No Finding"]
    assert parse_diseases("Yes") == []
    assert parse_diseases("Cardiomegaly, Pneumothorax") == ["Cardiomegaly", "Pneumothorax"]


def test_auc_ties_and_perfect_ranking():
    assert auc_one_vs_rest([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert auc_one_vs_rest([1, 1, 1, 1], [True, True, False, False]) == 0.5
    assert auc_one_vs_rest([0, 1, 2, 3], [True, True, False, False]) == 0.0


def _rows(preds, golds, splits=None):
    splits = splits or ["train"] * len(preds)
    return [{"pred": p, "labels": g, "split": s} for p, g, s in zip(preds, golds, splits)]


def test_report_perfect_majority_and_shuffled_control():
    golds = [["Mass"]] * 6 + [["Nodule"]] * 4
    perfect = disease_report(_rows([g[:] for g in golds], golds))
    assert perfect["acc_top"] == 1.0 and perfect["acc_strict"] == 1.0
    assert perfect["majority_label"] == "Mass" and perfect["majority_acc"] == 0.6
    assert perfect["shuffled_mean"] < 0.6          # shuffling destroys a perfect alignment
    always = disease_report(_rows([["Mass"]] * 10, golds))
    assert always["acc_top"] == 0.6 == always["majority_acc"]
    assert always["p_vs_majority"] == pytest.approx(1.0)
    assert always["per_class"]["Nodule"]["recall"] == 0.0


def test_report_multilabel_any_and_strict_and_unparsed():
    golds = [["Effusion", "Mass"], ["Mass"], ["Mass"], ["Nodule"]]
    rep = disease_report(_rows([["Mass"], ["Mass", "Effusion"], [], ["Nodule"]], golds))
    assert rep["acc_top"] == pytest.approx(3 / 4)      # top choice matches any gold label: rows 0, 1, 3
    assert rep["acc_any"] == pytest.approx(3 / 4)
    assert rep["acc_strict"] == pytest.approx(1 / 4)   # strict = exactly one name, equal to the whole gold set: row 3
    assert rep["unparsed"] == 1


def test_choose_method_is_validation_best_with_stable_ties():
    assert choose_method({"open": {"acc_top": 0.2}, "forced": {"acc_top": 0.4}, "perclass": {"acc_top": 0.3}}) == "forced"
    assert choose_method({"open": {"acc_top": 0.3}, "forced": {"acc_top": 0.3}}) == "open"


def test_perclass_predictions_and_class_table():
    scores = {"a": {"Mass": 2.0, "Nodule": -1.0}, "b": {"Mass": -2.0, "Nodule": 1.5}}
    assert perclass_predictions(scores) == {"a": ["Mass", "Nodule"], "b": ["Nodule", "Mass"]}
    t = class_table(scores, {"a": ["Mass"], "b": ["Nodule"]})
    assert t == {"Mass": 1.0, "Nodule": 1.0}


def test_format_report_runs():
    rep = disease_report(_rows([["Mass"]] * 4, [["Mass"], ["Mass"], ["Nodule"], ["Nodule"]],
                               ["train", "train", "test", "validation"]))
    assert "Disease naming" in format_disease_report({"open": rep})
