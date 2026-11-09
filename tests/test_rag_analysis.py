import csv

import pytest

from baseline.rag_analysis import (
    arm, arm_table, comparison_report, decision_rule, format_decision, guess_accuracy, load_arm_rows,
    make_rating_sheets, minimum_detectable_difference, paired, summarize_rating_files,
)

FIELDS = ["arm", "id", "source_dataset", "source_split", "is_dev", "knowledge_subtype", "question", "gold_answer",
          "oracle_disease", "guessed_disease", "disease_used", "passages", "prediction", "image_key",
          "item_recall", "item_precision", "item_f1", "token_f1"]


def make_csv(path, a1=0.2, a6=0.8, n_images=12):
    """12 held-out non-function questions (6 train rows, 6 validation/test rows) plus 2 function questions."""
    rows = []
    for i in range(n_images):
        qid = f"SLAKE_{i}"
        split = "train" if i % 2 == 0 else "validation"
        for arm_name, rec, pred in (("A1", a1, "Keep healthy"), ("A5", a1, "Keep healthy"),
                                    ("A6", a6, f"Quit smoking {i}"), ("A3", a1, "Keep healthy"),
                                    ("A4", a6, f"Quit smoking {i}")):
            rows.append({"arm": arm_name, "id": qid, "source_dataset": "SLAKE", "source_split": split, "is_dev": False,
                         "knowledge_subtype": "prevention", "question": "How to prevent?",
                         "gold_answer": "Quit smoking", "oracle_disease": "Mass", "guessed_disease": "Mass",
                         "disease_used": "Mass", "passages": "", "prediction": pred, "image_key": f"x{i}/source.jpg",
                         "item_recall": rec + (0.05 if i % 3 == 0 else 0.0), "item_precision": 0, "item_f1": 0, "token_f1": 0})
    for arm_name in ("A1", "A5", "A6"):
        for j in range(2):
            rows.append({"arm": arm_name, "id": f"SLAKE_f{j}", "source_dataset": "SLAKE", "source_split": "train",
                         "is_dev": False, "knowledge_subtype": "function", "question": "Function?", "gold_answer": "Breathe",
                         "oracle_disease": "", "guessed_disease": "", "disease_used": "", "passages": "",
                         "prediction": "Breathe", "image_key": f"f{j}/source.jpg", "item_recall": 1.0,
                         "item_precision": 1, "item_f1": 1, "token_f1": 1})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_loading_selections_and_paired_difference(tmp_path):
    rows = load_arm_rows(make_csv(tmp_path / "r.csv"))
    assert len(arm(rows, "A1", "nonfunction")) == 12 and len(arm(rows, "A1", "function")) == 2
    assert len(arm(rows, "A3", "oracle")) == 12 and arm(rows, "A2", "oracle") == []
    c = paired(rows, "A1", "A6", "heldout")
    assert c.n == 12 and c.diff == pytest.approx(0.6) and c.ci_low > 0
    assert "A1" in arm_table(rows) and "A1 -> A6" in comparison_report(rows)
    assert guess_accuracy(rows) == {"n": 12, "correct": 12}
    assert minimum_detectable_difference(rows, "A1", "A6", "heldout") >= 0


def test_blind_sheets_skip_identical_and_unblind(tmp_path):
    rows = load_arm_rows(make_csv(tmp_path / "r.csv"))
    info = make_rating_sheets(rows, str(tmp_path / "sheets"))
    assert info["A1_vs_A6"]["to_rate"] == 12 and info["A5_vs_A6"]["to_rate"] == 12
    assert info["A3_vs_A4"]["to_rate"] == 12
    assert summarize_rating_files(str(tmp_path / "sheets"), "A1", "A6") is None       # nothing rated yet
    sheet_path = tmp_path / "sheets" / "ratings_A1_vs_A6_sheet.csv"
    key = {k["id"]: k["answer_1_is"] for k in csv.DictReader(open(tmp_path / "sheets" / "ratings_A1_vs_A6_key.csv"))}
    sheet = list(csv.DictReader(open(sheet_path)))
    for r in sheet:      # an honest rater who always prefers the retrieval answer
        first_is_candidate = key[r["id"]] == "A6"
        r["rating_1"], r["rating_2"] = ("2", "0") if first_is_candidate else ("0", "2")
    with open(sheet_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    s = summarize_rating_files(str(tmp_path / "sheets"), "A1", "A6")
    assert s["wins"] == 12 and s["losses"] == 0 and s["sign_test_p"] < 0.001


def test_identical_pairs_are_left_off_the_sheet(tmp_path):
    rows = load_arm_rows(make_csv(tmp_path / "r.csv"))
    info = make_rating_sheets(rows, str(tmp_path / "s"))
    assert info["A1_vs_A5"]["identical"] if "A1_vs_A5" in info else True
    # A5 and A6 differ everywhere here; make A5 identical to A6 for two ids and check
    for r in rows:
        if r["arm"] == "A5" and r["id"] in ("SLAKE_0", "SLAKE_1"):
            r["prediction"] = f"Quit smoking {r['id'].split('_')[1]}"
    info = make_rating_sheets(rows, str(tmp_path / "s2"))
    assert info["A5_vs_A6"]["identical"] == 2 and info["A5_vs_A6"]["to_rate"] == 10


def test_all_identical_pairs_count_as_ties_not_pending(tmp_path):
    rows = load_arm_rows(make_csv(tmp_path / "r.csv", a1=0.5, a6=0.5))
    for r in rows:
        if r["arm"] == "A6":
            r["prediction"] = "Keep healthy"
    make_rating_sheets(rows, str(tmp_path / "s"))
    s = summarize_rating_files(str(tmp_path / "s"), "A1", "A6")
    assert s["to_rate"] == 0 and s["wins"] == 0 and s["sign_test_p"] == 1.0


def test_decision_rule_pending_pass_and_fail(tmp_path):
    rows = load_arm_rows(make_csv(tmp_path / "r.csv"))
    pending = decision_rule(rows, {})
    status = {c["condition"][:2]: c["status"] for c in pending}
    assert status["1."] == "PASS" and status["3."] == "PASS" and status["2."] == "PENDING"
    good = {"sign_test_p": 0.001, "wins": 11, "losses": 1, "ties": 0, "identical_pairs_tied": 0}
    # A5 is as weak as A1 in this fixture, so retrieval beats A5 as well
    full = decision_rule(rows, {"A1_vs_A6": good, "A5_vs_A6": good})
    assert [c["status"] for c in full if c["condition"][:2] != "4."] == ["PASS"] * 4
    assert "Verdict" in format_decision(full)
    flat = load_arm_rows(make_csv(tmp_path / "flat.csv", a1=0.5, a6=0.5))
    assert decision_rule(flat, {})[0]["status"] == "FAIL"
    assert "No demonstrated improvement" in format_decision(decision_rule(flat, {"A1_vs_A6": good, "A5_vs_A6": good}))


def test_guess_vs_retrieval_attribution(tmp_path):
    # A6 beats A1 and A5 only if retrieval helps beyond the disease hint; here A5 equals A6, so condition 5 fails
    path = make_csv(tmp_path / "r.csv", a1=0.2, a6=0.8)
    rows = load_arm_rows(path)
    a6 = {r["id"]: r["item_recall"] for r in rows if r["arm"] == "A6"}
    for r in rows:
        if r["arm"] == "A5":
            r["item_recall"] = a6.get(r["id"], r["item_recall"])
    good = {"sign_test_p": 0.001, "wins": 11, "losses": 1, "ties": 0, "identical_pairs_tied": 0}
    checks = decision_rule(rows, {"A1_vs_A6": good, "A5_vs_A6": good})
    assert checks[4]["status"] == "FAIL"
    assert "naming the disease, not from retrieval" in format_decision(checks)
