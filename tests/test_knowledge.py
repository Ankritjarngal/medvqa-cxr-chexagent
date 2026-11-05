import math

from baseline.categorize import is_knowledge_question, knowledge_subtype
from baseline.knowledge import (canonicalize, item_scores, make_blind_review, split_items,
                                summarize_blind_review, summarize_knowledge, score_knowledge_rows)


def test_synonym_canonicalization_is_not_doubled():
    assert canonicalize("Shortness of breath") == "dyspnea"
    assert canonicalize("Surgical treatment") == "surgery"
    assert canonicalize("Medical therapy") == "medication"
    assert canonicalize("coughing up blood") == "hemoptysis"


def test_split_items():
    assert split_items("Chest pain, dyspnea and cough") == ["Chest pain", "dyspnea", "cough"]


def test_item_scores_credit_paraphrase_that_exact_match_misses():
    s = item_scores("Chest tightness, shortness of breath", "Chest pain, dyspnea")
    assert math.isclose(s.recall, 0.5)
    assert math.isclose(s.precision, 0.5)


def test_item_scores_identical_and_disjoint():
    assert item_scores("Breathe", "Breathe").f1 == 1.0
    assert item_scores("Medical treatment, surgical treatment", "Medical treatment, surgical treatment").f1 == 1.0
    assert item_scores("Bacterial infection", "Bronchial obstruction").f1 == 0.0


def test_item_scores_partial_list_and_extra_items():
    s = item_scores("Medical treatment, surgical treatment", "Medical therapy, supportive therapy")
    assert math.isclose(s.recall, 0.5)
    assert math.isclose(s.precision, 0.5)
    s2 = item_scores("cough", "cough, fever, chest pain")
    assert math.isclose(s2.recall, 1 / 3) and s2.precision == 1.0


def test_item_scores_empty_inputs():
    assert item_scores("", "Breathe").f1 == 0.0


def test_knowledge_tagging_additions():
    assert is_knowledge_question("why is the left hemithorax more lucent?")
    assert is_knowledge_question("how can you tell there is no pneumothorax present?")
    assert not is_knowledge_question("is the heart enlarged?")
    assert knowledge_subtype("What kind of symptoms will the disease on the upper right of lung bring to the patient?") == "symptoms"
    assert knowledge_subtype("How to prevent the occurrence of the disease on the lower left lobe of lung in this image?") == "prevention"
    assert knowledge_subtype("How to treat the disease located on the left of lung in this image?") == "treatment"
    assert knowledge_subtype("What is the main cause of the disease on the lower right of the lung in the picture?") == "cause"
    assert knowledge_subtype("What is the function of the main organ in this picture?") == "function"
    assert knowledge_subtype("why is the right lung most common location for aspirations?") == "explanation"


def test_summarize_knowledge_groups_by_subtype():
    rows = score_knowledge_rows([
        {"id": "1", "prediction": "Breathe", "gold_answer": "Breathe", "knowledge_subtype": "function"},
        {"id": "2", "prediction": "Keep healthy", "gold_answer": "Enhance physical fitness", "knowledge_subtype": "prevention"},
    ])
    s = summarize_knowledge(rows)
    assert s["all"]["n"] == 2 and s["function"]["item_f1"] == 1.0 and s["prevention"]["item_f1"] == 0.0


def _rows(preds):
    return [{"id": str(i), "question": f"q{i}", "gold_answer": "g", "prediction": p} for i, p in enumerate(preds)]


def test_blind_review_hides_system_and_unblinds_correctly():
    base = _rows([f"base{i}" for i in range(40)])
    cand = _rows([f"cand{i}" for i in range(40)])
    sheet, key = make_blind_review(base, cand, seed=1, candidate_name="rag")
    assert len(sheet) == len(key) == 40
    assert {k["answer_1_is"] for k in key} == {"baseline", "rag"}  # both orders occur
    for row, k in zip(sheet, key):
        first_is_rag = k["answer_1_is"] == "rag"
        assert row["answer_1"].startswith("cand" if first_is_rag else "base")
        # reviewer rates the candidate answer 2, the baseline answer 0
        row["rating_1"], row["rating_2"] = (2, 0) if first_is_rag else (0, 2)
    out = summarize_blind_review(sheet, key, candidate_name="rag")
    assert out["mean_rag"] == 2.0 and out["mean_baseline"] == 0.0
    assert out["wins"] == 40 and out["losses"] == 0 and out["sign_test_p"] < 0.001


def test_blind_review_skips_unrated_rows():
    base, cand = _rows(["a", "b"]), _rows(["c", "d"])
    sheet, key = make_blind_review(base, cand, seed=0)
    sheet[0]["rating_1"], sheet[0]["rating_2"] = 1, 1
    out = summarize_blind_review(sheet, key)
    assert out["n_rated"] == 1 and out["ties"] == 1
