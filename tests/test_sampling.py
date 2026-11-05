from baseline.data import pool_summary, stratified_sample


def _r(q, cat="general", src="SLAKE", fmt="open"):
    return {"question": q, "category": cat, "source_dataset": src, "answer_format": fmt}


def test_caps_repeated_question_text_per_category():
    records = [_r("is the heart enlarged?", "severity") for _ in range(12)] + [_r("other", "severity")]
    sampled = stratified_sample(records, n_per_category=10, seed=0, max_per_question=3)
    heart = [r for r in sampled if r["question"] == "is the heart enlarged?"]
    assert len(heart) == 3
    assert len(sampled) == 4


def test_respects_n_per_category_and_is_deterministic():
    records = [_r(f"q{i}", "general") for i in range(30)] + [_r(f"l{i}", "laterality") for i in range(30)]
    a = stratified_sample(records, n_per_category=5, seed=1)
    b = stratified_sample(records, n_per_category=5, seed=1)
    assert [r["question"] for r in a] == [r["question"] for r in b]
    assert sum(r["category"] == "general" for r in a) == 5
    assert sum(r["category"] == "laterality" for r in a) == 5


def test_pool_summary_counts():
    records = [_r("a", "general", "SLAKE", "yes/no"), _r("b", "severity", "VQA-RAD", "open")]
    s = pool_summary(records)
    assert s["category"] == {"general": 1, "severity": 1}
    assert s["source"] == {"SLAKE": 1, "VQA-RAD": 1}
    assert s["answer_format"] == {"yes/no": 1, "open": 1}


def test_rib_matches_whole_words_only():
    from baseline.data import mentions_chest_term
    assert mentions_chest_term("Are the ribs fractured?")
    assert mentions_chest_term("Is there a rib fracture?")
    assert not mentions_chest_term("How would you describe the lesion?")
    assert not mentions_chest_term("what type of distributions is seen on this image?")
    assert not mentions_chest_term("is the mass well circumscribed?")
    assert not mentions_chest_term("what two tributaries of blood vessels are impacted?")


def test_chest_words_inside_compound_terms_still_match():
    from baseline.data import mentions_chest_term
    assert mentions_chest_term("is the right hemidiaphragm normal?")
    assert mentions_chest_term("is a pneumothorax present?")
    assert mentions_chest_term("how would you describe the aortopulmonary window?")
    assert mentions_chest_term("does this patient have pneumomediastinum?")
