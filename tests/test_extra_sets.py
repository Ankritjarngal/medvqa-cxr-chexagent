from baseline.extra_sets import describe_set, prompt_for_set, select_knowledge_records, select_side_records


def _rec(q, gold, key="img1", src="SLAKE", split="test", knowledge=False, fmt="open", uid=None):
    return {"question": q, "gold_answer": gold, "image_key": key, "source_dataset": src,
            "source_split": split, "is_knowledge": knowledge, "answer_format": fmt,
            "category": "general", "uid": uid or f"{src}_{q}_{key}"}


def test_knowledge_selection_dedupes_and_tags_subtype():
    pool = [
        _rec("How to treat the disease on the left of lung?", "Medical treatment", knowledge=True),
        _rec("How to treat the disease on the left of lung?", "Medical treatment", knowledge=True),
        _rec("How to treat the disease on the left of lung?", "Medical treatment", key="img2", knowledge=True),
        _rec("Is the heart enlarged?", "yes", fmt="yes/no"),
    ]
    out = select_knowledge_records(pool)
    assert len(out) == 2
    assert {r["knowledge_subtype"] for r in out} == {"treatment"}


def test_side_selection_is_balanced_and_capped():
    pool = (
        [_rec("Where is the abnormality?", "Lower Left Lung", key=f"l{i}") for i in range(30)]
        + [_rec(f"Which side {i}?", "Right", key=f"r{i}") for i in range(30)]
        + [_rec("Where?", "Both Lung", key=f"b{i}") for i in range(5)]
        + [_rec("Is it on the left?", "yes", fmt="yes/no")]
        + [_rec("Where is the pneumonia in detail please?", "In the upper left lobe of lung near the heart")]
        + [_rec("How to treat the left lung disease?", "Left", knowledge=True)]
    )
    out = select_side_records(pool, n_per_side=10, n_both=3, max_per_question_per_side=4, seed=0)
    sides = [r["gold_side"] for r in out]
    assert sides.count("right") == 10 and sides.count("both") == 3
    assert sides.count("left") == 4  # all 30 left rows share one question text, capped at 4
    assert all(r["answer_format"] == "open" and not r["is_knowledge"] for r in out)
    assert select_side_records(pool, 10, 3, 4, seed=0) == out


def test_side_granularity_tag():
    out = select_side_records([_rec("Which side?", "Left"), _rec("Where?", "Lower Left Lung")], n_per_side=5)
    assert {r["side_granularity"] for r in out} == {"side_only", "region"}


def test_describe_set_counts():
    d = describe_set([{**_rec("q1", "a"), "knowledge_subtype": "cause"}, {**_rec("q2", "a", key="img2"), "knowledge_subtype": "cause"}])
    assert d["n"] == 2 and d["unique_gold_answers"] == 1 and d["unique_images"] == 2


def test_prompts_differ_between_sets():
    assert prompt_for_set("knowledge", "Why?", "laterality") == "Why?"
    assert "visible in the image" in prompt_for_set("side", "Which side?", "laterality")
