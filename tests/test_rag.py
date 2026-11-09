from baseline.rag import (
    BM25, STYLE_INSTRUCTION, build_prompt, build_query, gold_item_coverage, medlineplus_passages_from_xml,
    retrieve_passages, tokenize, truncate_passage, usable_disease,
)

CORPUS = [
    {"title": "Pulmonology", "text": "Pneumonia is caused by bacterial infection of the lung. Bacteria such as pneumococcus."},
    {"title": "Cardiology", "text": "Cardiomegaly follows long standing hypertension and dilated cardiomyopathy."},
    {"title": "Surgery", "text": "A pneumothorax may need a chest tube. Smoking cessation lowers the risk of recurrence."},
    {"title": "Noise", "text": "The kidney filters blood. Nothing about the chest appears here."},
]


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The Lungs are keeping healthy") == ["lung", "keep", "healthy"]


def test_bm25_ranks_the_matching_passage_first_and_is_deterministic():
    bm = BM25(CORPUS)
    hits = bm.search("pneumonia bacterial infection", k=3)
    assert hits[0][1]["title"] == "Pulmonology"
    assert [h[1]["title"] for h in bm.search("pneumonia bacterial infection", k=3)] == [h[1]["title"] for h in hits]
    assert bm.search("zzzz unknown", k=3) == []


def test_repeated_query_term_weighs_more():
    bm = BM25(CORPUS)
    plain = bm.search("pneumonia chest", k=4)
    weighted = bm.search("pneumonia pneumonia pneumonia chest", k=4)
    assert weighted[0][1]["title"] == "Pulmonology"
    assert weighted[0][0] > plain[0][0]


def test_title_counts_for_matching():
    bm = BM25([{"title": "Pleural effusion", "text": "Fluid builds up."}, {"title": "Other", "text": "Fluid builds up."}])
    assert bm.search("pleural effusion", k=1)[0][1]["title"] == "Pleural effusion"


def test_truncate_prefers_sentence_end():
    text = "First sentence is here. Second sentence goes on and on without end beyond the limit"
    assert truncate_passage(text, 40) == "First sentence is here."
    assert truncate_passage("short", 40) == "short"
    assert truncate_passage("one two three four five six seven", 15).endswith("...")


def test_build_query_uses_disease_only_when_usable():
    q = "What is the main cause of the disease on the lower left of the lung in the picture?"
    assert build_query(q, "cause", "Pneumonia") == "pneumonia causes etiology"
    assert build_query(q, "cause", "Pneumonia", disease_weight=2) == "pneumonia pneumonia causes etiology"
    assert build_query(q, "cause", None) == q
    assert build_query(q, "cause", "No Finding") == q
    assert build_query("how do you determine cardiomegaly?", "explanation", "Cardiomegaly") == "how do you determine cardiomegaly?"
    assert not usable_disease("Unknown disease") and usable_disease("Mass")


def test_build_prompt_arms():
    q = "How to treat the disease?"
    assert build_prompt(q) == f"{q}\n{STYLE_INSTRUCTION}"
    with_hint = build_prompt(q, disease="Effusion")
    assert "The finding in this image is pleural effusion." in with_hint and with_hint.endswith(STYLE_INSTRUCTION)
    passages = [{"text_used": "Treat with drainage."}]
    full = build_prompt(q, disease="Effusion", passages=passages)
    assert full.startswith("Reference: Treat with drainage.") and q in full
    assert build_prompt(q, disease="No Finding") == build_prompt(q)


def test_retrieve_passages_truncates_and_scores():
    bm = BM25(CORPUS)
    hit = retrieve_passages(bm, "pneumonia bacterial", k=1, max_chars=30)[0]
    assert len(hit["text_used"]) <= 33 and hit["score"] > 0 and hit["title"] == "Pulmonology"


def test_gold_item_coverage_loose_vs_strict():
    passages = [{"text": "Pneumonia is caused by bacterial infection. Rest helps. Surgery is rare."}]
    cov = gold_item_coverage(passages, "Bacterial infection, surgery")
    assert cov == {"loose": 1.0, "strict": 1.0}
    split = [{"text": "Bacterial overgrowth. Chest infection follows."}]
    cov = gold_item_coverage(split, "Bacterial infection")
    assert cov["loose"] == 1.0 and cov["strict"] == 0.0     # words present, but never in one sentence
    assert gold_item_coverage(passages, "Quit smoking") == {"loose": 0.0, "strict": 0.0}
    assert gold_item_coverage(passages, "") == {"loose": 0.0, "strict": 0.0}


def test_gold_item_coverage_uses_synonyms_and_text_used():
    passages = [{"text": "long text " * 50, "text_used": "Patients report shortness of breath."}]
    assert gold_item_coverage(passages, "Dyspnea")["strict"] == 1.0


def test_medlineplus_sections_split_on_headings():
    xml = """<health-topics>
      <health-topic title="Pneumonia" id="1" language="English" url="https://x/p.html">
        <full-summary>&lt;h3&gt;What is pneumonia?&lt;/h3&gt;&lt;p&gt;Pneumonia is an infection in one or both of the lungs that fills air sacs.&lt;/p&gt;
        &lt;h3&gt;What causes pneumonia?&lt;/h3&gt;&lt;p&gt;Bacteria, viruses and fungi can cause pneumonia in people of all ages.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Bacteria&lt;/li&gt;&lt;/ul&gt;
        </full-summary></health-topic>
      <health-topic title="Neumonia" id="2" language="Spanish" url="https://x/n.html"><full-summary>&lt;p&gt;texto&lt;/p&gt;</full-summary></health-topic>
    </health-topics>"""
    out = medlineplus_passages_from_xml(xml)
    assert [p["title"] for p in out] == ["Pneumonia: What is pneumonia?", "Pneumonia: What causes pneumonia?"]
    assert "Bacteria; " in out[1]["text"] or out[1]["text"].endswith("Bacteria;")
    assert all(p["source"] == "medlineplus" for p in out)


def test_coverage_for_setting_and_choose_setting():
    from baseline.rag import choose_setting, coverage_for_setting, mean_coverage
    bm = {"textbooks": BM25(CORPUS)}
    recs = [{"id": "a", "question": "What causes it?", "knowledge_subtype": "cause", "gold_answer": "Bacterial infection",
             "oracle_disease": "Pneumonia"},
            {"id": "b", "question": "What causes it?", "knowledge_subtype": "cause", "gold_answer": "Hypertension",
             "oracle_disease": None}]
    rows = coverage_for_setting(recs, bm, "textbooks", k=2, max_chars=300)
    assert rows[0]["strict"] == 1.0 and rows[0]["query"] == "pneumonia causes etiology"
    assert rows[1]["query"] == "What causes it?"
    assert mean_coverage(rows)["n"] == 2
    res = {("a", 1, 400, 1): {"strict": 0.5, "loose": 0.9}, ("b", 2, 400, 1): {"strict": 0.5, "loose": 0.9},
           ("c", 1, 700, 1): {"strict": 0.4, "loose": 1.0}}
    assert choose_setting(res) == ("a", 1, 400, 1)      # equal coverage: the shorter context wins


def test_topic_filter_and_retrieve_context_sources():
    from baseline.rag import retrieve_context
    mp = BM25([
        {"topic": "Pneumonia", "title": "Pneumonia: What causes pneumonia?", "text": "Bacteria and viruses cause pneumonia."},
        {"topic": "Pneumonia", "title": "Pneumonia: Treatment", "text": "Antibiotics treat bacterial pneumonia."},
        {"topic": "Hiccups", "title": "Hiccups: causes", "text": "Hiccups cause spasms; causes etiology of hiccups causes."},
    ])
    tb = BM25(CORPUS)
    r = {"textbooks": tb, "medlineplus": mp}
    # restricted search never returns another topic even when it scores higher
    assert {h[1]["topic"] for h in mp.search("causes etiology hiccups", k=3, topics={"Pneumonia"})} == {"Pneumonia"}
    got = retrieve_context(r, "medlineplus_topic", "q", "cause", "Pneumonia", k=2, max_chars=200)
    assert {p["topic"] for p in got} == {"Pneumonia"} and len(got) == 2
    mixed = retrieve_context(r, "topic+textbooks", "q", "cause", "Pneumonia", k=2, max_chars=200)
    assert mixed[0]["topic"] == "Pneumonia"
    assert len(mixed) == 2
    # no usable disease: plain search over the whole corpus
    assert len(retrieve_context(r, "medlineplus_topic", "hiccups causes", "cause", None, k=1, max_chars=200)) == 1
    try:
        retrieve_context(r, "nope", "q", "cause", "Pneumonia", 1, 100)
        assert False
    except ValueError:
        pass


def test_mapped_topics_exist_in_cache_when_available():
    import pytest
    from pathlib import Path
    from baseline.rag import MEDLINEPLUS_TOPICS, QUERY_NAMES, load_medlineplus_passages
    from baseline.disease_labels import CHESTXRAY14
    assert set(MEDLINEPLUS_TOPICS) == set(QUERY_NAMES) == set(CHESTXRAY14)
    if not list(Path("rag_cache").glob("mplus_topics_*.xml")):
        pytest.skip("MedlinePlus file not downloaded")
    titles = {p["topic"] for p in load_medlineplus_passages()}
    for d, ts in MEDLINEPLUS_TOPICS.items():
        assert ts <= titles, (d, ts - titles)
