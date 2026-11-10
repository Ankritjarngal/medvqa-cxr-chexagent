# Phase 2 plan: knowledge-grounded RAG (pre-registration)

Written **before any model has been run on any Phase 2 test** (disease naming or RAG arms). What is decided here is
fixed; what is tuned later is listed under "What may be tuned, and where". Decision rule: Amendment 5 in
`docs/evaluation_protocol.md`. Code: `baseline/disease_labels.py` (labels and disease scoring), `baseline/rag.py`
(corpus, retrieval, prompts), notebooks 04 and 05.

## 1. Why the plan has this shape

Run 3 put the model at the constant-answer floor on knowledge questions, and run 4 showed that what it says on
perception questions follows the wording, not the image. Looking at the data behind the knowledge questions showed:

- **SLAKE's knowledge answers are a lookup keyed by the disease on the image.** For the 50 non-function SLAKE knowledge
  questions whose image has a disease label, the gold answer is the same for every image with the same disease and
  question type (for example pneumonia gives "Bacterial infection" as cause and "Medical therapy, supportive therapy" as
  treatment each time). The question text ("the disease on the lower left of lung") never names the disease.
- So the task is **two steps: (a) name the disease from the image, (b) recall its cause, symptoms, treatment or
  prevention.** Retrieval can only help (b). Step (a) is perception, and in run 2 the model named the disease right 2 of 11
  times.
- The labelled benchmark is small: 130 SLAKE chest images with a disease label (8 classes after merging spellings;
  majority class Cardiomegaly 29 of 130 = 22%), and 50 non-function knowledge questions with a known disease (31 train,
  11 validation, 8 test). Held-out test rows alone are too few to prove anything.

The experiment is therefore split so that each question has its own answer: can the model name the disease (Stage 1);
does retrieved text help once the disease is known (oracle arms); does the whole system help (the real arms).

## 2. Stages

1. **Stage 1, disease naming** (notebook 04, Colab): three ways of asking on 130 images (open, forced choice from the
   ChestX-ray14 list plus "No Finding", per-class yes-probability). Scored against the majority label and shuffled labels.
2. **Stage 2, retrieval check without the model** (this VM): does the corpus contain the gold answer's items when the
   disease is known? Tunes the retrieval settings on dev images only.
3. **Stage 3, generation arms** (notebook 05, Colab).
4. **Stage 4, no-harm check**: how many baseline-set questions the router would send to retrieval; `function` questions.
5. **Stage 5 (optional, not started)**: perception-targeted RAG on the heart-size questions; image-based case retrieval.
   Only after Stages 1 to 4 are reported.

## 3. Disease naming (Stage 1), fixed in advance

- Items: every SLAKE chest image with a disease question (`disease_id_items`), 130 images (train 90, validation 20,
  test 20). Labels merge SLAKE spellings (`Pulmonary Mass` to Mass, `Noudle` to Nodule, `Pleural Effusion` to Effusion,
  `Pulmonary Infiltration` to Infiltration). 5 images have two labels.
- Candidate vocabulary shown to the model: the 14 ChestX-ray14 pathology names plus "No Finding", alphabetical. This is a
  public list, not the list of diseases found in SLAKE.
- Methods: `open` ("What disease is shown in this chest X-ray?"), `forced` (pick one name from the list), `perclass`
  ("Is there X in this chest X-ray? Answer yes or no." for each of the 15 names; the predicted disease is the name with
  the highest yes-minus-no score). Free-text answers are parsed by `parse_diseases`; unparsed answers count as wrong.
- Scored: the top choice is correct if it matches any gold label (strict version, exactly one name equal to the whole
  gold set, also reported); against always answering the most common label and against shuffled gold labels; per-class
  recall; accuracy by split; for `perclass`, per-class AUC (does the score rank the true disease higher than other
  images, independent of how often the model says yes).
- **Method used downstream**: the one with the highest top-choice accuracy on the 20 validation images (ties go to the
  order open, forced, perclass; `choose_method`). Fixed here, before the numbers exist.
- **Reading**: the disease name is "not shown to be above chance" if p against the majority baseline is 0.05 or more, or
  accuracy does not exceed the 95th percentile of the shuffled-label control. In that case the report states that the
  bottleneck is perception and that the model-guess arms (A5, A6) are expected to show little. They are still run and
  reported.

## 4. Generation arms (Stage 3), fixed in advance

All arms: same model, greedy decoding, `max_new_tokens` 64, the image included, the same style instruction (below) so a
difference cannot be a style effect. Retrieval, when used, is placed before the question.

- **A0**: run 3 answer (plain question). Reference only; not style-matched.
- **A1**: style-matched, no retrieval, no disease. The fair baseline.
- **A2**: retrieval from the question text alone (oracle-set questions only). Shows what a missing disease costs.
- **A3**: the true disease named in the prompt, no retrieval (oracle-set only).
- **A4**: the true disease named, plus retrieval (oracle-set only).
- **A5**: the model's own disease guess (the Stage 1 method) named in the prompt, no retrieval.
- **A6**: the model's own guess plus retrieval. This is the system.

Style instruction appended to every arm: "Answer with a short comma-separated list of 2 to 4 items." Disease hint
wording: "The finding in this image is {disease}." (a model guess is stated the same way as a true label, so the model
cannot tell them apart). Retrieved context wording: "Reference: {passages}".

- **Which questions**: A1, A5 and A6 run on all 105 non-function knowledge questions (99 SLAKE, 6 VQA-RAD), and on the 25
  `function` questions as a no-harm check. A2, A3 and A4 run only on the 50 questions with a known single disease.
- **Query**: for symptoms, treatment, prevention and cause, "{disease} {subtype words}" (for example "pneumonia causes
  etiology"). For explanation and other questions (the VQA-RAD "why" / "how do you determine"), the question text itself.
  No gold answer and no SLAKE annotation is used by A5 or A6; only A2 to A4 use the true label, and they are labelled
  oracle everywhere.
- **Routing**: only questions matched by `is_knowledge_question` are sent to retrieval; everything else keeps the baseline
  path unchanged.

## 5. What may be tuned, and where

- Retrieval settings (source: textbooks / MedlinePlus / both; passages k in 1, 2, 3; passage length 400 or 700
  characters; query word list per subtype) are chosen in Stage 2 **without the model**, using how many gold items appear
  in the retrieved text for **dev images only** (`split_dev_heldout`, dev about 30% by image, about 15 oracle questions).
  The chosen setting is written into the experiment log before any Stage 3 run. A token-budget check in the Stage 3
  notebook may shorten passages if image plus prompt exceed 2048 tokens; this is logged and applies to all arms.
- Nothing else is tuned. The Stage 1 method is chosen on validation images as above.
- Held-out knowledge questions are scored once, after both choices are frozen.

## 6. Scoring and tests

- Item recall with scorer version 2 (frozen), and the constant-answer floor and the number of distinct answers per
  subtype (Amendment 2). Blind manual ratings (0/1/2, `RATING_RUBRIC`): A1 versus A6 on held-out non-function questions, and
  A3 versus A4 on the held-out oracle questions.
- Paired comparisons by image-clustered bootstrap (item recall), exact sign test (ratings), exact McNemar for any binary
  outcome, as in the protocol. All results are also shown for train rows and for validation/test rows separately.
- Comparisons that will all be reported: A1 vs A6 (the system), A5 vs A6 (what retrieval adds to the disease guess), A1 vs
  A5 (what the guess alone adds), A3 vs A1, A4 vs A3 and A4 vs A1 (oracle decomposition), A2 vs A1.

## 7. How outcomes will be read (fixed in advance)

| Outcome | Reading |
|---|---|
| Disease naming not above chance | The model cannot supply step (a). The system cannot beat what the oracle arms show is possible, and A6 is expected to be near A1. Perception is the bottleneck; propose fine-tuning (Phase 3) or a different model. |
| A3 above A1 but A4 not above A3 | Knowing the disease helps; the retrieved text adds nothing the model did not already know. |
| A4 above A3 | Retrieved text helps once the disease is known (an upper bound on what retrieval can do on these questions). |
| A6 above A1 and above A5, meeting the Amendment 5 rule | A bounded gain on SLAKE template questions, with the caveats below. |
| A6 above A1 but not above A5 | The gain is from naming the disease, not from retrieval. Not reported as a retrieval gain. |
| A6 not above A1 | No demonstrated improvement. A valid result. |

## 8. Corpus (independent of SLAKE)

- **Textbooks**: the MedRAG textbooks corpus on Hugging Face (18 textbooks, 125,847 chunks of at most 1000 characters,
  about 212 MB). Licence not stated and the books are copyrighted; used for research only, downloaded at run time, not
  committed to this repository.
- **MedlinePlus Health Topics** (National Library of Medicine, public domain), health-topic pages for the diseases.
- Not used: SLAKE's knowledge graph, any SLAKE or VQA-RAD answer text, MIMIC-CXR.
- Retrieval: BM25 over the title and text (`baseline/rag.py`). No dense retriever in this round; the first question is
  whether the corpus holds the gold items at all.

## 9. Limits stated before the run

- 50 oracle questions on 7 diseases (31 train, 11 validation, 8 test). The model has very probably seen SLAKE train rows.
  Held-out dev/test counts are small; large effects only are detectable.
- Gold answers are SLAKE's own short phrases ("Keep healthy" for pulmonary mass). A medically correct textbook answer can
  score low on item recall; the blind rating decides, and the rater is a student.
- A 2048-token context allows about one to three short passages.
- Results say nothing about Medical VQA in general and are not clinical evidence.

## 10. Frozen settings and clarifications (added before any model run and before any held-out retrieval number)

**Clarifications to sections 3 and 4**
- Passages in the prompt are limited to 1400 characters in total (about 350 tokens), because the model's context is
  2048 tokens including the image. The Stage 2 grid only contains settings within that limit.
- If the model's guessed disease is "No Finding", or no disease name can be parsed, A5 gets no disease hint and A6
  retrieves with the question text only (`usable_disease`, `build_query`).
- Retrieval sources considered in Stage 2: `textbooks`, `medlineplus`, `medlineplus_topic` (BM25 restricted to the
  MedlinePlus topics for the disease, `MEDLINEPLUS_TOPICS` in `baseline/rag.py`) and `topic+textbooks` (the best topic
  passage first, the remaining passages from the textbooks).

**Stage 2 result on dev questions (16 oracle questions, 16 images; no model; no held-out question touched)**
Two rounds, both logged in `docs/experiment_log.md` (entry 14).
1. Round 1, query = radiology name plus question-type words, sources textbooks / MedlinePlus / both: best strict
   coverage 0.27, under the 0.30 gate. Diagnosis (dev only): the question-type words pulled in unrelated articles, and some
   diseases are filed under another name (cardiomegaly under cardiomyopathy and "enlarged heart", atelectasis and
   pneumothorax under "collapsed lung", effusion under "pleural disorders").
2. Round 2, after adding alternative disease words (`QUERY_NAMES`) and the topic lookup: best strict coverage 0.42.

**Frozen retrieval setting for Stage 3, chosen by the pre-stated rule (`choose_setting`):**
source = `topic+textbooks`, k = 2 passages, 700 characters each, disease weight 1.

The coverage numbers on 16 questions come from picking the best of 24 settings and are optimistic; the held-out
numbers, reported after this commit, are the honest ones.

## 11. Status and how to run it

Stage 2 (retrieval check) is done and Stage 4 (router check) is done; see `docs/experiment_log.md`, entries 13 to 16. **The first
run of notebook 04 was text-only because the image never reached the model** (entry 17, Amendment 6); the notebooks were fixed and
rebuilt, and must be rerun. Nothing in this plan changes: arms, thresholds, retrieval setting and decision rule are as frozen.

1. Open `notebooks/04_disease_naming.ipynb` in Colab (T4). Run with `QUICK = True` first, then `False`. Note the
   `CHOSEN_METHOD` it prints, and download `disease_results.csv` and `disease_checkpoint.json`.
2. Open `notebooks/05_rag_knowledge_arms.ipynb`, paste `CHOSEN_METHOD` into the build cell (`DISEASE_METHOD = None`), run with
   `QUICK = True` then `False`. Download `rag_results.csv`, `rag_settings.json` and `rag_checkpoint.json`.
3. Upload the files. Analysis here:
   `python3 scripts/analyze_disease.py disease_results.csv` and
   `python3 scripts/analyze_rag.py rag_results.csv --make-sheets ratings/`; the reviewer fills `rating_1` / `rating_2`
   (0, 1, 2) in the three sheets, then `python3 scripts/analyze_rag.py rag_results.csv --ratings ratings/` prints the decision
   rule.

**Stage 4 result:** 25 of the 219 baseline questions are routed to retrieval, all of them also in the knowledge test set. The
rest are unchanged by construction, so protocol condition 4 uses those 25 plus the `function` questions.

