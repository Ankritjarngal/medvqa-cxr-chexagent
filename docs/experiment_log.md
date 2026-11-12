# Experiment log

Chronological record of what was done, what it showed, and what was changed or corrected afterwards.
Newest entries at the bottom. Add an entry before changing any rule in `docs/evaluation_protocol.md`, and
record corrections here, not by quietly editing earlier notes.

Dates are given only where they were recorded. Commit ids are on the Cursor remote unless stated.

---

## 1. Literature review and plan

Six papers reviewed (PubMedCLIP, LLaVA-Med, Med-Flamingo, CheXagent, PMC-VQA/MedVInT, RULE). Plan:
Phase 1 baseline of a pretrained model, Phase 2 knowledge-grounded retrieval aimed at whatever Phase 1
shows is weak, Phase 3 targeted fine-tuning only if needed. Choice of model: `StanfordAIMI/CheXagent-2-3b`
(3.1B parameters, runs on a free Colab T4). See `docs/literature_review.md`.

Known caveat recorded from the start: CheXagent's training data (CheXinstruct) includes SLAKE and
VQA-RAD, so tests on them are close to in-distribution.

## 2. Repository and tooling

GitHub repository created and connected (`Ankritjarngal/medvqa-cxr-chexagent`). Push access needed a
`GITHUB_TOKEN` secret; the Cursor cloud agent that wrote this log has no GitHub credentials, so pushes to
GitHub are done by a separate agent with the token (see `docs/AGENT_HANDOFF.md`).

## 3. Colab environment problems and fixes

| Problem | Cause | Fix |
|---|---|---|
| `tokenizers` fails to build | Colab moved to Python 3.13; `transformers==4.40.0` needs `tokenizers<0.20` with no wheel for it | Use `transformers==4.46.3` |
| `AssertionError` about version 4.40.0 | The model's own code asserts the exact version | `patch_version_asserts()` replaces the two asserts with `pass` in a local copy |
| `generate()` fails on `DynamicCache` | `transformers >= 4.50` removed methods the model code uses | Pin 4.46.3 |
| Numerical problems on T4 | T4 has no native bfloat16 | float16 on T4 |

## 4. Pilot run 1 (60 questions)

20 per keyword-assigned category. Overall 61.7%. Observations: laterality lowest (45%) with overlapping
intervals; yes/no sensitivity 33% vs specificity 90% (the model says "no" too often); strict string match
too harsh on open answers. The set itself had problems: a non-clinical radiation-damage template question
scored as "severity", one question repeated 12 times, and some non-chest VQA-RAD items.
Write-up: `docs/pilot_run_1_findings.md`. Data: `results/pilot_run_1/`.

## 5. Pipeline improvements after the pilot

Dropped the radiation template, capped repeated question text at 4 per category, added `token_f1`,
Wilson confidence intervals, yes/no sensitivity and specificity with a majority-class baseline, a
knowledge-vs-perception tag, stable question ids taken from the datasets' own row ids (the first scheme
used signed URLs that change every fetch), and 100 questions per category. Code: `baseline/`.

## 6. Run 2 (224 questions)

100 general, 100 laterality, 24 severity (the pool contains few severity questions); 119 SLAKE, 105 VQA-RAD.
Overall 57.1% (51-63%). Committed as `fc38cf0`. Full analysis: `docs/baseline_run_2_findings.md`.

Findings, in order of how much they can be trusted:

1. Never answered "yes" on 8 enlargement questions where "yes" was correct (0 of 8; about 0.4% chance if
   its true sensitivity were 50%). Reproduces the pilot.
2. Laterality 44% vs general 70% (p = 0.0003), but the gap is confounded by dataset and question template: it
   is clear in SLAKE and absent in VQA-RAD.
3. Knowledge-type questions 4 of 24 by exact match (later re-scored, see entry 8).
4. Disease naming 2 of 11; canned answers ("Pneumonia", "Effusion").

Analysis checks run: Fisher tests between buckets; breakdown by source, by question template, by answer
format; whether the laterality gap survived removing knowledge and naming questions (it shrank from 26 to 17
points and stayed significant); within-dataset comparisons (which showed the confound).

Inspection of 12 random wrong answers (text only) showed a possible right-side default and one case of the
model answering "No" to a "left or right" question.

## 7. Planning the next steps (5 Nov 2026)

Decision: retrieval (Phase 2) aimed at knowledge-type questions; the enlargement blind spot is recorded as
a measured limitation and tested separately later. Before Phase 2, build bigger evaluation sets and fix the
scoring and the comparison rules, so that a later improvement can be tested honestly. Question raised and
answered: can an improvement be proven against a benchmark? Yes, with limits (small knowledge set, model
has probably seen the data, public sets mostly test perception). Recorded in
`docs/evaluation_protocol.md`.

## 8. Phase 1b: knowledge and side test sets (5 Nov 2026)

Commits: `27208cc`, `4c19a73` and the documentation commit that adds this entry.

**What was done**

1. Fetched candidates from SLAKE (test, validation, train) and VQA-RAD. Knowledge pool: 127 questions
   before de-duplication, about 5 templates. Side pool: 211 questions whose short gold answer names a side.
2. Built the knowledge set (130 questions) and the left/right-balanced side set (140 questions).
3. Wrote scoring that suits each set: side extraction and bias report (`baseline/sides.py`), item-level
   scoring with a synonym table and blind review sheets (`baseline/knowledge.py`), paired comparison tools
   (`baseline/compare.py`, `baseline/stats.py`).
4. Wrote `docs/evaluation_protocol.md` with the decision rule, written before any retrieval result exists.
5. Added a dev / held-out split by image so retrieval can be tuned without touching the final test.
6. Built `notebooks/02_knowledge_and_side_baselines.ipynb`.

**Things found along the way**

- **The knowledge pool is small and repetitive.** 130 questions, 42 wordings, 33 distinct gold answers. More
  questions did not mean more independent tests. This is why comparisons resample by image, results are split
  by subtype, and the protocol states what the set cannot detect.
- **Prompt confound in run 2.** 20 of 24 knowledge questions got the suffix "based only on what is visible in
  the image". The new knowledge set uses the plain question; the 24 shared questions allow a direct check.
- **Run 2's "laterality" tag is by question wording.** "Where is the abnormality?" with answer "Lower Left Lung"
  was tagged `general`. The side set selects by the gold answer.
- **Filter bug.** Viewing the 30 VQA-RAD images in the side set exposed a brain scan. The keyword `rib`
  matched inside `describe`, `distributions`, `circumscribed`, `tributaries`. Fixed with whole-word matching
  for `rib`/`ribs` only. 26 of 432 VQA-RAD candidates were affected; 5 of them are in run 2, all answered
  wrong (`results/run_2/excluded_ids.txt`). Run 2 without them: 58.4% overall; no conclusion changes.
  A broader fix (word-start matching for all keywords) was tested and rejected because it dropped real chest
  terms such as `hemidiaphragm` and `pneumothorax`.
- **Notebook bug caught by running it.** Executing the notebook's cells with a stand-in model showed that the
  inlined `extra_sets.py` would run its command-line `main()` and crash on Colab's own arguments. The
  builder now strips `__main__` blocks.
- **Scorer sanity check.** With a stand-in model that always answers "Right", the side report flags a
  complete left-to-right error rate with p < 0.001, as it should.

**Corrections to earlier statements**

- Run 2's knowledge result was described as "mostly missed" (4 of 24 exact). Item-level scoring shows partial
  success on treatment (69% recall) and symptoms (46%) and zero on cause, prevention and "why" (9 questions).
  The treatment credit largely comes from one generic answer. `docs/baseline_run_2_findings.md` was corrected.
- Run 2's side results (11 of 13 "right") are a lead only. With the wider definition (23 questions) the
  left-to-right vs right-to-left asymmetry is p = 0.10.

**Not done / limits**

- The model has not been run on the new sets. That needs a Colab T4 and is the next step for the user.
- No full image-by-image review of run 2's VQA-RAD rows (only the 30 side-set images and 12 text samples).
- No retrieval corpus chosen yet. The protocol requires it to be independent of SLAKE's knowledge graph.
- The synonym table is hand-written and not clinically reviewed.

**Next**

1. Run notebook 02 on Colab; save `knowledge_results.csv` and `side_results.csv` as `results/run_3/`.
2. Read the side result using the table in `docs/phase1b_test_sets.md`; read the knowledge result per subtype
   against the most-common-answer floor.
3. Choose a retrieval corpus and record it here before building the retriever.

---

## 9. Colab failure: HTTP 429 when building the sets (10 Nov 2026)

**Symptom.** Notebook 02 stopped at the first build cell with `HTTPError: 429 Too Many Requests` from
`datasets-server.huggingface.co` (SLAKE train, offset 7600).

**Causes, all mine.**
1. Both datasets were read by paging through the Hugging Face `datasets-server` API at 100 rows per request:
   about 150 requests for the full pool. Colab's shared addresses are rate-limited by it.
2. The notebook fetched the full pool twice, once for each set.
3. Retries waited a few seconds and ignored the server's Retry-After.

My earlier tests ran from a different network and never hit the limit, so this was not caught.

**Fix.**
- Read the datasets' own files instead (SLAKE JSON x3, VQA-RAD parquet x2); the paging API is the fallback.
- Fetch the pool once and share it between the two sets.
- Retries honour Retry-After on 429 and back off exponentially; downloads are streamed to a temp file.
- Side effect: VQA-RAD images come from the parquet (no expiring URLs) and are identified by content hash.
  This changed the dev/held-out split (now 42 / 88) and the distinct-image counts (103 and 106); set sizes
  and source/split counts are unchanged (130 and 140). Docs updated.

**Verified.** Records identical between the two paths on SLAKE test and VQA-RAD test; all 224 run-2 ids
unchanged; a from-scratch build of both sets takes about 20 seconds; notebook 02's cells run end to end
with a stand-in model from an empty directory; 59 unit tests pass, including simulated 429 / 503 / 404
responses. Not verified: a run from a Colab address (that is the next real test).

**Lesson recorded.** Network behaviour on Colab differs from this sandbox. Any new data download should be
checked for request count and rate-limit handling before it goes into a notebook.

---

## 10. Run 3: knowledge and side sets run on Colab (10 Nov 2026)

Files uploaded by the user: `knowledge_results.csv` (130), `side_results.csv` (140) and the two test-set jsonl files.
Saved as `results/run_3/`. Checked first: row counts match, ids equal the sets built locally, no empty predictions,
the notebook's own `side_correct` column equals a recompute with the repo code. Full analysis: `docs/run_3_findings.md`.

**Results in one paragraph.** Side: on explicit "which side" questions the model said "Right" on 26 of 29 (90%), unrelated
to the correct side (13 of 14 gold-left and 13 of 15 gold-right answered "right"); elsewhere there is no right default;
overall, when it names left or right it is right 41 of 73 times (56%, chance-level); 30% of answers name no side;
one answer ("Left Lung, Right") is a template for one question wording. Knowledge: outside the trivial `function`
subtype (25 questions, one answer "Breathe") item recall is 0.37, equal to the constant-answer floor (0.38); only
symptoms clearly beat the floor; answers barely vary with the image.

**Analysis steps that mattered**
- Split side questions by wording. The overall asymmetry test (p = 0.037) understated the finding; the explicit
  questions alone show it clearly, and the other wordings show none. Lesson: the aggregate hid a sharp subset.
- First reading of "Left Lung, Right": it looked informative (10 of 19 bilateral, Fisher p < 0.001). Checking showed
  it is given on every question with that wording, so it reflects how the dataset assigns wording, not the image.
  Not reported as a finding about the model.
- Built a constant-answer floor for the knowledge set. This changed the reading from "model scores 0.45" to "model is
  at the floor, except on symptoms".
- Compared the 25 questions shared with run 2 to isolate the prompt change: no detectable accuracy effect.

**Changes made after seeing results (declared)**
1. Scorer v1 to v2: light suffix stripping, because "keeping healthy" scored 0 against "Keep healthy". Overall item recall
   0.45 to 0.49, prevention 0.00 to 0.22, nothing else changed. Test added. Both versions reported in the findings.
2. Protocol amendments 1-4 in `docs/evaluation_protocol.md` (exclude `function` from the primary metric; report the floor
   and answer variety; freeze scorer v2; perception diagnostics first). Made before any retrieval result exists.
3. Notebook 02 now keeps `image_key` in the saved test-set files. The uploaded files lack it, so the dev / held-out
   split could not be recomputed from them. The locally built set has identical ids and is used for the split.

**Not done / limits**
- The flip test, prompt variants and the heart-size "describe first" test were not built at this point (see entry 11).
- No manual (blind) rating of the knowledge answers yet; item scoring is a loose judge of full sentences.
- Findings 1 is on 29 questions from 2 datasets; the interval is wide (74-96%), though the direction is not in doubt.
- The explanation that cause / symptom answers depend on the disease, which the model cannot name, is a hypothesis.

**Next (proposed, pending decision):** a diagnostics notebook: (a) flipped-image side test; (b) prompt variants on the
explicit side questions; (c) heart-size questions with "describe the heart size first". All zero training, free GPU.

## 11. Phase 1c: perception diagnostics notebook (10 Nov 2026)

**Why:** run 3 showed perception faults (constant "Right", no "yes" on enlargement). Amendment 4 puts cheap diagnostics
before any retrieval work.

**Built (no model result yet)**
- `baseline/diagnostics.py`: flip test, six side-prompt variants, heart-size selection and four prompt variants,
  parsing of yes/no and "normal/enlarged" answers, reports (swap rate with Wilson interval, Fisher test of dependence on
  the true side, sensitivity / specificity / Youden's J).
- `notebooks/03_perception_diagnostics.ipynb` (built by `scripts/build_diagnostics_notebook.py`): about 750 short
  generations on a T4, checkpointed every 20 answers, `QUICK` smoke-test flag, results in long-format CSV.
- Test sets: 120 left/right side items (29 explicit "which side" questions) for the flip test; 60 of them for prompt
  variants; 47 heart-enlargement questions (22 "yes", 25 "no"; 40 VQA-RAD, 7 SLAKE).
- `docs/phase1c_diagnostics.md`: what each outcome will mean, written before the run.
- 83 unit tests pass.

**Checks done without the model:** ran every notebook code cell with two stand-in models. A model that always says
"Right"/"No" gives swap rate 0%, 98% "right", Fisher p = 1.0, sensitivity 0/22. A model that follows the image gives
swap rate 100%. This confirms the reports separate the two cases. It also showed that a perfect swap does not imply
correctness (the second stand-in swapped perfectly and was right 48% of the time), which is why accuracy is reported
next to swap rate. The stand-ins say nothing about CheXagent.

**Change made while building:** Youden's J added to the heart table, so a prompt that only makes the model say "yes"
more often cannot look like improvement.

**Not done / limits:** the model has not been run on these tests. The VM has no GPU, so the notebook must be run on
Colab by the user (the assistant cannot drive Colab). Samples are small (29 explicit side questions, 22 enlarged hearts).

**Next:** run the notebook on Colab, upload `diagnostics_results.csv`, analyse against the pre-stated readings (done, entry 12).

## 12. Run 4: perception diagnostics run on Colab (10 Nov 2026)

Full run (788 generations) uploaded as `results/run_4/`. Analysis: `scripts/analyze_diagnostics.py`; write-up:
`docs/phase1c_findings.md`. Readings applied are the ones fixed in `docs/phase1c_diagnostics.md` before the run.

**Findings**
1. Flip test: left/right answers swapped when the image was mirrored in 2 of 73 cases (3%, 95% CI 1-9%); 112 of 120
   answers were byte-identical on both images (29 of 29 on explicit questions). Repeatability 20 of 20. The model's side
   answer does not depend on the image's orientation.
2. Side wording: the default flips with the wording ("Right" with run 3's wording; "Left" in 52 of 60 with the one-word
   wording) and no wording made answers depend on the true side (all p > 0.25). No fix found.
3. Heart size: the wording moves the yes-rate (2% / 26% / 94% / 0% across four wordings) more than the image does. The
   plain wording (no instruction) found 9 of 22 enlarged hearts and wrongly flagged 3 of 25 normal (Fisher p = 0.042, not
   corrected for 4 wordings; Youden J 0.29). Under the pre-stated rule this is "partial", not a fix. "Describe first"
   answered a bare "Yes" 94% of the time and did not describe anything.

**Things that mattered in the analysis**
- Accuracy is reported next to the swap rate, and Youden's J next to sensitivity, so a changed default cannot pass as skill.
- The best heart wording (`plain`) was chosen after seeing four results, so its p-value is flagged as uncorrected; its
  gain over run 2's wording on the same questions is not significant (McNemar p = 0.23).
- A possible alternative explanation for the flip result (training with random mirroring) is stated in the findings as
  something we cannot check.

**Not done / limits:** small samples (29 explicit side questions, 22 enlarged hearts); six side wordings and four heart
wordings is a small hand-picked set; the model is tested on questions from its own probable training sets.

**Decisions pending (not made):** Phase 3 fine-tuning for laterality / enlargement, or a second model on the same
test sets; a larger heart-size set to test the `plain` lead once; knowledge answers still need a blind manual rating.

## 13. Phase 2 design from the data (10 Nov 2026)

Question: what scope gives the most informative RAG experiment? Looked at the data behind run 3's knowledge questions
before building anything.

**Found**
- SLAKE knowledge answers are a lookup keyed by the disease on the image; the question text never names the disease.
  Joined to SLAKE's own disease question on the same image: for the 50 non-function questions with a known single disease
  (31 train, 11 validation, 8 test) the gold answer is the same for the same disease and question type.
- 130 SLAKE chest images carry a disease label (8 classes after merging spellings; Cardiomegaly 29, Atelectasis 21,
  Mass 19, Nodule 18, Pneumonia 17, Effusion 13, Pneumothorax 10, Infiltration 9; 5 images with two labels). The majority
  label covers 22%.
- CheXagent-2-3b has a 2048-token context. Corpora checked as downloadable from this VM: MedRAG textbooks (125,847
  chunks), MedlinePlus bulk XML (1,017 English topics).

**Built (no model run)**
- `docs/phase2_rag_plan.md` and Amendment 5 (written first): arms A0 to A6, oracle arms labelled as upper bounds, the
  disease-naming test, which method is used downstream (best on the 20 validation images), and a fifth decision
  condition (retrieval must beat the same model-guessed disease without retrieval).
- `baseline/disease_labels.py` and `tests/test_disease_labels.py`: label extraction, spelling map, free-text parser,
  accuracy against majority and shuffled labels, per-class AUC.
- `baseline/rag.py` and `tests/test_rag.py`: BM25 (own implementation), corpus loaders, query and prompt builders,
  gold-item coverage scorer. The MedRAG textbooks are downloaded at run time into `rag_cache/`, which is git-ignored
  (copyrighted).

**Corrections to what I wrote earlier:** the dates of Amendment 5 and entry 11 said 11 Nov; today is 10 Nov. Fixed.

## 14. Stage 2: retrieval check without a model (10 Nov 2026)

`scripts/retrieval_check.py tune` measures, for the oracle questions on dev images only (16 questions), how many of the
gold answer's items appear in the retrieved text (strict: all words of an item inside one sentence; loose: anywhere).
It uses no model.

- **Round 1** (radiology disease name plus question-type words; sources textbooks / MedlinePlus / both; 30 settings): best
  strict coverage 0.27 (`medlineplus`, k=3, 400 characters). Under the 0.30 gate that was fixed in the plan.
- **Diagnosis (dev questions only):** the question-type words ("treatment management therapy") pulled in unrelated
  articles ("Clinical Trials", "Retinal Disorders"). Diseases that reference texts file under another name returned
  nothing relevant ("cardiac enlargement" gave Sudden Cardiac Arrest; "lung mass" gave bone mass). Prevention gold answers
  ("Live healthy, enhance physical fitness", "Keep healthy") are generic and cannot be matched by disease-specific text,
  which puts a ceiling on this metric for that subtype regardless of the corpus.
- **Round 2 changes** (made after seeing round 1, declared): alternative disease words in `QUERY_NAMES` and a topic lookup
  (`MEDLINEPLUS_TOPICS`), chosen from MedlinePlus's list of topic titles and medical knowledge, not from gold answers.
  Grid of 24 settings, with k x characters limited to 1400. Best strict coverage 0.42 (`topic+textbooks`, k=2, 700
  characters, weight 1), 0.47 loose. Passes the gate.
- **Frozen for Stage 3:** `topic+textbooks`, k=2, 700 characters, disease weight 1 (also recorded in the plan, section
  10, before any held-out number was produced).
- **Limits:** 16 dev questions; the best of 24 settings is optimistic. The held-out coverage, with a wrong-disease control
  and a question-only control, is reported next (entry 15).


## 15. Stage 2 result: frozen setting on held-out questions, with controls (10 Nov 2026)

`python3 scripts/retrieval_check.py report 2 topic+textbooks 700 1`, output in `results/phase2/retrieval_check.json`.
Strict gold-item coverage of the retrieved text (no model):

| Questions | n | true disease | wrong disease | question only |
|---|---|---|---|---|
| dev | 16 | 0.42 | 0.16 | 0.05 |
| held-out | 34 | 0.29 | 0.17 | 0.03 |
| all | 50 | 0.33 | 0.13 | 0.04 |

By subtype with the true disease, all 50: symptoms 0.67 (n=11), cause 0.44 (11), treatment 0.23 (15), prevention 0.08 (13).
Held-out: symptoms 0.60, cause 0.35, treatment 0.27, prevention 0.00.

**Reading**
- Knowing the disease matters for retrieval: 0.33 with the true disease against 0.13 with a wrong one and 0.04 from the
  question alone. Question-only retrieval finds almost none of the gold items, so A2 is expected to add nothing.
- The dev figure (0.42) shrank to 0.29 on held-out, as expected for a setting picked as the best of 24 on 16 questions.
  The held-out figure is the one to quote.
- The corpus holds the gold items for symptoms and cause (0.67, 0.44) but little for treatment and almost nothing for
  prevention. Gold treatment answers are generic category words ("Medical treatment, surgical treatment") and
  prevention answers are generic advice ("Live healthy, enhance physical fitness", "Keep healthy"), which disease-specific
  text rarely states in those words. This limits how much retrieval can raise item recall on those two subtypes whatever
  the model does; the blind rating is the fairer judge there.
- A wrong disease still reaches 0.13 to 0.17 because some items are generic ("surgery", "chest pain").
- This is a ceiling on what retrieved text offers a perfect reader, not a model result.

## 16. Phase 2 notebooks and analysis tools built; Stage 4 router check (10 Nov 2026)

**Built, none run on the model yet**
- `notebooks/04_disease_naming.ipynb` (builder `scripts/build_disease_notebook.py`): Stage 1. 130 labelled SLAKE chest images,
  three ways of asking (open, forced choice, per-class yes-minus-no score read from one generated token), reported against the
  majority label and shuffled labels, train and validation/test separately; prints `CHOSEN_METHOD` (best on the 20
  validation images) and the pre-stated reading.
- `notebooks/05_rag_knowledge_arms.ipynb` (builder `scripts/build_rag_notebook.py`): Stage 3. Arms A1 to A6 on the 130 knowledge
  questions (A2 to A4 on the 50 with a known disease), the frozen retrieval setting, a token-budget check, checkpointing every
  20 answers. `DISEASE_METHOD` must be pasted in from notebook 04; the notebook refuses to run without it.
- `baseline/rag_analysis.py`, `scripts/analyze_rag.py`, `scripts/analyze_disease.py`: item recall by arm with the constant-answer
  floor and distinct answers, all planned paired comparisons (image-clustered bootstrap, train and validation/test split), blind
  rating sheets for A1 vs A6, A5 vs A6 and A3 vs A4 (identical answer pairs are tied by definition and left off the sheets), and
  the decision rule of Amendment 5 as PASS / FAIL / PENDING lines. `constant_floor` and `distinct_answers` added to
  `baseline/knowledge.py` (the scorer itself is unchanged, version 2).
- `scripts/stub_run_notebook.py`: runs a notebook's code cells here with a stand-in model.

**Checks done without the model:** both notebooks ran end to end with two stand-in models. A model that ignores the image gives
disease-naming accuracy at or below the 22% majority rate and the "NOT shown to be above chance" reading; one that reads the
hidden label gives 100% and "above chance". The 05 run produced the 540 expected answers (A1, A5, A6 on 130 questions; A2 to
A4 on 50), the analysis script ran on its output, and a completed dummy rating sheet unblinded correctly. These say nothing about
CheXagent. 119 unit tests pass.

**Untested on the real model (first Colab run may need a fix):** the yes/no score (`yes_no_score` reads `scores[0]` of
`generate(..., output_scores=True)` for one token and the yes/no token ids), and the token counts (the stand-in used a word-count
estimate). Run each notebook with `QUICK = True` first.

**Stage 4 router check** (`scripts/router_check.py`, output `results/phase2/router_check.json`): of the 219 baseline-set questions,
25 match the knowledge router (cause 3, explanation 2, function 2, prevention 4, symptoms 6, treatment 8). All 25 are
questions of the knowledge test set, so Stage 3 already covers them; the other 194 never reach retrieval and are unchanged by
construction. Protocol condition 4 is therefore evaluated on those 25 (run 2's answer against A6), plus the 25 `function`
questions that must still answer "Breathe". No extra run is needed.

**Limits stated again:** the second passage in the frozen setting is often off-topic (a sample A6 prompt for pneumonia symptoms
carried a Harrison passage on interstitial lung disease history after the correct MedlinePlus section), so the prompt is noisier
than the coverage numbers suggest. This is the frozen setting; a change would have to be logged before the model sees it.


## 17. The image was not reaching the model (found 10 Nov 2026)

**How it surfaced.** The first run of notebook 04 (`results/run_5_text_only/`) gave identical answers for all 130 different
chest X-rays: `open` said "Pneumonia" 130 times, `forced` said "No Finding" 130 times, `perclass` ranked "Infiltration" first 130
times. The per-class scores differed between images only at the second or third decimal (standard deviation 0.002 to 0.009).
That is not a model that reads weakly; it is a model whose output hardly depends on the image.

**Cause (found by reading the code, then reproduced on a toy model).**
- `modeling_chexagent.py` (line 836) encodes the image only `if past_key_values is None and torch.any(input_ids == img_start_id)`.
- The notebooks install transformers 4.46.3. From 4.45 on, `generate` creates an empty `DynamicCache` and passes it on the first
  step for models that declare `_supports_cache_class` (CheXagent does). `past_key_values` is therefore not `None`, the condition
  is false, and the image encoder is skipped. The image file path stays in the prompt as plain text, which is why scores vary
  slightly between images.
- Toy check (`/tmp` script, not kept): a small model with the same condition, run with transformers 4.46.3, takes the image
  branch with `use_cache=False` and skips it with `use_cache=True`. The real model has not yet been tested for this; the new
  IMAGE CHECK cell does that on the first Colab run.
- The 4.40.0 version the model was written for passes `None`, which is why the model card example works.

**What this affects.** Every model run so far used `use_cache=True` with transformers 4.46.3: pilot run 1, run 2, run 3
(knowledge and side sets), run 4 (flip test, side prompt variants, heart size) and the first notebook-04 run. They were all
produced by a model that could not see the image.
- Withdrawn: the claims that the model cannot read left/right, that the flip test showed it does not follow orientation, that it
  never says "enlarged" when it should, the 77 to 79% versus 90% category gaps as evidence about vision, and the Phase 1c
  readings. They describe what the model answers from the question text alone.
- Still valid as stated: the data pipeline, the scorer, the test sets, the constant-answer floor, and the observation that
  knowledge questions score at the floor *for a model answering from text alone*. The earlier runs are now a **text-only
  control**, which is useful: it is what a system that cannot see the image scores.
- Phase 2 retrieval check (entries 14 and 15) used no model and is unaffected.

**Fix.**
- `patch_image_cache_check` (`baseline/model.py`, and the same code in the notebook model cell) makes the condition also true for an
  empty cache. Idempotent; raises if the model code is not as expected. Tests: `tests/test_model_patch.py`.
- Every notebook now has an IMAGE CHECK cell right after the model cells. It writes black, white and noise images to the same file
  path (so the prompt text is identical) and raises an error if the first-token probabilities do not change by at least 0.01.
- Notebook 05's token-budget check now counts the image slots (it used text only before).
- Amendment 6 relabels earlier results and fixes the rerun order. No pre-registered rule or threshold changed.

**Lesson recorded.** The earlier "model gives a constant answer whatever the image" findings should have prompted this check. A
sanity test that the image changes the output belongs before any perception experiment, not after.

**Not yet confirmed:** that the patched model uses the image well; the IMAGE CHECK only proves it uses it at all. Whether disease
naming, laterality and the rest improve is for the reruns to show.

## 18. First Colab run with the image branch active: dtype mismatch in the visual tower (13 Nov 2026)

**What happened.** The first run of a fixed notebook stopped in the IMAGE CHECK with `RuntimeError: Input type
(torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`, raised inside `modeling_visual.py`
(`encode`, then the SigLIP patch-embedding convolution). The traceback shows `visual.encode` being called, so the image branch
of entry 17 is now reached: the cache patch works. This error had never appeared before because the image branch never ran.

**Cause (read from `modeling_visual.py`, reproduced on CPU with a toy tower).** `CLIPModel.encode` casts the pixels to the dtype
of `next(self.parameters())`. The tower registers a float32 position-embedding parameter on itself, which comes first in
`parameters()`, while the SigLIP weights are float16 (the model is loaded with `torch_dtype=float16` on a T4). The pixels are
therefore float32 and the first convolution fails.

**Fix.** `align_visual_dtype(model, dtype)` (`baseline/model.py` and the model cell of notebook 01, reused by 02 to 05) casts the
whole visual tower to the model dtype and raises if the tower is not found or its dtype is still mixed. All five notebooks were
rebuilt. Tests: `tests/test_model_patch.py` (reproduces the failure on a toy tower, then checks the fix). 123 tests pass.

**Not yet confirmed.** The fix is verified on a toy tower only. The first real confirmation is the IMAGE CHECK passing on Colab.
Further problems after the visual tower (for example in how image features are merged into the prompt) cannot be excluded until
that run completes. No result from a run with the image is recorded yet.

**Order change (logged as promised in `docs/next_steps.md`).** Notebook 03 is run before 04 and 05 so that the perception
result is available for the presentation. No rule, threshold or scorer changes; Amendment 6, point 4 is the only text affected.

## 19. First model results with the image visible: perception diagnostics (13 Nov 2026)

Notebook 03 completed on Colab T4 after the entry-18 fix (788 generations; `results/run_6_with_image/`; analysis in
`docs/phase1c_findings_with_image.md`). Against the text-only control on the same questions: mirrored X-ray changes the left/right
answer in 78% of pairs (3% before); left/right accuracy on 60 questions 90% (45%); heart enlargement sensitivity 17/22 with 25/25
specificity (9/22 and 22/25), accuracy 89% against 53% for always-no. By the readings fixed in advance the model does use the image for
both properties; the earlier claims that it cannot read them are dropped (Amendment 6). The 78% swap rate is just under the 80% line of
the registered table and is reported as such. The deck (slide 7) and the report (section 5.5) now show these results; the bug slide was
removed from the deck and the bug stays documented in the report (section 5.4) and in entries 17 and 18. Not run with the image yet: 01, 02, 04, 05.
