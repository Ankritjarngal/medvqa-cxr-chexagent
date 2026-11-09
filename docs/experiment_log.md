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
