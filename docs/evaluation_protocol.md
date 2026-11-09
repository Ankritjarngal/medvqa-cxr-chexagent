# Evaluation protocol for Phase 2 (retrieval) and later phases

Written before any retrieval result exists, so the criteria cannot be adjusted to fit the outcome.
If something here has to change, change it in `docs/experiment_log.md` with the reason and the date,
before looking at the affected results.

## What is being tested

Whether adding retrieved knowledge to CheXagent-2-3b's prompt improves answers to
**knowledge-type** questions (symptoms, treatment, prevention, cause, function, "why") without
harming **perception** questions (what is visible in the image).

Baseline run 2 and the sets below show where the model is weak. Retrieval can only plausibly help the
knowledge part. It cannot be expected to fix errors in what the model sees (it answered "no" on all
8 enlargement questions whose answer was "yes"). Those are tracked, not targeted, in Phase 2.

## Fixed reference sets

| Set | File | Questions | Role |
|---|---|---|---|
| Baseline | `results/run_2/baseline_results.csv` minus `results/run_2/excluded_ids.txt` | 219 | Guard against harm on perception questions; reference numbers already exist |
| Knowledge | built by `baseline/extra_sets.py` (notebook 02) | 130 | Primary test for retrieval |
| Side | built by `baseline/extra_sets.py` (notebook 02) | 140 | Tracks the side-bias question; not a Phase 2 target |

Each set is run once without retrieval (the baseline for that set) and once with retrieval. Same model,
same greedy decoding, same prompt wording except for the added retrieved text. Questions are matched
by their stable `id`.

## Rules that protect the result

1. **Dev and held-out split.** `split_dev_heldout()` in `baseline/extra_sets.py` splits the knowledge set
   by image (dev about 30%, held-out about 70%; 42 / 88 questions). What is retrieved, how many passages,
   and the prompt wording may be tuned on the dev part only. The held-out part is scored once, after the
   design is frozen. The frozen design is recorded in the experiment log first.
2. **Retrieval source independence.** SLAKE's gold answers come from its own knowledge graph. The
   retrieval corpus must not include SLAKE's knowledge graph, SLAKE answer text, or VQA-RAD answers.
   Record the corpus, its licence and how it was cleaned in the experiment log. Otherwise the test checks
   whether the system can look up its own answer key.
3. **No test-time access to gold.** Retrieval queries are built from the question and from the model's own
   output only. Not from `gold_answer`, not from SLAKE disease annotations.
4. **Both orders reported.** Every result is shown for train rows and test/validation rows separately
   (CheXagent probably saw the SLAKE train rows). A claim needs the same direction in both.
5. **All planned comparisons are reported**, including the ones that did not work.

## Metrics

| Question type | Primary metric | Also report |
|---|---|---|
| Knowledge | Item recall (`baseline/knowledge.py`), plus blind manual rating | Item F1, token F1, exact match, per-subtype |
| Perception (baseline set) | Exact match | Token F1, yes/no sensitivity and specificity |
| Side | Side correct (`baseline/sides.py`) | Confusion table, asymmetry test |

Item recall is a screening score: it gives credit for covering items of the gold list after a short,
readable synonym table (`CONCEPT_SYNONYMS`). It is more lenient than exact match and says nothing about
clinical correctness. The **blind manual rating** is what decides the knowledge question: baseline and
retrieval answers are shuffled into `answer_1` / `answer_2`, rated 0 / 1 / 2 with the rubric in
`baseline/knowledge.py` (`RATING_RUBRIC`), and unblinded afterwards. The reviewer is a student, not a
clinician. If a claim goes beyond "the answers read as better", a clinician should spot-check a sample.

## Tests

- **Paired binary outcomes** (exact match, side correct): exact McNemar test on questions where only one
  system was right (`baseline/compare.py`).
- **Paired scores** (item recall, token F1): mean paired difference with a 95% bootstrap interval that
  resamples whole images, because several questions share an image and are not independent.
- **Manual ratings:** wins / losses / ties with an exact sign test.
- Group comparisons inside a run (for example laterality vs general) use Fisher's exact test.
  Implementations are in `baseline/stats.py` and checked against known values in `tests/test_stats.py`.

## Decision rule for Phase 2

Retrieval is reported as helping on knowledge questions only if **all** of these hold on the held-out part:

1. Item recall improves, with a 95% interval that excludes zero.
2. The blind manual rating favours retrieval (sign test p < 0.05).
3. The direction is the same on train rows and on test/validation rows.
4. On the 219-question baseline set, retrieval does not do significantly worse
   (McNemar p >= 0.05 against retrieval being worse) and does not lose more than 3 points of exact match.

If 1 and 2 hold but 3 fails, report "improvement on rows the model probably saw, not shown elsewhere".
If 1 or 2 fail, report no demonstrated improvement. Either is a valid result for the write-up.

## How much can these sets detect

For a paired binary comparison, the exact McNemar test needs this many wins out of the questions on which
the two systems disagree (`m`) to reach p < 0.05:

| Disagreements m | Wins needed (win : loss) |
|---|---|
| 6 | 6 : 0 |
| 10 | 9 : 1 |
| 20 | 15 : 5 |
| 30 | 21 : 9 |
| 40 | 27 : 13 |
| 60 | 39 : 21 |

Held-out knowledge has 88 questions from 72 images, 35 distinct question wordings and 26 distinct gold
answers. If most answers do not change between systems, m will be small and only a
clear improvement will register. Set-wide differences of a few points will not be detectable, and the
held-out test/validation rows alone (28) can only confirm a direction, not prove an effect.

## What this protocol will not support

- A claim that retrieval improves "Medical VQA" in general. The knowledge set is a handful of SLAKE
  templates plus a few VQA-RAD questions.
- Any claim about clean zero-shot generalisation. The model has very probably seen SLAKE and VQA-RAD.
- Any clinical claim. This is research code and none of it is validated for clinical use.

## Commands

```bash
# build sets, run on Colab: notebooks/02_knowledge_and_side_baselines.ipynb
python -m baseline.compare knowledge_baseline.csv knowledge_rag.csv --metric item_recall --by source_split
python -m baseline.compare run2.csv rag_on_baseline_set.csv --metric exact_match --by category \
    --exclude-ids results/run_2/excluded_ids.txt
```
