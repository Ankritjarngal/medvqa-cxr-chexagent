# Phase 1b: knowledge and side test sets

Why these exist, how they were built, what they contain, what is wrong with them, and what was
found while building them. Code: `baseline/extra_sets.py`, `baseline/sides.py`,
`baseline/knowledge.py`. Notebook to run on Colab: `notebooks/02_knowledge_and_side_baselines.ipynb`
(`python -m baseline.run_baseline --set knowledge|side` does the same from a terminal).

**Status:** the sets are built and the model has been run on both (run 3, 10 Nov 2026). Results and analysis:
`docs/run_3_findings.md`. The section "Reference numbers from run 2" below is the earlier 224-question run, re-scored
with the new code, kept for comparison. The reading table at the end was written before run 3; run 3 matched its
first row for explicit side questions only (see the findings note).

## Why a second round of baseline testing

Run 2 (224 questions) left two things unresolved:

1. **Knowledge questions: only 24.** Too few to detect a retrieval improvement (see
   `docs/evaluation_protocol.md`, "How much can these sets detect").
2. **Side bias: only 13 questions.** The model answered "right" on 11 of them, but 13 is too few to say
   anything. The "laterality" tag is also assigned from the *question text*, so a question like "Where is
   the abnormality in this image?" with the answer "Lower Left Lung" was tagged `general` and never
   counted as a side question.

## Knowledge set

**Selection.** Every question marked knowledge by the keyword rules in `baseline/categorize.py`,
from SLAKE test, validation and train (chest X-ray rows) and VQA-RAD (chest-keyword rows), de-duplicated on
(source, image, question text). Version 2 of the rules adds `why` and `how can you tell` to the original
symptoms / treatment / prevention / cause / function / how-do-you-determine patterns.

**Composition (130 questions).**

| | |
|---|---|
| Source | SLAKE 124, VQA-RAD 6 |
| Split | train 84, validation 22, test 24 |
| Subtype | treatment 30, cause 25, function 25, prevention 23, symptoms 22, explanation 5 |
| Distinct images / question wordings / gold answers | 103 / 42 / 33 |

**What this means for interpretation.**

- The questions are template questions ("How to treat the disease located on the left of lung in this
  image?"). 33 distinct gold answers across 130 questions means the answer is nearly the same for many
  images. A model that always writes the most common generic answer will score well on those, whether or
  not it looked at the image. Results should be read per subtype, not only as one number.
- The questions depend on which disease is in the image. A retriever given only the question text cannot
  know the disease. Phase 2 has to decide where the disease name comes from (for example the model's own
  description of the image), without using gold annotations.
- Train rows (84) are very probably in CheXagent's training data. They are kept because the set would
  otherwise be too small, and every result is split by `source_split`.
- A dev / held-out split by image (42 / 88) is provided for tuning retrieval without touching the final test.

**Prompt.** The plain question, for every row. Run 2 appended "Answer concisely based only on what is
visible in the image" to rows tagged laterality or severity, which covered 20 of its 24 knowledge
questions. That suffix is the wrong instruction for a question answered from knowledge, so run 2's knowledge
answers are not directly comparable to this set's. The 24 questions shared with run 2 (same `id`s) allow a
direct check of the prompt effect with `baseline/compare.py`.

## Side set

**Selection.** Open (not yes/no, not knowledge) questions whose gold answer is at most 4 words and names a
side: `Right`, `Lower Left Lung`, `Both Lung`, `bilateral lungs`. Side is extracted by `side_of()` in
`baseline/sides.py`. At most 8 questions with identical wording per side. Sampled at random with a fixed
seed to **60 left, 60 right, 20 both**, so that guessing one side cannot score well.

**Composition (140 questions).**

| | |
|---|---|
| Gold side | left 60, right 60, both 20 |
| Source | SLAKE 111, VQA-RAD 29 |
| Split | train 87, validation 19, test 34 |
| Distinct images / question wordings / gold answers | 106 / 67 / 27 |

**Scoring.** A prediction is correct on side when `side_of(prediction) == side_of(gold)`. "Left lung" is
correct for "Lower Left Lung". Exact match, loose match and token F1 are still reported but understate
performance here. The report gives the confusion table (gold side by predicted side, including "no side
named"), the rate at which gold-left questions get "right" versus gold-right questions get "left"
(Fisher's exact test for the asymmetry), and left-vs-right accuracy against 50% chance.

**Prompt.** The visible-in-image suffix for every row, as run 2 used for laterality questions.

**Limits.** "Left" and "right" in the gold answers are taken as written by the dataset annotators. The
datasets do not state whether they follow the patient's side or the side of the picture, and the model may
not follow the same convention. A consistent flip would show as accuracy well below 50% on left/right items,
and would be reported as such, not as a "right default".

## Data download

Both datasets are read from their own files on the Hugging Face Hub (SLAKE: `train.json`, `validation.json`,
`test.json`; VQA-RAD: two parquet files), about 5 downloads in total, with retries that honour the server's
Retry-After on HTTP 429. The first version paged through the `datasets-server` API (about 150 requests); on
Colab this returned `429 Too Many Requests` and stopped the notebook. The paging API remains as an automatic
fallback. VQA-RAD's parquet also contains the images, so there are no expiring signed URLs, and images are
identified by content hash, which lets questions about the same picture share one image identity. Checked:
the file path and the paging API return identical records (SLAKE test 358 / 358, VQA-RAD test 104 / 104), and
all 224 question ids of run 2 resolve to the same question and answer.

## Bug found while building these sets

Looking at the 30 VQA-RAD images in the side set showed one brain scan ("what two tributaries of blood
vessels are impacted?" with the answer "left ACA and MCA"). Cause: the VQA-RAD chest filter matched the
keyword `rib` anywhere inside a word, so `describe`, `circumscribed`, `distributions` and `tributaries`
all passed as chest questions. Brain, abdomen and other non-chest "describe the lesion" questions entered
the pool: 26 of the 432 VQA-RAD candidates entered only through this bug (most are not chest images; at least
one, a pulmonary-nodule description, probably is, and is now dropped because its question names no chest term).

Fix (`baseline/data.py`): `rib` / `ribs` match only as whole words. Other keywords stay substring matches
because the compound forms are real chest terms (`hemidiaphragm`, `pneumothorax`, `aortopulmonary`,
`pneumomediastinum`). A first idea, requiring every keyword to start a word, would have dropped those, and
was rejected after checking the 54 questions it would have changed. Regression tests are in
`tests/test_sampling.py`.

**Impact on run 2.** 5 of its 224 rows came in through this bug, all from VQA-RAD train, all answered wrong
(listed in `results/run_2/excluded_ids.txt`). Without them: overall 58.4% (was 57.1%); general 72.9% (70.0%);
laterality 44.0% (unchanged); severity 60.9% (58.3%); VQA-RAD 51.0% (48.6%). No conclusion changes. In
particular the enlargement result (0 of 8) is unaffected, since the dropped severity row was an open question.

Not done: a full image-by-image review of run 2's 105 VQA-RAD rows. Only the 30 side-set VQA-RAD images and
a random 12 of run 2's wrong answers (text only) were looked at. Other non-chest VQA-RAD items could still
be present. The filter remains a keyword heuristic.

## Knowledge scoring

Exact match scores a correct paraphrase as 0 and token F1 can reward unrelated shared words ("Aspiration is
more likely ... because it is more active" gets 0.27 against the correct anatomical reason). Added in
`baseline/knowledge.py`:

- **Item recall / precision / F1.** Gold and predicted answers are split on commas, semicolons, "and" and
  "or". A short hand-written synonym table maps variants to one spelling (`shortness of breath` to `dyspnea`,
  `surgical treatment` to `surgery`, `medical therapy` to `medication`, and a few more; the full table is
  in the file). A gold item counts as covered when its content words appear in the prediction. It is
  lenient by design and is not a clinical ontology. It works for list-style answers and is weak on
  full-sentence answers, where the manual rating matters more.
- **Blind manual review sheets** (`make_blind_review`, `summarize_blind_review`): two systems' answers shuffled
  into `answer_1` / `answer_2`, rated 0 / 1 / 2, unblinded afterwards, with an exact sign test.

## Reference numbers from run 2, re-scored with the new code

Knowledge questions in run 2, under version 2 of the knowledge rules (25 questions, not 24):

| Subtype | n | Item recall | Item F1 | Token F1 | At least one item right |
|---|---|---|---|---|---|
| all | 25 | 0.41 | 0.41 | 0.38 | 60% |
| treatment | 8 | 0.69 | 0.65 | 0.52 | 100% |
| symptoms | 6 | 0.46 | 0.51 | 0.38 | 83% |
| prevention | 4 | 0.00 | 0.00 | 0.19 | 0% |
| cause | 3 | 0.00 | 0.00 | 0.00 | 0% |
| explanation | 2 | 0.00 | 0.00 | 0.14 | 0% |
| function | 2 | 1.00 | 1.00 | 1.00 | 100% |

This changes how run 2 should be read. Exact match said 4 of 24 correct, which made the knowledge result look
like a broad failure. Item scoring shows the model is partly right on treatment and symptoms and fails
entirely on cause, prevention and the "why" questions (only 9 questions between them, so this is a lead). The treatment credit comes largely from one generic
answer ("Medical treatment, surgical treatment") that overlaps the gold lists, so it is not evidence that the
model reasoned about the individual image. This is the pattern the larger set is designed to separate.

Side questions in run 2 (23 open questions whose gold names a side, including ones tagged `general`):

| Gold side | n | Correct | Answered "right" |
|---|---|---|---|
| left | 6 | 2 | 4 |
| right | 11 | 6 | n/a |
| both | 6 | 3 | 2 |

Gold-left answered "right" 67%; gold-right answered "left" 14%; Fisher p = 0.10. The model named no side on 4
questions. This is a lead, not a result. The 140-question set is the test.

## What the results will mean once the model has run

| Result on the side set | Reading |
|---|---|
| Gold-left answered "right" much more often than gold-right answered "left" (asymmetry p < 0.05), accuracy near 50% on left/right items | Right-side default confirmed. Points to a perception fix (fine-tuning or a different prompt), not retrieval. |
| Accuracy clearly above 50% on left/right items, symmetric errors | Run 2's pattern was small-sample noise. |
| Accuracy clearly below 50%, errors symmetric | Likely a left/right convention mismatch with the dataset. Investigate before concluding anything. |
| Large share of "no side named" | Instruction-following problem; report separately from side errors. |

For the knowledge set, the question is how much of the score is generic answers versus image-specific ones.
Compare per-subtype item recall with the most common gold answer for that subtype (a model that always writes
that answer is the floor to beat).

## Files

| File | Purpose |
|---|---|
| `baseline/extra_sets.py` | Builds both sets, prompts per set, dev / held-out split |
| `baseline/sides.py` | `side_of`, side scoring and bias report |
| `baseline/knowledge.py` | Item-level scoring, blind review sheets |
| `baseline/compare.py`, `baseline/stats.py` | Paired comparison tools and exact tests |
| `notebooks/02_knowledge_and_side_baselines.ipynb` | Colab notebook; built by `scripts/build_extra_notebook.py` |
| `results/run_2/excluded_ids.txt` | The 5 run-2 rows admitted by the filter bug |
