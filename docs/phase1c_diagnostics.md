# Phase 1c: perception diagnostics

Written **before the notebook has been run on the model**, so the reading of each outcome cannot be fitted to the
result. Notebook: `notebooks/03_perception_diagnostics.ipynb` (built by `scripts/build_diagnostics_notebook.py`).
Logic and tests: `baseline/diagnostics.py`, `tests/test_diagnostics.py`. Protocol link: Amendment 4 in
`docs/evaluation_protocol.md`.

## Why

Run 3 (`docs/run_3_findings.md`) found two perception faults in CheXagent-2-3b:

- On explicit "which side" questions it answers "Right" 26 times in 29 (90%), unrelated to the correct side.
- It answered "no" to every heart-enlargement question whose answer was "yes" (0 of 8 in run 2).

Retrieval cannot fix either, because the missing information is in the image. Before deciding the next phase, these
tests ask whether the faults come from the model not seeing the information, or from how it is asked. All of them use
the same model with no training, on the free T4.

## The three tests

### A. Flip test (does the model read left/right from the image?)

Each of the 120 left-or-right-gold side questions (29 of them explicit "which side" questions) is asked on the original
image and on its left-right mirror, with the same prompt run 3 used. The mirror reverses which side everything is on,
so a model that reads laterality should change its answer from left to right or the reverse.

Reported: among questions answered left/right on both images, the **swap rate** (answer changed side) with a 95% Wilson
interval; how many answers were byte-identical on both images; the share answering "right" on each; accuracy on the
original and on the mirror (against the mirrored gold side). Also a repeatability check (20 questions asked twice), so
that run-to-run noise is visible.

Caveat: mirroring makes an anatomically unusual image (heart on the right). A model that infers side from heart position
will still swap; that counts as using the image.

### B. Side prompt variants (is "Right" a wording effect?)

60 side questions (all 29 explicit ones plus 31 random others, 31 gold-right and 29 gold-left), six wordings:
`suffix` (run 3's prompt), `plain`, `one_word` ("Answer with exactly one word: Left or Right."), `describe_first`,
`patient_side` (states that left and right are the patient's), `image_side` (states that they are the picture's).

Reported per variant: share naming a side, share answering "right", accuracy on gold-left and gold-right questions, and
`depends on truth p` (Fisher's exact test of whether the answer is related to the true side).

### C. Heart size

47 yes/no questions where "yes" means the heart is enlarged ("Is the heart enlarged?", "Is there cardiomegaly?",
"Can cardiomegaly be observed in this picture?"), 22 with answer "yes" and 25 with "no" (40 from VQA-RAD, 7 from SLAKE).
Reversed wordings ("Is the size normal?"), other structures (hilum, vessels) and location-specific cardiomegaly questions
are excluded. Four wordings: `run2_suffix`, `plain`, `describe_first` (describe the heart size, then finish with
"Answer: yes/no"), `normal_or_enlarged` (one-word choice, mapped to yes/no).

Reported per variant: sensitivity, specificity, accuracy against the always-"no" baseline, share answering yes,
unparsed answers (counted as wrong) and **Youden's J** (sensitivity + specificity - 1, which is about 0 for a model that
ignores the image, including one that says "yes" to everything).

## How each outcome will be read (fixed in advance)

### Test A

| Result | Reading |
|---|---|
| Swap rate under 20% (upper end of its interval under 35%) | The model's left/right answer does not depend on the image's orientation. It is not reading laterality with this prompt. |
| Swap rate 80% or more, accuracy near 50% on both original and mirror | The model responds to orientation but does not map it to the dataset's left/right. A convention mismatch or poor calibration; read with test B's convention variants. |
| Swap rate 80% or more, accuracy clearly above 50% on both | The model does read laterality. The constant "Right" in run 3 would then be a wording or format effect on explicit questions. |
| Anything in between | Mixed. Reported as mixed; no conclusion drawn beyond the numbers. |

### Test B

With 5 variants compared against `suffix`, a variant counts as a **real fix** only if its answers depend on the true side
at p < 0.01 (0.05 / 5) **and** accuracy overall is above 50%. A variant that only lowers the "right" rate toward 50%
without making answers depend on the truth has changed the default, not fixed the problem. With 60 questions, accuracy
differences under about 15 points are not detectable.

### Test C

| Result | Reading |
|---|---|
| A variant with sensitivity 50% or more and Youden J of 0.3 or more | The failure is largely wording or format; a cheap prompt fix exists. |
| Sensitivity above 0 but J under 0.3 | Partial: the model can say "enlarged" sometimes, but not in a way that tracks the image. |
| Sensitivity near 0 for every variant | The model does not express enlargement under these wordings. A perception-level fix would be needed. |

Baseline for comparison: run 2 had sensitivity 0 of 8 on a smaller set of such questions.

## What would follow

| Outcome | Next step |
|---|---|
| A shows no image dependence, B finds no fix, C finds no fix | The model does not use the image for these properties. Phase 3 (targeted fine-tuning on laterality and enlargement) moves up, or a different model is evaluated. Retrieval would not address it. |
| A shows image dependence and B or C finds a prompt fix | A prompting change goes into the pipeline; it is tested on the held-out sets with the paired comparison in the protocol. |
| A shows orientation response with chance accuracy | Investigate the dataset's left/right convention before drawing any conclusion about the model. |

## Limits of these tests

- Small samples: 29 explicit side questions, 22 enlarged hearts. Intervals will be wide and are printed.
- VQA-RAD labels are noisy, and most of the heart questions come from it. Train rows were probably seen by the model.
- Prompt variants are a small, hand-picked set, not a search. A failure to find a fix is not proof none exists.
- Run-to-run noise is checked with 20 repeats only.
- Nothing here is clinical evidence.

## Running it

Open the notebook in Colab (T4 GPU). `QUICK = True` in the build cell gives a few-minute smoke test; set it back to
`False` for the real run (about 700 generations). Download `diagnostics_results.csv` and
`diagnostics_checkpoint.json` and upload them to the project. The analysis will use the repo code on the CSV.
