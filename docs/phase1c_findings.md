# Phase 1c findings: perception diagnostics (run 4)

> **CORRECTION (10 Nov 2026): the image did not reach the model in this run.** With transformers 4.46.3 the model
> code skipped the image encoder, so every answer here came from the question text alone (`docs/experiment_log.md`,
> entry 17; Amendment 6 in `docs/evaluation_protocol.md`). The numbers below are a valid **text-only control** (what the model
> answers without seeing the image). Statements about what the model can or cannot read from the X-ray do not follow from them
> and are withdrawn until the rerun with the fixed notebooks.

Run on Colab T4 with `notebooks/03_perception_diagnostics.ipynb` on 10 Nov 2026 (788 generations, full run, not
`QUICK`). Raw data: `results/run_4/diagnostics_results.csv` and `diagnostics_checkpoint.json`. Reproduce every number
below with `python3 scripts/analyze_diagnostics.py results/run_4/diagnostics_results.csv`. The readings applied are the
ones fixed in advance in `docs/phase1c_diagnostics.md`.

Plain-language summary: **the model's left/right answers do not depend on the image, and the answer you get is decided
by the wording of the question. No prompt wording fixed this. For "is the heart enlarged?", one wording gives a weak
signal that is not strong enough to count as a fix under the rule set in advance.**

## A. Flip test: left/right does not follow the image

120 questions with a left or right correct answer (29 explicit "which side" questions), each asked on the original
image and on its mirror.

| | all 120 | explicit "which side" (29) |
|---|---|---|
| answered left/right on both images | 73 | 28 |
| answer changed side when mirrored | 2 (3%, 95% CI 1-9%) | 0 (0%, CI 0-12%) |
| byte-identical answer on both images | 112 / 120 | 29 / 29 |
| share answering "right" | 38% orig, 40% mirrored | 90%, 90% |
| correct on original | 34% | 48% |
| correct on mirrored (against mirrored gold) | 29% | 48% |

- A model that reads laterality would swap close to 100%. This one swapped 2 times in 73.
- The 8 answers that did differ between the two images were mostly "Upper lung" becoming "Upper left lung", not a side
  swap.
- Repeatability: 20 of 20 repeated questions gave identical text, so the identical answers are not a noise artefact.
- Among the 73 answers that named a side on the original image, 41 (56%) were correct. This matches run 3 (56%).
- 38 of 120 answers named no side (for example "Upper lung").

Reading under the pre-stated table: swap rate under 20% with the interval's upper end under 35%, so **the model's
left/right answer does not depend on the image's orientation**.

Limit: this shows the answer is unaffected by mirroring. It does not show the image is ignored in general (the model
does get some other questions right). One possible alternative explanation: if the model had been trained with random
mirroring of images, its answers would be unchanged by mirroring even if it used the image. That would be a training
choice we cannot check from here; either way, the practical result is the same, since left/right answers do not track
the true side.

## B. Side wording: the wording sets the default, none of them fixes it

60 side questions (29 explicit, 31 others), six wordings. Accuracy is the share of all 60 answered with the correct side.

| variant | named a side | said "right" | correct when truth is left | correct when truth is right | accuracy | p, answer depends on truth |
|---|---|---|---|---|---|---|
| suffix (run 3 wording) | 78% | 60% | 21% | 68% | 45% | 0.51 |
| plain | 82% | 58% | 24% | 65% | 45% | 0.76 |
| one_word | 100% | 13% | 93% | 19% | 55% | 0.26 |
| describe_first | 85% | 62% | 24% | 68% | 47% | 0.76 |
| patient_side | 72% | 23% | 45% | 23% | 33% | 1.00 |
| image_side | 88% | 45% | 45% | 48% | 47% | 0.79 |

Explicit questions only (29): `suffix`, `plain` and `describe_first` all say "right" 90% of the time; `one_word` says
"right" 24% (correct on 86% of left-truth questions, 33% of right-truth, accuracy 59%, p = 0.39); `patient_side` 41% right, p = 1.0.

- The default flips with the wording. With "Answer with exactly one word: Left or Right." the model answers "Left" 52
  times and "Right" 8 times out of 60. With the run 3 wording it is mostly "Right". Neither default follows the truth.
- No variant reached the pre-stated bar of p < 0.01 with accuracy above 50%. `one_word` is the closest (55% overall,
  but p = 0.26, and its answers are Left for the majority of questions, so it is a different default, not a fix).
- Reading: **no wording fixes it**. This matches test A.

## C. Heart size: the wording moves the yes-rate; one wording shows a weak signal

47 questions where "yes" means enlarged (22 yes, 25 no; 40 VQA-RAD, 7 SLAKE).

| variant | sensitivity | specificity | accuracy (always-"no" = 53%) | said yes | Youden J |
|---|---|---|---|---|---|
| run2_suffix | 1/22 (5%) | 25/25 (100%) | 55% | 2% | 0.05 |
| plain | 9/22 (41%) | 22/25 (88%) | 66% | 26% | 0.29 |
| describe_first | 22/22 (100%) | 3/25 (12%) | 53% | 94% | 0.12 |
| normal_or_enlarged | 0/22 (0%) | 25/25 (100%) | 53% | 0% | 0.00 |

- Run 2's wording almost never says "enlarged" (1 of 22), as in run 2 (0 of 8).
- `describe_first` did not describe anything: the answers were the single word "Yes". It says yes to 94% of questions,
  so its 100% sensitivity is a bias shift, not skill (J = 0.12; Fisher p for "depends on truth" = 0.24).
- `normal_or_enlarged` always says normal.
- `plain` (the question with no instruction attached) is the only variant that responds to the truth: 9 of 22 enlarged
  hearts were called enlarged against 3 of 25 normal ones (Fisher p = 0.042). But this does not clear the bar set in
  advance (sensitivity 41% is under 50% and J = 0.29 is under 0.3). Accuracy 66% (31 / 47) against the 53%
  always-"no" baseline has p = 0.08, and `plain` against `run2_suffix` on the same questions has McNemar p = 0.23
  (8 gained, 3 lost). Because `plain` was the best of four wordings, its p = 0.042 is not corrected for that choice and
  would not survive a correction for 4 comparisons.
- Exploratory only (tiny groups): `plain` found 4 of 15 enlarged hearts in VQA-RAD but 5 of 7 in SLAKE.

Reading: **partial**. The wording changes how often the model says "enlarged" far more than the image does, but with
`plain` there may be a small image-dependent signal. This needs a larger set to confirm or drop.

## What this means for the project

| Question | Answer from these tests |
|---|---|
| Is retrieval the right next step for left/right and heart size? | No. The information is not in the answer the model produces; retrieved text cannot supply which side a finding is on. |
| Is there a cheap prompt fix for left/right? | Not among the six wordings tried. |
| Is there a cheap prompt fix for heart size? | Not by the rule set in advance; `plain` is a weak lead that needs more data. |
| What moves up? | A perception fix: targeted fine-tuning on laterality and enlargement (Phase 3), or testing a different model. Retrieval remains relevant for the knowledge questions, with the caveat that run 3 found the model at the constant-answer floor there. |

Together with run 3 this makes a consistent picture: for properties that have to be read from the image, CheXagent-2-3b
on these questions behaves as if it answers from the question wording plus a default; for general knowledge it is not
better than a fixed answer.

## Limits

- Small samples: 29 explicit side questions, 22 enlarged hearts, and the mirror changes the anatomy as well as the sides.
- Heart-size questions come mostly from VQA-RAD, whose labels are noisy, and train rows were probably seen in training.
- Six side wordings and four heart wordings is a small hand-picked set, not a search. Not finding a fix is not proof none
  exists.
- The model is tested on its own training datasets' questions; behaviour on real clinical questions is not measured here.
- Nothing here is clinical evidence.

## Decisions to take next (proposed, not yet made)

1. Decide between Phase 3 (fine-tune for laterality / heart size) and trying a second model on the same four test sets
   (baseline, knowledge, side, heart).
2. If heart size is pursued, build a larger enlarged / not-enlarged set (the `plain` lead would then be tested once,
   with the wording named in advance).
3. Still open from run 3: blind manual rating of the knowledge answers, and a retrieval corpus independent of SLAKE's
   knowledge graph.
