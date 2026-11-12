# Phase 1c findings with the image visible (run 6)

Run on Colab T4 with `notebooks/03_perception_diagnostics.ipynb` on 13 Nov 2026 (788 generations, full run), after the
image-branch patch and the visual-tower dtype fix (`docs/experiment_log.md`, entries 17 and 18). The notebook stops with an error
if its IMAGE CHECK fails, so a completed run means the pixels changed the output. Raw data: `results/run_6_with_image/`.
Reproduce every number with `python3 scripts/analyze_diagnostics.py results/run_6_with_image/diagnostics_results.csv`.
The text-only control for the same questions is `results/run_4/` (`docs/phase1c_findings.md`, withdrawn as a perception result).
The readings applied are the ones fixed in advance in `docs/phase1c_diagnostics.md`.

Plain-language summary: **with the image visible, the model's left/right answers follow the image and "is the heart enlarged?"
is answered well above the always-no rate. The earlier conclusion that the model cannot read these properties came from the
image never reaching it, and is dropped.**

## A. Flip test

| | image skipped (run 4) | image visible (run 6) |
|---|---|---|
| pairs answered left/right on both images (of 120) | 73 | 99 |
| answer changed side when mirrored | 2 (3%) | 77 (78%, 95% CI 69-85%) |
| byte-identical answer on both images | 112 / 120 | 22 / 120 |
| correct on original | 34% | 82% |
| correct on mirrored (against mirrored gold) | 29% | 68% |

Explicit "which side" questions (29): swap rate 74% (20 of 27, CI 55-87%); correct 86% on the original and 66% on the mirror.

Reading: the swap rate is just under the 80% line of the pre-registered table, with accuracy clearly above 50% on both images.
The closest row is "the model does read laterality". The row was written for 80% or more; 78% with an interval reaching 85% does
not separate the two cases, and the registered table has no row for it. The drop from 82% to 68% on the mirrored image means the
model is less accurate on mirrored X-rays, which are unusual images (the heart appears on the right).

## B. Side prompt wordings (60 questions)

Accuracy by wording: plain 90%, one_word 95%, describe_first 90%, suffix 88%, image_side 85%, patient_side 82% (text-only: 33% to 55%).
Share answering "right" is 47-53% for every wording, so the answers are not a constant. For every wording the answer depends on the true
side (Fisher p below 0.001). No wording fix is needed.

## C. Heart size (47 questions, 22 enlarged)

Plain wording: sensitivity 17/22 (77%), specificity 25/25 (100%), accuracy 89% against 53% for always answering "no", Youden J 0.77
(text-only: 9/22, 22/25, J 0.29). Other wordings: describe_first J 0.82, normal_or_enlarged J 0.68. The rule "sensitivity 50% or more and
J of 0.3 or more" is met by every wording, so the failure seen in the text-only run was an artefact of the missing image, not of wording.

## Caveats

- CheXagent's training data includes SLAKE and VQA-RAD, so these are not clean held-out tests; part of the result may be training exposure.
- Small samples: 60 side questions, 47 heart questions (22 positive); the heart sensitivity has a wide interval.
- Only the perception notebook (03) has been rerun with the image. Notebooks 01, 02, 04 and 05 are still to be rerun.
- No clinician has reviewed anything.

## What follows

Phase 3 (fine-tuning for laterality and enlargement) loses its motivation: the model already follows the image for both properties.
Next runs: notebook 04 (disease naming), then 05, 01 and 02, as in `docs/next_steps.md`.
