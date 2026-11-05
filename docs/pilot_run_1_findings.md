# Pilot run 1: CheXagent-2-3b zero-shot on 60 chest X-ray questions

Raw results: `results/pilot_run_1/baseline_results.csv` (Colab free T4, float16, `transformers==4.46.3`).
This was a small pilot. Numbers below are descriptive; most between-bucket differences are **not**
statistically established (see confidence intervals).

## Test set

60 questions, 20 per keyword-assigned category (general / laterality / severity), from SLAKE
(chest X-ray rows, metadata-filtered, test split) and VQA-RAD (keyword-filtered chest rows).
CheXagent's training data (CheXinstruct) already includes both datasets, so this is close to
in-distribution, not a clean held-out test.

## Results

| Category | Correct | Accuracy | 95% CI |
|---|---|---|---|
| general | 14 / 20 | 70% | 48-85% |
| laterality | 9 / 20 | 45% | 26-66% |
| severity | 14 / 20 | 70% | 48-85% |
| overall | 37 / 60 | 61.7% | 49-73% |

Loose-match scoring was identical to exact-match on every row.

## What the data supports

1. **Laterality is the lowest bucket**, in the direction CheXagent's paper reports, but the intervals
   overlap with the other categories, so this is suggestive, not proven. The cleanest "which side"
   questions (choose left / right / both) went 1 of 4.
2. **The model under-calls findings.** On the 33 yes/no questions it answered "no" 27 times; the correct
   answer was "no" 21 times. Sensitivity (correct when the finding is present) was 33% (4 / 12);
   specificity was 90% (19 / 21). On heart-size questions it answered "yes" once across 16 questions
   in which "yes" was correct 5 times. Its 70% yes/no accuracy is barely above the 64% obtained by
   always answering "no".
3. **Knowledge-type open questions** (symptoms, treatment, prevention) were mostly wrong (3 of 4 in the
   pilot), but these are 4 questions.
4. **Strict scoring is harsh on open answers**: "Chest tightness, shortness of breath" vs gold
   "Chest tightness, fatigue" scored 0. The token-overlap score added in the next version gives
   partial credit.

## Known problems with the pilot test set (fixed in the next version)

- 4 of the 20 "severity" questions were SLAKE's "How much damage will be caused to the body taking such
  an image?" (a radiation-template question, not clinical severity). The model got all 4 right,
  which inflated that bucket.
- Only 47 unique question texts out of 60; "is the heart enlarged?" appeared about a dozen times.
- Some VQA-RAD items may not be chest X-rays (the keyword filter is a heuristic), for example
  "what is the vascular distribution of these infarcts?". Images were not reviewed.
- Question IDs were derived from signed image URLs, so they were not stable across runs.

## Implication for the plan

Retrieval can plausibly help with **knowledge** errors. It cannot fix **perception** errors (the "no"
bias, mixing up sides), because those happen in what the model sees. Phase 2 should therefore be scoped
to knowledge questions and the evaluation should report perception errors separately. Persistent
perception errors point towards confidence scoring or targeted fine-tuning.
