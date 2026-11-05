# Baseline run 2: CheXagent-2-3b zero-shot on 224 chest X-ray questions

Raw results: `results/run_2/baseline_results.csv` (Colab free T4, float16, `transformers==4.46.3`,
test set built by the improved pipeline: 100 laterality, 100 general, 24 severity; 119 SLAKE, 105 VQA-RAD).
CheXagent's training data (CheXinstruct) already includes SLAKE and VQA-RAD, so this is close to
in-distribution, not a clean held-out test. Images were not manually reviewed for this write-up.

## Headline numbers

| Bucket | Correct | Accuracy | 95% CI |
|---|---|---|---|
| overall | 128 / 224 | 57.1% | 51-63% |
| general | 70 / 100 | 70.0% | 60-78% |
| laterality | 44 / 100 | 44.0% | 35-54% |
| severity (24 questions, 22 yes/no) | 14 / 24 | 58.3% | 39-76% |
| yes/no questions | 74 / 108 | 68.5% | 59-77% |
| open-ended questions | 54 / 116 | 46.6% | 38-56% |
| knowledge-type questions (symptoms, treatment, ...) | 4 / 24 | 16.7% | 7-36% |

SLAKE 64.7%, VQA-RAD 48.6%. Token-overlap F1 is higher than exact match on open answers (0.54 vs 0.47).

## Findings, with how much weight each can bear

1. **The model never says "yes" to enlargement/size questions.** On the 22 severity yes/no questions
   (almost all "is the heart enlarged?" variants) it answered "no" every time; the correct answer was "yes"
   in 8 of them. If its true sensitivity were 50%, 0 of 8 would happen about 0.4% of the time. This
   reproduces the pilot run and is the most solid result so far. It is specific to these questions: on
   general yes/no questions sensitivity was 92%.
2. **Laterality is lower than general, but the comparison is partly confounded.**
   - Overall 44% vs 70% (Fisher p = 0.0003). It survives some controls: yes/no only 57% vs 82% (p = 0.018);
     excluding knowledge and disease-naming questions 53% vs 70% (p = 0.02).
   - But it is not seen in both datasets. Within SLAKE: general 83% vs laterality 43% (SLAKE's general
     bucket contains 24 trivial template questions, such as organ name, colour and counts, answered 100%
     correctly). Within VQA-RAD: general 46% vs laterality 46% (no difference; VQA-RAD is hard throughout).
   - So "fine-grained side questions are harder" is supported in SLAKE yes/no questions (95% vs 56%,
     p = 0.005) but not demonstrated in VQA-RAD. Treat it as likely, not settled.
   - Only 13 open questions ask the model to choose a side (left/right/both). It answered "right" on 11
     of them, but "right" was correct for only 7. When the correct side was left or both (6 questions),
     it answered "right" 5 times. Suggestive of defaulting to "right", but n is far too small to claim it.
3. **Knowledge questions are mostly missed, with canned answers.** 4 of 24 exact; the model repeated
   "Medical treatment, surgical treatment" 8 times, "Chest tightness, shortness of breath" 6 times and
   "Keep healthy" for prevention questions regardless of image. SLAKE's gold answers are annotator-written
   lists ("Medical therapy, supportive therapy"), so exact match is a poor judge here; any retrieval
   experiment on these needs a semantic or human judge.
4. **Naming a finding**: 2 of 11 correct. The model answered "Pneumonia" or "Effusion" for most. Small n.
5. **Dataset effects are large**: SLAKE yes/no laterality questions show the model saying "yes" 3 times
   where "yes" was correct 10 times (under-calling); VQA-RAD laterality yes/no is balanced (12 vs 11).

## Implication for the plan

- A retrieval layer can plausibly help the knowledge-type errors (finding 3). It cannot fix finding 1
  (never saying "enlarged"), which is a perception problem.
- For finding 1, candidate approaches are confidence/uncertainty reporting or a small targeted
  fine-tune on enlargement and laterality cases, evaluated against the 0 of 8 baseline.
- Before Phase 2 comparisons, review the images behind wrong answers to confirm they are chest X-rays.
