# Run 3: CheXagent-2-3b on the knowledge (130) and side (140) test sets

Raw results: `results/run_3/knowledge_results.csv`, `results/run_3/side_results.csv`
(Colab free T4, float16, `transformers==4.46.3`, zero-shot, notebook 02).
Sets and their caveats: `docs/phase1b_test_sets.md`. Both sets include SLAKE train rows that the model very
probably saw; the split-by-split numbers below show no clear advantage on those rows, but the intervals are wide.
Knowledge answers use the plain question as the prompt; side answers use the "based only on what is visible in
the image" suffix. Knowledge scores use scorer version 2 (see "Scorer change" below).

## Side questions

| Bucket | Correct | Rate | 95% CI |
|---|---|---|---|
| Side correct, all | 52 / 140 | 37.1% | 30-45% |
| gold left | 14 / 60 | 23% | |
| gold right | 27 / 60 | 45% | |
| gold both | 11 / 20 | 55% | |
| Model named no side at all | 42 / 140 | 30% | 23-38% |
| Exact match, all | 23 / 140 | 16% | |

Findings, in order of how much they can be trusted:

1. **On explicit "which side" questions the model says "Right" almost regardless of the image.** 29 questions
   have a left or right gold answer and ask "which side" or "left or right". The model answered "right" on 26 of
   them (90%, 95% CI 74-96%): on 13 of 14 whose correct answer was left, and on 13 of 15 whose correct answer was
   right. The answer is unrelated to the truth (Fisher p = 1.0). Its 14 / 29 correct is simply the share of right
   answers in the set. It holds in SLAKE (20 of 20 "right") and mostly in VQA-RAD (6 of 9). This confirms the lead
   from run 2, which had only 13 such questions.
   Not excluded: the "answer concisely" prompt suffix may push the model toward a one-word default. A coordinate
   convention mismatch would show left and right swapped, not a constant answer.
2. **On other side questions ("Where is the abnormality?") there is no right default.** Among 45 left/right questions
   where the model named a side, gold-left was answered "right" 6 of 19 times and gold-right was answered "left" 12 of
   26 times (p = 0.37); 27 / 45 (60%) were correct.
3. **Overall, when the model commits to left or right it is right 41 / 73 times (56%, CI 45-67%).** That cannot be told
   apart from guessing (exact binomial p = 0.35), and it includes finding 1.
4. **30% of answers name no side.** The model answers with an up-down location ("Upper lung", "Lower lung area",
   "Bibasilar area") to questions whose gold answer is e.g. "Lower Left Lung". Where both gold and model gave an
   up-down location they agreed 24 / 36 times. This is partial information that side-only scoring does not credit.
5. **One answer is a template.** "Left Lung, Right" was given on all 19 questions worded "Where is/are the abnormality
   located...?", whatever the image. Scored as bilateral, it is correct on the 10 of those 19 whose gold answer is
   bilateral, because the dataset uses that wording for bilateral cases about half the time. That is where most
   of the 55% on bilateral questions comes from, so it is not evidence that the model saw bilateral disease.
6. Train rows 35 / 87 (40%) vs test and validation rows 17 / 53 (32%): not a clear difference.

What is **not** yet known: whether the model uses the image at all for left/right. The direct test is to flip the
image horizontally and see whether the answer changes (not run yet).

## Knowledge questions

Item recall is the share of gold answer items the prediction covers (scorer v2). The floor is the score of a system
that ignores the image and always writes the most common gold answer for that question type (leave-one-out, so a
question's own answer is never used). It is a generous floor, since it uses knowledge of the answer distribution; it
shows what template-copying achieves, not what is feasible.

| Subtype | n | Distinct gold answers | Model item recall | Constant floor |
|---|---|---|---|---|
| treatment | 30 | 5 | 0.78 | 0.77 |
| symptoms | 22 | 8 | 0.42 | 0.21 |
| prevention | 23 | 5 | 0.22 | 0.51 |
| cause | 25 | 9 | 0.03 | 0.03 |
| explanation ("why") | 5 | 5 | 0.00 | 0.00 |
| function | 25 | 1 | 1.00 | 1.00 |
| **all except function** | 105 | | **0.37 (CI 0.29-0.45, by image)** | 0.38 |
| all | 130 | | 0.49 | |

1. **`function` is not informative.** All 25 gold answers are "Breathe" and the model said "Breathe" every time. These
   25 free points inflate every overall number: exact match is 29 / 130 overall but 4 / 105 outside `function`.
2. **Outside `function`, the model does not beat the constant-answer floor** (0.37 vs 0.38). It is clearly above the
   floor on symptoms only (0.42 vs 0.21, n = 22). It matches the floor on treatment and cause, and is below it on
   prevention, where it writes "keeping healthy" (17 of 23) while the gold answers are mostly "Live healthy, enhance
   physical fitness" or similar.
3. **Answers barely change with the image.** Treatment: 5 distinct answers across 30 questions (18 identical);
   prevention: 4 distinct across 23 (17 identical); function: 1.
4. **Style.** With the plain prompt the model writes full sentences for symptoms and cause ("The main cause of the
   disease on the lower left of the lung is typically an infection, such as pneumonia..."). Item scoring is a loose
   judge of sentences; the blind manual rating in the protocol is the real check.
5. **Prompt effect.** On the 25 questions shared with run 2 (which had the "visible in the image" suffix), item recall
   went from 0.41 to 0.45 with the plain prompt (difference +0.04, 95% CI -0.04 to +0.12). 19 of the 25 answers changed
   wording. No detectable effect on accuracy.
6. Train rows 0.42 (n = 84), validation 0.51 (22), test 0.49 (24). No sign that the train rows are easier.
7. The 6 VQA-RAD questions all scored 0.

## Scorer change made after seeing these results

Scorer version 1 gave "keeping healthy" a score of 0 against the gold answer "Keep healthy". Version 2 adds light
suffix stripping (`keeping` to `keep`, `lungs` to `lung`) to both sides. Effect on this run: overall item recall
0.45 to 0.49, prevention 0.00 to 0.22; every other subtype unchanged. This was a change made after looking at results,
for a plain defect; it is recorded in `docs/experiment_log.md`. Run 2's knowledge numbers are identical under both
versions. From here on both systems in any comparison are scored with version 2.

## What this means for the plan

- **Side bias is real but narrow:** a constant "Right" on explicit side questions. That is a perception fault.
  Retrieval cannot fix it, because the missing information is in the image, not in a textbook.
- **The SLAKE knowledge questions leave little room to show a retrieval gain.** They are templates whose gold answers
  come from a fixed set of strings, and a system that copies the most common string already matches the model.
  Cause and symptom answers also depend on which disease is in the image, and in run 2 the model named the disease
  correctly 2 times in 11. A retriever fed the model's own disease guess would likely retrieve the wrong text
  (a hypothesis; not tested).
- So the larger, clearer, measurable failures are all on the perception side: enlargement (0 of 8 "yes"), explicit
  side questions (constant "Right"), disease naming (2 of 11). Phase 2 as first planned (retrieval for knowledge
  questions) has a narrow target on these benchmarks, and the protocol has been amended accordingly
  (`docs/evaluation_protocol.md`, Amendments).
