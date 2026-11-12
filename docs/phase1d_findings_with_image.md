# Phase 1d findings with the image visible: disease naming (run 7)

Run on Colab T4 with `notebooks/04_disease_naming.ipynb` on 13 Nov 2026, full run, 130 SLAKE chest X-rays with a disease label
(train 90, validation 20, test 20; eight diseases). Raw data: `results/run_7_disease_with_image/`. Reproduce with
`python3 scripts/analyze_disease.py results/run_7_disease_with_image/disease_results.csv`. Readings and the method choice rule
were fixed in advance (`docs/phase2_rag_plan.md`, section 3). Earlier disease-naming output (`results/run_5_text_only/`) is not
evidence about the model's reading of images.

Plain-language summary: **the model names the disease from the X-ray well above chance with all three ways of asking, so step (a)
of the retrieval design is feasible. The method used downstream is yes/no per disease.**

| method | top-choice accuracy (95% CI) | majority label | shuffled labels | unparsed |
|---|---|---|---|---|
| open question | 56% (48-64%) | 22% | 13% | 2 |
| forced choice (15 names) | 36% (28-45%) | 22% | 11% | 0 |
| yes/no per disease | 52% (44-61%) | 22% | 12% | 0 |

All three are above chance by both pre-registered criteria (p < 0.001 against the majority label; above the 95th percentile of the
shuffled-label control).

**Method used downstream (`CHOSEN_METHOD`):** `perclass`, because it has the best accuracy on the 20 validation images
(55%, against 45% for forced and 30% for open). The rule was fixed before the numbers existed. The open question has the best
overall accuracy (56%) but only 30% on the validation images.

Per-disease AUC for the yes/no scores: pneumothorax 0.99, cardiomegaly 0.95, effusion 0.89, nodule 0.86, mass 0.86, pneumonia 0.82,
atelectasis 0.77, infiltration 0.76.

Weak spots: the yes/no method never predicts pneumonia (recall 0 of 17); the open question finds atelectasis in 1 of 21 images; forced
choice over-predicts atelectasis (51 times) and never predicts infiltration or pneumonia.

## Caveats

- 130 images, eight diseases, 20 validation images: the choice of method rests on a small sample and is noisy.
- SLAKE training images are probably in CheXagent's training data.
- Labels are SLAKE annotations (ChestX-ray14 names after merging spellings), not clinician-verified.
- Only naming is tested here. Whether retrieval adds anything is decided by notebook 05 (arms A1 to A6).

## What follows

Run notebook 05 with `DISEASE_METHOD = "perclass"`. The model-guess arms (A5, A6) are expected to show an effect only if naming
accuracy of about half carries through; the analysis reports A6 against A1 and A5 as fixed in Amendment 5.
