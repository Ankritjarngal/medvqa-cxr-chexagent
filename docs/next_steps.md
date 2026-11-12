# What happens next (single source of truth)

Written 12 Nov 2026. This file puts the plan for the work after the image-fix in one place, says where each step is already
specified, and says plainly which parts are **not yet designed**. Nothing here changes a frozen rule: the arms, thresholds,
scorer, retrieval setting and decision rule are as fixed in `docs/phase2_rag_plan.md` and `docs/evaluation_protocol.md`
(Amendments 1 to 6).

## Where things stand

| Phase | What | State |
|---|---|---|
| 0 | Literature review, six papers, gap analysis | Done (`docs/literature_review.md`) |
| 1, 1b, 1c | Zero-shot baseline, knowledge and side sets, perception diagnostics | Run, **text-only control** (image did not reach the model) |
| 1d | Disease naming on 130 images | First run exposed the bug (`docs/experiment_log.md`, entry 17) |
| Fix | `patch_image_cache_check` plus IMAGE CHECK cell in every notebook | Done, unit tested; **not yet confirmed on the real model on Colab** |
| 2 | Knowledge-grounded RAG: corpus, BM25, six arms, analysis scripts | Built; retrieval check and router check done (no model); **model runs pending** |
| 3 | Targeted fine-tuning / second model | Not designed (see "Not yet designed") |

## Ordered plan

Run order is fixed by Amendment 6, point 4 of `docs/evaluation_protocol.md`: 04, 05, then 01, 02, 03.

| # | Step | Who / where | Inputs and outputs | Pass / decision | Specified in |
|---|---|---|---|---|---|
| 1 | Run notebook 04 (disease naming), `QUICK = True` then `False`. The IMAGE CHECK must pass first. | You, Colab T4 | out: `disease_results.csv`, `disease_checkpoint.json`, printed `CHOSEN_METHOD` | Disease naming above the majority baseline (22%) and the shuffled-label control; otherwise perception is the bottleneck and A5/A6 are expected to show little | `docs/phase2_rag_plan.md` sections 3, 7, 11 |
| 2 | Analyse step 1: `python3 scripts/analyze_disease.py disease_results.csv` | Here | in: the CSV; out: top-choice accuracy per method, per-class AUC | Method with the best validation accuracy is used downstream | same, section 3 |
| 3 | Run notebook 05 (arms A1 to A6) with `DISEASE_METHOD` set to the chosen method | You, Colab T4 | out: `rag_results.csv`, `rag_settings.json`, `rag_checkpoint.json` | none yet | section 4, 11 |
| 4 | `python3 scripts/analyze_rag.py rag_results.csv --make-sheets ratings/` | Here | out: three shuffled rating sheets | none | section 6, 11 |
| 5 | Blind rating: fill `rating_1` / `rating_2` with 0, 1 or 2 | You (a student, not a clinician) | out: filled sheets | none | `baseline/knowledge.py` (`RATING_RUBRIC`) |
| 6 | `python3 scripts/analyze_rag.py rag_results.csv --ratings ratings/` | Here | out: PASS / FAIL / PENDING for the five conditions of Amendment 5 | A6 must beat A1 **and** A5 with the stated intervals and sign test; "no demonstrated improvement" is a valid outcome | `docs/evaluation_protocol.md`, Amendment 5 |
| 7 | Rerun notebooks 01, 02, 03 (baseline, knowledge and side sets, perception diagnostics) | You, Colab T4 | out: new `results/` folders; analysis scripts exist (`analyze_diagnostics.py`, and the baseline analysis in the notebooks) | The text-only control and the constant-answer floor become reference points: a model that sees the image should beat both | Amendment 6; `docs/phase1c_diagnostics.md` |
| 8 | Write the results up, replacing the text-only numbers where the image run exists, keeping the text-only runs as a labelled control | Here | report, slides, `docs/experiment_log.md` | none | `docs/experiment_log.md` conventions |

## What the outcome of steps 1 to 7 decides

| Result | Next move | Where stated |
|---|---|---|
| Model cannot name diseases above chance with the image | Perception is the bottleneck. A6 is expected to be near A1. Move to Phase 3 or test a second model; do not claim a retrieval result | `phase2_rag_plan.md` section 7 |
| A3 above A1, A4 not above A3 | Knowing the disease helps, the retrieved text adds nothing the model did not know | same |
| A4 above A3 | Retrieval helps once the disease is known (upper bound only) | same |
| A6 above A1 and A5, rule met | A bounded gain on SLAKE template questions; report with the stated caveats | same, and Amendment 5 |
| A6 above A1 but not above A5 | The gain comes from naming the disease, not retrieval; not reported as a retrieval gain | same |
| A6 not above A1 | No demonstrated improvement (valid result) | same |
| Rerun of 03 shows no image dependence for left/right and heart size, and no wording fixes it | Phase 3 (targeted fine-tuning on laterality and enlargement) moves up, or a different model is evaluated | `docs/phase1c_diagnostics.md`, "What would follow" |
| Rerun of 03 shows the model does use the image | The earlier left/right and heart-size claims are dropped; the diagnostics are reported as the new baseline | Amendment 6 |

## Not yet designed (must be written and frozen before any run)

These are mentioned in several documents as options but have no plan, test set, or decision rule. Each needs a short
pre-registration (in the style of `docs/phase2_rag_plan.md`) **before** it is run.

1. **Phase 3, targeted fine-tuning.** The README and handoff say "targeted LoRA fine-tune, only if needed". Missing: training data
   source (CheXinstruct is the named candidate), held-out split by image, hyperparameters, compute plan for a free T4, and a
   success rule. Trigger: step 1 or step 7 shows a perception gap that retrieval cannot address.
2. **A second model on the same four test sets** (baseline, knowledge, side, heart). Missing: which model, and why. Listed as an
   alternative to Phase 3 in `docs/experiment_log.md` (end of entry 12) and `docs/phase1c_findings.md`.
3. **A larger heart-size set** to test the weak `plain` wording lead once, with the wording named in advance
   (`docs/phase1c_findings.md`, "Decisions to take next", point 2).
4. **Stage 5 of the Phase 2 plan** (image-based case retrieval; perception-targeted retrieval for heart size). Marked "optional,
   not started" in `docs/phase2_rag_plan.md` section 2, with no design.
5. **An independent retrieval corpus check and a clinician spot-check.** `docs/evaluation_protocol.md` states that any claim
   beyond "the answers read as better" needs a clinician to review a sample; no clinician is arranged.
6. **Uncertainty / abstention layer**, named as a stretch direction in `docs/literature_review.md`. No design.

## Known limits that apply to every step

- Zero-shot only; one model; small test sets (50 oracle questions, 130 disease images, 20 validation images).
- CheXagent's training data includes SLAKE and VQA-RAD; the tests are not clean held-out evaluation.
- The rater is a student. No clinician reviewed any output. Nothing here supports a claim of clinical validity or safety.
- The image fix is unit tested and reproduced on a toy model; the first Colab run is its first test on the real model. If that
  run fails the IMAGE CHECK, stop and diagnose before any other step.
