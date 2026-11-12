# Medical Visual Question Answering for Chest X-ray Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ankritjarngal/medvqa-cxr-chexagent/blob/main/notebooks/01_baseline_chexagent.ipynb)

A research project investigating how well AI can answer natural-language questions about
chest X-rays, using a pretrained vision-language model (VLM) as a baseline and exploring
whether retrieval-augmented generation (RAG), fine-tuning, and/or uncertainty communication
can improve answer quality and reliability.

This is **not** a diagnostic tool and is intended for educational/research purposes only.

## Project roadmap

1. **Phase 1 — Baseline testing (run; results are a text-only control, see the correction below):** run pretrained CheXagent-2-3b
   zero-shot on a chest-X-ray question set, broken down by question difficulty, to measure
   where it's actually weak.
2. **Phase 2 — RAG layer (built, retrieval check done, model runs pending):** add knowledge-grounded retrieval (authoritative finding
   definitions/criteria, not just similar patients' reports) targeted at whatever Phase 1
   shows is weak, designed around RULE's documented RAG failure modes.
3. **Phase 3 — Fine-tuning (only if needed):** a targeted LoRA fine-tune, only if Phase 2's
   RAG layer doesn't close the gap on its own.

The ordered next steps, the decision table and the parts not yet designed are in [`docs/next_steps.md`](docs/next_steps.md).

## Repository contents

```
docs/
  experiment_log.md              # Chronological record: what was done, found, fixed and corrected
  evaluation_protocol.md         # How Phase 2 will be judged (fixed before any retrieval result)
  phase1b_test_sets.md           # Knowledge and side test sets: design, caveats, reference numbers
  run_3_findings.md              # Results of the model on the knowledge and side sets
  phase1c_diagnostics.md         # Perception diagnostics (flip test, prompt variants, heart size): plan and readings
  phase1c_findings.md            # Results of those diagnostics (run 4)
  phase2_rag_plan.md             # Phase 2 RAG: pre-registered design, frozen settings, how to run
  baseline_run_2_findings.md     # 224-question baseline analysis (pilot: pilot_run_1_findings.md)
  literature_review.md          # Deep-dive review of 6 papers, dataset catalogue, gap analysis
  papers/README.md               # Links to the 6 source papers (PDFs are not redistributed)
  walkthrough.md / .pdf          # Plain-language walkthrough of the literature review findings
baseline/
  categorize.py                  # Tags each question general / laterality / severity (GPU-free)
  data.py                        # Builds the chest-X-ray-only test set from SLAKE + VQA-RAD (GPU-free)
  evaluate.py                    # Exact-match / loose-match scoring + category breakdown (GPU-free)
  model.py                       # CheXagent-2-3b inference wrapper (needs GPU + torch/transformers)
  run_baseline.py                 # CLI entry point tying the above together (--set baseline|knowledge|side)
  extra_sets.py                  # Knowledge (130 q) and left/right-balanced side (140 q) test sets (GPU-free)
  sides.py                       # Side extraction, confusion table, right-default test (GPU-free)
  knowledge.py                   # Item-level scoring for knowledge answers, blind review sheets (GPU-free)
  compare.py, stats.py           # Paired baseline-vs-candidate comparison, exact tests (GPU-free)
results/
  pilot_run_1/, run_2/, run_3/   # Raw per-question results of each model run
tests/                           # Unit tests for the GPU-free parts of baseline/
notebooks/
  01_baseline_chexagent.ipynb     # Self-contained Phase 1 notebook -- upload straight to Colab
  02_knowledge_and_side_baselines.ipynb  # Phase 1b: knowledge + side sets (same model, same settings)
  03_perception_diagnostics.ipynb        # Phase 1c: flip test, side prompt variants, heart-size prompts
  04_disease_naming.ipynb                # Phase 1d / Phase 2 Stage 1: can the model name the disease?
  05_rag_knowledge_arms.ipynb            # Phase 2 Stage 3: no-retrieval, oracle and RAG arms on the knowledge questions
scripts/
  make_figures.py                 # Chart, table and annotated-X-ray figures (matplotlib)
  render_diagrams.py              # Renders the Mermaid block diagrams (deliverables/diagrams/*.mmd)
  build_deck.py                   # Builds the 12-slide project presentation (python-pptx)
  build_slides.py                 # Old 5-slide literature-review deck (superseded)
  build_baseline_notebook.py      # Regenerates notebooks/01_baseline_chexagent.ipynb
  build_extra_notebook.py         # Regenerates notebooks/02_knowledge_and_side_baselines.ipynb
  build_diagnostics_notebook.py   # Regenerates notebooks/03_perception_diagnostics.ipynb
  build_disease_notebook.py       # Regenerates notebooks/04_disease_naming.ipynb
  build_rag_notebook.py           # Regenerates notebooks/05_rag_knowledge_arms.ipynb
  retrieval_check.py              # Phase 2 Stage 2: does the corpus hold the gold items? (no model)
  router_check.py                 # Phase 2 Stage 4: which baseline questions reach retrieval
  analyze_rag.py, analyze_disease.py, analyze_diagnostics.py   # Re-run reports from uploaded result files
  stub_run_notebook.py            # Run a notebook's cells here with a stand-in model
slides/
  Medical_VQA_CXR_Literature_Review.pptx      # Old 5-slide deck (superseded)
deliverables/
  presentation/                   # Minimalist 12-slide project presentation (.pptx and .pdf)
  diagrams/                       # Mermaid sources (.mmd) of the block diagrams, plus config
  report/                         # LaTeX project report (black and white): main.tex, main.pdf; figures/ = colour (slides), figures_bw/ = grey (report)
```

## Literature review

Six papers were reviewed to map out the Medical VQA landscape before deciding between a
fine-tuning vs. RAG improvement strategy:

| # | Paper | Found by |
|---|---|---|
| 1 | PubMedCLIP (EACL Findings 2023) | Teammate |
| 2 | LLaVA-Med (2023) | Teammate |
| 3 | Med-Flamingo (2023) | Teammate |
| 4 | CheXagent (2024) | This review |
| 5 | PMC-VQA / MedVInT (2023) | This review |
| 6 | RULE (EMNLP 2024) | This review |

See [`docs/literature_review.md`](docs/literature_review.md) for the full deep dive, a
cross-paper comparison table, a dataset catalogue (what's available, how datasets differ,
which to prefer for a chest-X-ray-only project), and the specific gaps/loopholes this
project can realistically target. The slide deck in `slides/` summarizes the same findings
for a presentation.

### Headline recommendation coming out of the review

- **Baseline:** [CheXagent](https://arxiv.org/abs/2401.12208) — an open-source, chest-X-ray-specific
  3.1B-parameter foundation model — rather than a general-purpose medical VLM, since it already
  matches this project's exact scope and is far cheaper to run than Med-Flamingo (9B) or LLaVA-Med (7B).
- **Improvement direction:** knowledge-grounded RAG (retrieving authoritative finding
  definitions/guidelines rather than similar patients' case reports, as in RULE), explicitly
  designed around RULE's documented failure modes (naive RAG causes ~47-59% of its own errors
  via over-reliance on retrieved text).
- **Evaluation:** combine accuracy on a chest-filtered VQA-RAD/SLAKE subset + a small
  hand-built Open-I-based set with a qualitative rubric for relevance and unsupported/misleading
  claims — not just automated text-overlap metrics, which Med-Flamingo shows can diverge sharply
  from real clinical usefulness.

> **Correction, 10 Nov 2026: every model result in this repository so far (runs 1 to 4 and the first notebook-04 run) was
> produced without the model seeing the image.** With transformers 4.46.3 the model's own code skipped its image encoder.
> The results are a valid *text-only control*; conclusions about what CheXagent can or cannot read from an X-ray are
> withdrawn until everything is rerun with the fixed notebooks (which now stop with an error if the image does not reach the
> model). Details: [`docs/experiment_log.md`](docs/experiment_log.md) entry 17 and Amendment 6 in
> [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).
>
> **Update, 13 Nov 2026:** the perception diagnostics (notebook 03) have been rerun with the image visible. Mirroring the X-ray
> changes the left/right answer in 78% of pairs (3% before) and the model finds 17 of 22 enlarged hearts; see
> [`docs/phase1c_findings_with_image.md`](docs/phase1c_findings_with_image.md). The other notebooks are still to be rerun.

## Pilot run 1 (60 questions) and findings

A first 60-question pilot on Colab gave 62% overall (general 70%, laterality 45%, severity 70%), with
wide confidence intervals, and showed the model under-calls findings (sensitivity 33% on yes/no
questions). Details, caveats and the test-set problems it exposed:
[`docs/pilot_run_1_findings.md`](docs/pilot_run_1_findings.md). Raw results are in
`results/pilot_run_1/`.

**Run 2 (224 questions, improved test set):** 57% overall; the model never answered "yes" on 8
enlargement questions where "yes" was correct; laterality is lower than general but partly
confounded by dataset; knowledge-type questions are mostly missed. See
[`docs/baseline_run_2_findings.md`](docs/baseline_run_2_findings.md) and `results/run_2/`.

## Running Phase 1 (baseline testing)

The easiest way: open **`notebooks/01_baseline_chexagent.ipynb`** directly in Google Colab
(Runtime → Change runtime type → **T4 GPU**, then Run all). It is fully self-contained —
no need to clone this repo into Colab — and will:

1. Build a chest-X-ray-only test set (up to 100 questions per category, roughly 220 in total,
   because severity questions are rare) from SLAKE (metadata-filtered) and VQA-RAD
   (keyword-filtered), stratified across `general` / `laterality` / `severity`. The non-clinical
   radiation-damage template question is dropped and repeated questions are capped.
2. Download and run `StanfordAIMI/CheXagent-2-3b` zero-shot on every question.
3. Score predictions (exact, loose, token-F1, each with a 95% confidence interval) and print
   breakdowns by category, answer format, knowledge-vs-perception, plus yes/no sensitivity and
   specificity against an always-answer-"no" baseline. These decide what Phase 2 should target.
4. Show images for the wrong answers (laterality first) for a manual spot-check; automated
   metrics alone are not trustworthy, and the VQA-RAD chest filter is a heuristic.
5. Save `baseline_results.csv` / `.jsonl` for the write-up.

To run the same pipeline from the command line instead (e.g. a Colab terminal, or any
machine with a CUDA GPU):

```bash
pip install -r requirements-baseline.txt
python -m baseline.run_baseline --n-per-category 100
```

Add `--dry-run` to build and inspect the test set without a GPU (useful for checking the
data pipeline on a laptop before spending GPU time). Outputs land in `data/`.

**Note on what this does and doesn't measure:** CheXagent's own training corpus
(CheXinstruct) already includes VQA-RAD, SLAKE, and Open-I among its 32 source datasets,
so this is not a clean held-out generalization test — it's close to in-distribution, same
as every other paper in our review that reuses these benchmarks. The goal is to get our
own measured numbers on the known laterality/severity weak spot, not to claim a novel
accuracy figure.

### Phase 1b: knowledge and side questions

Run 2 left two questions open, so two supplementary test sets were built and are run with the same model
and settings: **130 knowledge questions** (symptoms, treatment, prevention, cause, function, "why"; the
reference for Phase 2 retrieval) and **140 side questions with equal numbers of left and right answers** (does
the model default to "right"?). Run `notebooks/02_knowledge_and_side_baselines.ipynb` on Colab. Design and
caveats: [`docs/phase1b_test_sets.md`](docs/phase1b_test_sets.md). How Phase 2 will be judged:
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md). Full history:
[`docs/experiment_log.md`](docs/experiment_log.md).

**Run 3 result:** on explicit "which side" questions the model answers "Right" about 90% of the time whatever the
image; on knowledge questions (outside one trivial subtype) it scores no better than always writing the most common
answer. Details: [`docs/run_3_findings.md`](docs/run_3_findings.md).

### Phase 1c: perception diagnostics

Three zero-training tests of why the model fails on left/right and heart size: a mirror-image test (does the
answer change when the image is flipped?), six wordings of the side question, and four wordings of the
heart-enlargement question. Run `notebooks/03_perception_diagnostics.ipynb` on a Colab T4. What each outcome will mean
was written before running: [`docs/phase1c_diagnostics.md`](docs/phase1c_diagnostics.md).

**Run 4 result:** mirroring the image changed the left/right answer in only 2 of 73 cases (3%), and no wording of the
question fixed it; the wording sets the default answer. For heart enlargement the wording moves the yes-rate far more
than the image does, with one weak lead. Details: [`docs/phase1c_findings.md`](docs/phase1c_findings.md).

### Phase 2: knowledge-grounded RAG

Built from what the data shows: SLAKE's knowledge answers are fixed by the disease on the image (8 chest diseases), and
the question never names it. So the experiment separates (1) can the model name the disease (notebook 04), (2) does retrieved
text from MedlinePlus and medical textbooks hold the answer once the disease is known (done without a model: 33% of gold
items with the true disease, 13% with a wrong one, 4% from the question alone), and (3) does the full system beat a
style-matched baseline and the disease guess alone (notebook 05). Design, frozen settings and the decision rule were written
before any model run: [`docs/phase2_rag_plan.md`](docs/phase2_rag_plan.md), Amendment 5 in
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md). Not yet run on the model.

### Regenerating the notebook or the test harness

```bash
pip install nbformat
python3 scripts/build_baseline_notebook.py   # regenerates notebooks/01_baseline_chexagent.ipynb
python3 scripts/build_extra_notebook.py      # regenerates notebooks/02_knowledge_and_side_baselines.ipynb (run after 01)
python3 scripts/build_diagnostics_notebook.py # regenerates notebooks/03_perception_diagnostics.ipynb (run after 01)
python3 scripts/build_disease_notebook.py     # regenerates notebooks/04_disease_naming.ipynb (run after 01)
python3 scripts/build_rag_notebook.py         # regenerates notebooks/05_rag_knowledge_arms.ipynb (run after 01)
python3 -m pytest tests/                      # runs unit tests for the GPU-free logic
```

## Regenerating the presentation and the report

```bash
pip install python-pptx matplotlib pandas pillow
npm install @mermaid-js/mermaid-cli  # needs Chrome; set MMDC=node_modules/.bin/mmdc
MMDC=node_modules/.bin/mmdc python3 scripts/render_diagrams.py   # block diagrams (colour, for slides)
python3 scripts/make_figures.py      # charts and photo (colour, for slides)
FIG_BW=1 MMDC=node_modules/.bin/mmdc python3 scripts/render_diagrams.py && FIG_BW=1 python3 scripts/make_figures.py   # grey set for the report
python3 scripts/build_deck.py        # deliverables/presentation/...pptx
cd deliverables/report && latexmk -pdf main.tex   # makes main.pdf (needs a TeX Live install)
```

The title slide and title page hold placeholders for supervisor, team and institution until the official format is supplied.

## License and attribution

Code in this repository is released under the [MIT License](LICENSE).

Third-party assets are fetched at runtime and are **not** redistributed here:

- [CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) — MIT license (Stanford AIMI).
- [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) — CC-BY-4.0; please cite Liu et al., ISBI 2021.
- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) — CC0-1.0 (Lau et al., 2018).

## Disclaimer

This project is for research and education only. It is not a medical device and must not
be used for clinical decision-making.
