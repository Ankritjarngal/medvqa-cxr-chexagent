# Medical Visual Question Answering for Chest X-ray Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ankritjarngal/medvqa-cxr-chexagent/blob/main/notebooks/01_baseline_chexagent.ipynb)

A research project investigating how well AI can answer natural-language questions about
chest X-rays, using a pretrained vision-language model (VLM) as a baseline and exploring
whether retrieval-augmented generation (RAG), fine-tuning, and/or uncertainty communication
can improve answer quality and reliability.

This is **not** a diagnostic tool and is intended for educational/research purposes only.

## Project roadmap

1. **Phase 1 — Baseline testing (current stage, see below):** run pretrained CheXagent-2-3b
   zero-shot on a chest-X-ray question set, broken down by question difficulty, to measure
   where it's actually weak.
2. **Phase 2 — RAG layer:** add knowledge-grounded retrieval (authoritative finding
   definitions/criteria, not just similar patients' reports) targeted at whatever Phase 1
   shows is weak, designed around RULE's documented RAG failure modes.
3. **Phase 3 — Fine-tuning (only if needed):** a targeted LoRA fine-tune, only if Phase 2's
   RAG layer doesn't close the gap on its own.

## Repository contents

```
docs/
  literature_review.md          # Deep-dive review of 6 papers, dataset catalogue, gap analysis
  papers/README.md               # Links to the 6 source papers (PDFs are not redistributed)
  walkthrough.md / .pdf          # Plain-language walkthrough of the literature review findings
baseline/
  categorize.py                  # Tags each question general / laterality / severity (GPU-free)
  data.py                        # Builds the chest-X-ray-only test set from SLAKE + VQA-RAD (GPU-free)
  evaluate.py                    # Exact-match / loose-match scoring + category breakdown (GPU-free)
  model.py                       # CheXagent-2-3b inference wrapper (needs GPU + torch/transformers)
  run_baseline.py                 # CLI entry point tying the above together
tests/                           # Unit tests for the GPU-free parts of baseline/
notebooks/
  01_baseline_chexagent.ipynb     # Self-contained Phase 1 notebook -- upload straight to Colab
scripts/
  build_slides.py                 # Regenerates the slide deck (python-pptx)
  build_baseline_notebook.py      # Regenerates notebooks/01_baseline_chexagent.ipynb
slides/
  Medical_VQA_CXR_Literature_Review.pptx   # 5-slide presentation deck for the supervisor meeting
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

## Pilot run 1 (60 questions) and findings

A first 60-question pilot on Colab gave 62% overall (general 70%, laterality 45%, severity 70%), with
wide confidence intervals, and showed the model under-calls findings (sensitivity 33% on yes/no
questions). Details, caveats and the test-set problems it exposed:
[`docs/pilot_run_1_findings.md`](docs/pilot_run_1_findings.md). Raw results are in
`results/pilot_run_1/`.

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

### Regenerating the notebook or the test harness

```bash
pip install nbformat
python3 scripts/build_baseline_notebook.py   # regenerates notebooks/01_baseline_chexagent.ipynb
python3 -m pytest tests/                      # runs unit tests for the GPU-free logic
```

## Regenerating the slide deck

```bash
pip install python-pptx
python3 scripts/build_slides.py
```

Output is written to `slides/Medical_VQA_CXR_Literature_Review.pptx`.

## License and attribution

Code in this repository is released under the [MIT License](LICENSE).

Third-party assets are fetched at runtime and are **not** redistributed here:

- [CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) — MIT license (Stanford AIMI).
- [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) — CC-BY-4.0; please cite Liu et al., ISBI 2021.
- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) — CC0-1.0 (Lau et al., 2018).

## Disclaimer

This project is for research and education only. It is not a medical device and must not
be used for clinical decision-making.
