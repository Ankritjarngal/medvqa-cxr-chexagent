# Medical Visual Question Answering for Chest X-ray Analysis

A research project investigating how well AI can answer natural-language questions about
chest X-rays, using a pretrained vision-language model (VLM) as a baseline and exploring
whether retrieval-augmented generation (RAG), fine-tuning, and/or uncertainty communication
can improve answer quality and reliability.

This is **not** a diagnostic tool and is intended for educational/research purposes only.

## Repository contents

```
docs/
  literature_review.md         # Deep-dive review of 6 papers, dataset catalogue, gap analysis
  papers/                       # The 6 source PDFs (3 found by teammate, 3 found in this review)
scripts/
  build_slides.py                # Regenerates the slide deck (python-pptx)
slides/
  Medical_VQA_CXR_Literature_Review.pptx   # 5-slide presentation deck for the supervisor meeting
```

## Current stage: Literature review (Week 1)

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

## Regenerating the slide deck

```bash
pip install python-pptx
python3 scripts/build_slides.py
```

Output is written to `slides/Medical_VQA_CXR_Literature_Review.pptx`.
