# Agent handoff briefing

Read this file first. It tells you what this project is, what is done, and what to do next.

## Project

Medical Visual Question Answering on chest X-rays. Three phases:

1. Phase 1: baseline test of the pretrained `StanfordAIMI/CheXagent-2-3b` model (zero-shot).
2. Phase 2: add a knowledge-grounded RAG layer aimed at whatever Phase 1 shows is weak.
3. Phase 3: targeted LoRA fine-tuning, only if Phase 2 is not enough.

Read `README.md` and `docs/literature_review.md` for background. Everything is research-only,
not for clinical use.

## Current state

- Phase 1 code is complete and committed: `baseline/`, `notebooks/01_baseline_chexagent.ipynb`,
  `tests/` (8 passing unit tests), CI workflow in `.github/workflows/tests.yml`.
- The data pipeline and scoring were tested against real SLAKE and VQA-RAD data. The model
  itself has NOT been run: the cloud sandbox has no GPU. The first real model run happens on
  Google Colab (free T4), so expect to fix small issues in `baseline/model.py` or the notebook.
- Repo prepared for GitHub: MIT license, attribution, paper PDFs untracked (links in
  `docs/papers/README.md`), Colab badge in the README.

## Task 1: push this repo to GitHub

Target: https://github.com/Ankritjarngal/medvqa-cxr-chexagent (exists, currently empty).

- A secret named `GITHUB_TOKEN` should be available as an environment variable. Check with
  `env | grep -c GITHUB_TOKEN` (never print the value).
- Add the remote and push the full history of `main`:
  ```bash
  git remote add github https://github.com/Ankritjarngal/medvqa-cxr-chexagent.git
  git push https://x-access-token:${GITHUB_TOKEN}@github.com/Ankritjarngal/medvqa-cxr-chexagent.git main
  ```
  Do not store the token in git config or in any file. Do not use `--force`.
- Verify with `git ls-remote https://github.com/Ankritjarngal/medvqa-cxr-chexagent.git`.
- If there is no token, or the push is rejected, stop and report the exact error. Do not try
  other forges or workarounds.
- Do not open pull requests unless asked.

## Task 2: after the push

- Confirm the "Open in Colab" badge URL in `README.md` matches the GitHub repo path.
- Check that the GitHub Actions run for the unit tests passes, if the token allows reading it.
- Do not run model inference here; there is no GPU. Phase 1 results come from the user running
  the notebook on Colab. Wait for the user to share the category breakdown
  (general / laterality / severity accuracy).

## Notes for working with the user

- Keep answers plain-language and short. The user is presenting this to a supervisor and wants
  to say the project is feasible for free (Colab T4, CheXagent 3.1B fits in about 8 GB VRAM).
- Slides are uniform: no "new paper" tags. Dataset recommendation stars stay.
- Known limitation to state honestly: CheXagent's training data (CheXinstruct) already includes
  VQA-RAD, SLAKE and Open-I, so the Phase 1 test is close to in-distribution, not clean zero-shot.
