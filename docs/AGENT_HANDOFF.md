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

- **READ FIRST (10 Nov 2026): the image never reached the model in runs 1 to 4 or in the first notebook-04 run.** transformers
  4.46.3 passes an empty `DynamicCache` on the first generation step; CheXagent's code only encodes the image when
  `past_key_values is None`. Fixed by `patch_image_cache_check` (`baseline/model.py` and the notebook model cell); every notebook now
  has an "IMAGE CHECK" cell that raises if changing the pixels does not change the output. All earlier model results are text-only
  controls (banners on the findings docs, `results/run_5_text_only/`). Phase 1 and Phase 1c conclusions about perception are
  withdrawn until rerun in the order 04, 05, then 01, 02, 03 (Amendment 6, point 4). **Update 13 Nov 2026: notebook 03 was rerun with the image (after a second fix, entry 18) and the perception conclusions reversed: see `docs/phase1c_findings_with_image.md`, entry 19, `results/run_6_with_image/`. Notebooks 01, 02, 04, 05 are still to be rerun.** `docs/experiment_log.md` entry 17, Amendment 6.
  The ordered plan, decision table and the list of parts not yet designed are in `docs/next_steps.md`.

- Phase 1 code is complete and committed: `baseline/`, `notebooks/01_baseline_chexagent.ipynb`,
  `tests/` (123 passing unit tests), CI workflow in `.github/workflows/tests.yml`.
- Phase 1 has been run on Colab twice (pilot of 60 questions, then 224). Results are in `results/`,
  analysis in `docs/`. Read `docs/experiment_log.md` for the history and `docs/evaluation_protocol.md`
  for how Phase 2 must be evaluated. Phase 1b (knowledge and side test sets, notebook 02) is built and
  run on Colab (run 3, `docs/run_3_findings.md`). Phase 1c (perception diagnostics, notebook 03,
  `docs/phase1c_diagnostics.md`) was run on Colab (run 4, `docs/phase1c_findings.md`, data in `results/run_4/`): left/right answers do not follow the
  image and no wording fixed it; heart size shows one weak lead. Next decision: Phase 3 fine-tuning or a second model.
- Phase 2 (RAG) is built but NOT run on the model: `docs/phase2_rag_plan.md` (pre-registered; frozen retrieval setting in
  section 10; run order in section 11), `baseline/disease_labels.py`, `baseline/rag.py`, `baseline/rag_analysis.py`,
  notebooks 04 and 05, analysis scripts. Retrieval check and router check are done (`docs/experiment_log.md`, entries 13-16).
  Do not change the frozen settings or the scorer after seeing model results without logging the reason first. The user runs
  the notebooks on Colab and uploads the CSVs; the corpora are downloaded at run time into `rag_cache/` (git-ignored).
- The data pipeline and scoring were tested against real SLAKE and VQA-RAD data. The cloud sandbox has no GPU; the model is run on
  Google Colab (free T4) by the user, who uploads the result files. Runs 1 to 4 and the first notebook-04 run were text-only
  controls (see above); the only run with the image reaching the model so far is notebook 03 (run 6, 13 Nov 2026).
- Repo prepared for GitHub: MIT license, attribution, paper PDFs untracked (links in
  `docs/papers/README.md`), Colab badge in the README.

## Task 1: push this repo to GitHub (done; kept as reference for the push procedure)

Target: https://github.com/Ankritjarngal/medvqa-cxr-chexagent (exists, currently empty).

- The user connected GitHub to Cursor, and may also have saved a `GITHUB_TOKEN` secret.
  Try these in order and use the first that works:
  1. Add the remote and push with the existing credentials:
     ```bash
     git remote add github https://github.com/Ankritjarngal/medvqa-cxr-chexagent.git
     GIT_TERMINAL_PROMPT=0 git push github main
     ```
  2. If that asks for a login, and `env | grep -c GITHUB_TOKEN` is non-zero (never print the
     value), push with the token:
     ```bash
     git push https://x-access-token:${GITHUB_TOKEN}@github.com/Ankritjarngal/medvqa-cxr-chexagent.git main
     ```
  Do not store the token in git config or in any file. Do not use `--force`.
- UPDATE: GitHub `main` already contains this project's history (merged at `b1f850a`). Newer
  commits exist on the Cursor remote's `main`. First run `git fetch` for both remotes; if GitHub's
  `main` is an ancestor of local `main` (`git merge-base --is-ancestor`), skip the merge below and
  just push local `main` normally (fast-forward, no force). Always run `python -m pytest tests/`
  before pushing.
- Older situation (already resolved, keep for reference): the GitHub repo was NOT empty. It has one auto-generated "Initial commit"
  (`8bbff6e`) that shares no history with this repo, so a plain push will be rejected. Do this:
  ```bash
  git fetch https://x-access-token:${GITHUB_TOKEN}@github.com/Ankritjarngal/medvqa-cxr-chexagent.git main:github-main
  git merge github-main --allow-unrelated-histories -m "Merge GitHub initial commit"
  ```
  If files conflict (likely `README.md`, `LICENSE` or `.gitignore`), keep OUR version of each
  file (`git checkout --ours <file>`), `git add` it, and finish the merge. Then run the unit tests
  (`python -m pytest tests/`) before pushing. Then push normally (no force) as above.
- Previous attempts: two agent runs failed with "could not read Username" because no token was
  set. The GitHub integration alone did not provide push credentials. A `GITHUB_TOKEN` secret is
  required.
- Verify with `git ls-remote https://github.com/Ankritjarngal/medvqa-cxr-chexagent.git`.
- If there is no token, or the push is rejected, stop and report the exact error. Do not try
  other forges or workarounds.
- Do not open pull requests unless asked.

## Standing task: keep GitHub in sync

The cloud agent that does the analysis has no GitHub credentials; it commits to the Cursor remote. When
asked to sync: `git fetch` both remotes, push local `main` to GitHub as a fast-forward (never force), run
`python -m pytest tests/` first, and report the resulting commit id. Commits not yet on GitHub as of the
Phase 1b work: everything after `b1f850a` and `d34777f`, up to the newest on the Cursor remote (the run 2
results, the Phase 1b code, and the documentation).

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
- Documentation rule: every piece of work gets an entry in `docs/experiment_log.md`, and corrections
  to earlier statements are recorded there and in the affected note, not made silently.
- Known limitation to state honestly: CheXagent's training data (CheXinstruct) already includes
  VQA-RAD, SLAKE and Open-I, so the Phase 1 test is close to in-distribution, not clean zero-shot.
