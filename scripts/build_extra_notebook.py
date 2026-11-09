"""Generates notebooks/02_knowledge_and_side_baselines.ipynb.

Same approach as build_baseline_notebook.py: the notebook inlines the repo's own
tested modules so Colab runs exactly the code in `baseline/`. The install and
model-loading cells are copied from notebook 01 so the two stay identical.
Run build_baseline_notebook.py first if you changed notebook 01.
"""

import json

import nbformat as nbf

SRC_NB = "notebooks/01_baseline_chexagent.ipynb"
OUT_NB = "notebooks/02_knowledge_and_side_baselines.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


def inline_module(relpath):
    """Module source with package-relative imports and the command-line entry
    point removed (an inlined `main()` would parse Colab's own sys.argv)."""
    keep = []
    for ln in open(relpath).read().splitlines():
        if ln.startswith('if __name__ == "__main__":'):
            break
        if ln.startswith("from __future__") or ln.startswith("from ."):
            continue
        keep.append(ln)
    return "\n".join(keep).strip() + "\n"


def cell_from_01(marker):
    """Source of the first code cell in notebook 01 containing `marker`."""
    for c in json.load(open(SRC_NB))["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise SystemExit(f"marker not found in {SRC_NB}: {marker}")


md(r"""# Phase 1b: knowledge and side-question baselines for CheXagent-2-3b

Two supplementary zero-shot test sets, run with the **same model and the same settings as
notebook 01** (no retrieval, no training). They exist because the 224-question run left two
questions open:

1. **Knowledge questions** (symptoms, treatment, prevention, cause, function, "why"): run 2 had
   only 24. About 130 exist in SLAKE + VQA-RAD. This set is the reference that a Phase 2
   retrieval layer will later be compared against.
2. **Side questions** ("which side", "where is the abnormality" with an answer like *Right* or
   *Lower Left Lung*): run 2 hinted that the model defaults to "right" from 13 questions. This
   set has equal numbers of left and right answers (60 + 60, plus 20 bilateral) to test that.

**Read before interpreting:**
- Both sets include SLAKE *train* rows. CheXagent was very likely trained on those, so results are
  reported separately for train and test/validation rows. Train rows make the set bigger, not
  cleaner.
- The knowledge questions come from a few fixed templates (about 42 distinct question texts and 33
  distinct gold answers across 130 questions). Treat them as a handful of independent tests, not 130.
- Knowledge questions use the **plain question as the prompt**. Run 2 appended "based only on what is
  visible in the image" to some of them, which is the wrong instruction for a knowledge question.

**Runtime:** free Colab, **T4 GPU**. About 270 questions in total.
""")

md("## 1. Setup\n\nIdentical to notebook 01. If Colab asks you to restart the runtime after this cell, do so and continue from the next cell.")
code(cell_from_01("Install dependencies"))

md("## 2. Build both test sets\n\nThe cells below are the repo's own modules, inlined unchanged. Order matters: later modules use names defined by earlier ones.")
for mod in ("categorize", "data", "evaluate", "stats", "sides", "knowledge", "extra_sets"):
    code(inline_module(f"baseline/{mod}.py"))

code(r'''# One download of the dataset files (about 5 requests, a minute at most), shared by both sets.
# Earlier versions paged through the Hugging Face API (~150 requests) and got HTTP 429 on Colab.
pool = fetch_full_pool()
knowledge_set = build_knowledge_test_set(images_dir="images", pool=pool)
side_set = build_side_test_set(images_dir="images", pool=pool)
import json
for name, s in (("knowledge", knowledge_set), ("side", side_set)):
    print(name, json.dumps(describe_set(s), indent=1))
''')

md("## 3. Load CheXagent-2-3b\n\nIdentical to notebook 01 (downloads about 12.6 GB the first time).")
code(cell_from_01("snapshot_download(MODEL_NAME"))
code(cell_from_01("def ask_chexagent"))

md("## 4. Run inference\n\nPredictions are saved to disk every 20 questions so a dropped Colab session does not lose the whole run.")
code(r'''import os
from tqdm.auto import tqdm

def run_set(name, records, checkpoint):
    done = json.load(open(checkpoint)) if os.path.exists(checkpoint) else {}
    for i, r in enumerate(tqdm(records, desc=name)):
        if r["id"] in done:
            r["prediction"] = done[r["id"]]
            continue
        prompt = prompt_for_set(name, r["question"], r["category"])
        r["prediction"] = ask_chexagent(r["image_path"], prompt)
        done[r["id"]] = r["prediction"]
        if i % 20 == 0:
            json.dump(done, open(checkpoint, "w"))
    json.dump(done, open(checkpoint, "w"))

run_set("knowledge", knowledge_set, "knowledge_checkpoint.json")
run_set("side", side_set, "side_checkpoint.json")
print("Done.")
''')

md("""## 5. Knowledge questions: scores

`item_recall` / `item_f1` split the gold answer into items ("Chest pain, dyspnea" has two) and give
credit when the prediction covers an item, after mapping a short table of synonyms
(`CONCEPT_SYNONYMS`, readable in the cell above) onto one spelling. It is more lenient than exact match
and is a screening score, not a clinical judgement. Exact match and token F1 are shown for comparison.
""")
code(r'''k_report = evaluate(knowledge_set)
k_rows = score_knowledge_rows(k_report.rows)
print(format_report(k_report))
print()
print(format_knowledge_summary(summarize_knowledge(k_rows)))

print("\nBy train vs test/validation rows (CheXagent probably saw the train rows):")
for split in ("train", "validation", "test"):
    sub = [r for r in k_rows if r["source_split"] == split]
    if sub:
        print(f"  {split:>10}: n={len(sub):<3} item_recall={sum(r['item_recall'] for r in sub)/len(sub):.2f} "
              f"exact={sum(r['exact_match'] for r in sub)/len(sub):.1%}")
''')

md("""## 6. Side questions: scores

A prediction is correct on side if the extracted side (left / right / both) matches the gold side, so
"Left lung" counts as correct for "Lower Left Lung". The report shows the confusion table and tests
whether errors are lopsided: if gold-left questions are answered "right" much more often than
gold-right questions are answered "left", that is a right-side default.
""")
code(r'''s_report = evaluate(side_set)
s_rows = score_side_rows(s_report.rows)
print(format_report(s_report))
print()
print(format_side_report(side_report(s_rows)))

print("\nBy train vs test/validation rows:")
for split in ("train", "validation", "test"):
    sub = [r for r in s_rows if r["source_split"] == split]
    if sub:
        print(f"  {split:>10}: n={len(sub):<3} side_correct={sum(r['side_correct'] for r in sub)/len(sub):.1%}")
print("\nBy source:")
for src in ("SLAKE", "VQA-RAD"):
    sub = [r for r in s_rows if r["source_dataset"] == src]
    if sub:
        print(f"  {src:>8}: n={len(sub):<3} side_correct={sum(r['side_correct'] for r in sub)/len(sub):.1%}")
''')

md("## 7. Look at some wrong answers\n\nSide errors first. Check that the images are chest X-rays and whether the model's answer is wrong or just worded differently.")
code(r'''from PIL import Image
import matplotlib.pyplot as plt

wrong = [r for r in s_rows if not r["side_correct"]][:8]
print(f"{sum(not r['side_correct'] for r in s_rows)} wrong side answers; showing {len(wrong)}")
for r in wrong:
    plt.figure(figsize=(3, 3)); plt.imshow(Image.open(r["image_path"]).convert("RGB"), cmap="gray"); plt.axis("off")
    plt.title(f"{r['source_dataset']} / {r['source_split']}", fontsize=8); plt.show()
    print(f"Q: {r['question']}\n   gold={r['gold_answer']!r}  pred={r['prediction']!r}")
''')

md("## 8. Save results\n\nKeep these files. The Colab VM is wiped when the session ends.")
code(r'''import csv

def save(name, rows, extra):
    base = ["id", "source_dataset", "source_split", "category", "answer_format", "is_knowledge",
            "image_path", "question", "gold_answer", "prediction", "exact_match", "loose_match", "token_f1"]
    fields = base + extra
    with open(f"{name}_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    with open(f"{name}_test_set.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

save("knowledge", k_rows, ["knowledge_subtype", "item_recall", "item_precision", "item_f1"])
save("side", s_rows, ["side_granularity", "gold_side", "pred_side", "side_given", "side_correct"])
print("Saved knowledge_results.csv, side_results.csv and the matching *_test_set.jsonl files.")
try:
    from google.colab import files
    for fn in ("knowledge_results.csv", "side_results.csv", "knowledge_test_set.jsonl", "side_test_set.jsonl"):
        files.download(fn)
except Exception as exc:
    print("Auto-download unavailable; use the Colab file browser instead.", exc)
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": "02_knowledge_and_side_baselines.ipynb", "gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT_NB}")
