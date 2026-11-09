"""Generates notebooks/04_disease_naming.ipynb (Phase 1d / Stage 1 of docs/phase2_rag_plan.md).

The repo's own tested modules are inlined so Colab runs exactly the code in `baseline/`. Install and model
cells are copied from notebook 01. Run build_baseline_notebook.py first if 01 changed.
"""

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb_common import GUESS_CELL, YES_NO_CELL, cell_from_01, inline_module  # noqa: E402

OUT_NB = "notebooks/04_disease_naming.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(r"""# Phase 1d: can CheXagent-2-3b name the disease on a chest X-ray?

Stage 1 of the Phase 2 plan (`docs/phase2_rag_plan.md`). SLAKE's knowledge answers (cause, symptoms, treatment,
prevention) are fixed by the disease on the image, and the question never names it. So a retrieval system first needs the
model to say which disease it sees. This notebook measures that, with no retrieval and no training.

- **Images:** the 130 SLAKE chest X-rays that have a disease label (train 90, validation 20, test 20). Labels use the
  ChestX-ray14 names (SLAKE spellings merged).
- **Three ways of asking:** `open` (free question), `forced` (pick one name from the 14 ChestX-ray14 names plus "No Finding"),
  and `perclass` ("Is there X in this chest X-ray?" for each of the 15 names; the name with the highest yes-minus-no score).
- **Scored against:** always answering the most common label (Cardiomegaly, 22%) and shuffled labels, with per-class recall
  and train versus validation/test shown separately.
- **Downstream choice, fixed in advance:** the method with the best accuracy on the 20 *validation* images is the one the
  RAG notebook uses (`choose_method`).

**Runtime:** free Colab **T4**. `open` and `forced` are 130 short generations each; `perclass` is 15 one-token passes per
image (about 2,000). Expect 30 to 60 minutes. Progress is saved to `disease_checkpoint.json` every 20 answers, so a dropped
session can resume. Set `QUICK = True` in the build cell for a short smoke test first.
""")

md("## 1. Setup\n\nIdentical to the other notebooks. If Colab asks you to restart the runtime after this cell, do so and continue from the next cell.")
code(cell_from_01("Install dependencies"))

md("## 2. Build the items\n\nThe cells below are the repo's own modules, inlined unchanged. Order matters.")
for mod in ("categorize", "data", "evaluate", "stats", "sides", "extra_sets", "disease_labels"):
    code(inline_module(f"baseline/{mod}.py"))

code(r'''import json, os, random
from collections import Counter

QUICK = False   # True = 12 images, then set back to False for the real run

pool = fetch_full_pool()
items = attach_images(disease_id_items(pool), "images")
if QUICK:
    items = random.Random(0).sample(items, 12)
print("QUICK mode:", QUICK)
print("Images:", len(items), dict(Counter(i["source_split"] for i in items)))
print("Labels:", dict(Counter(l for i in items for l in i["labels"])))
''')

md("## 3. Load CheXagent-2-3b\n\nIdentical to the other notebooks (downloads about 12.6 GB the first time).")
code(cell_from_01("snapshot_download(MODEL_NAME"))
code(cell_from_01("def ask_chexagent"))
code(YES_NO_CELL)
code(GUESS_CELL)

md("## 4. Run\n\nEvery answer is stored under a key and written to disk every 20 answers.")
code(r'''from tqdm.auto import tqdm

CKPT = "disease_checkpoint.json"
store = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
_since = 0

def cached(key, fn):
    global _since
    if key not in store:
        store[key] = fn()
        _since += 1
        if _since % 20 == 0:
            json.dump(store, open(CKPT, "w"))
    return store[key]

for method in ("open", "forced"):
    for it in tqdm(items, desc=method):
        cached(f"{method}|{it['id']}", lambda: ask_chexagent(it["image_path"], DISEASE_PROMPTS[method], 48))

for it in tqdm(items, desc="perclass"):
    for name in CANDIDATES:
        cached(f"perclass|{it['id']}|{name}", lambda: yes_no_score(it["image_path"], perclass_question(name)))
json.dump(store, open(CKPT, "w"))
print("Done.")
''')

md("""## 5. Results

`acc` is the share of images where the top choice matches any gold label. `majority` is always answering the most common
label; `shuffled` is the same answers scored against shuffled gold labels (a model that ignores the image lands here).
""")
code(r'''def rows_for(method, subset=None):
    rows = []
    for it in (subset or items):
        if method == "perclass":
            scores = {n: store[f"perclass|{it['id']}|{n}"] for n in CANDIDATES}
            pred = sorted(scores, key=lambda c: -scores[c])[:1]
        else:
            pred = parse_diseases(store[f"{method}|{it['id']}"])
        rows.append({"id": it["id"], "labels": it["labels"], "pred": pred, "split": it["source_split"]})
    return rows

METHODS = ("open", "forced", "perclass")
reports = {m: disease_report(rows_for(m)) for m in METHODS}
print(format_disease_report(reports))
''')
code(r'''scores_by_image = {it["id"]: {n: store[f"perclass|{it['id']}|{n}"] for n in CANDIDATES} for it in items}
labels_by_image = {it["id"]: it["labels"] for it in items}
print("perclass: one-vs-rest AUC per candidate (0.5 = no signal; sensible only where the class has images)")
counts = Counter(l for it in items for l in it["labels"])
for name, auc in class_table(scores_by_image, labels_by_image).items():
    if counts.get(name):
        print(f"  {name:>14}  n={counts[name]:>2}  AUC={auc:.2f}")
top3 = sum(any(l in perclass_predictions(scores_by_image)[i][:3] for l in labels_by_image[i]) for i in scores_by_image)
print(f"perclass: a gold label is among the top 3 scores on {top3} / {len(scores_by_image)} images")
''')

md("""## 6. Method chosen for the RAG notebook, and the reading

Chosen on the **validation images only**, as fixed in the plan. The reading below is the pre-stated one.
""")
code(r'''val = [i for i in items if i["source_split"] == "validation"]
val_reports = {m: disease_report(rows_for(m, val), n_shuffles=200) for m in METHODS}
for m in METHODS:
    print(f"validation ({len(val)} images) {m:>9}: {val_reports[m]['acc_top']:.0%}")
CHOSEN_METHOD = choose_method(val_reports)
print("\nCHOSEN_METHOD =", repr(CHOSEN_METHOD), "  (use this value in notebook 05)")
print("\nReading for the chosen method on all images:")
print(" ", disease_reading(reports[CHOSEN_METHOD]))
for m in METHODS:
    if m != CHOSEN_METHOD:
        print(f"  [{m}] {disease_reading(reports[m])}")
''')

md("## 7. Save results\n\nKeep these files. The Colab VM is wiped when the session ends.")
code(r'''import csv

rows = []
for it in items:
    for m in ("open", "forced"):
        raw = store[f"{m}|{it['id']}"]
        pred = parse_diseases(raw)
        rows.append({"id": it["id"], "split": it["source_split"], "gold": ", ".join(it["labels"]), "method": m,
                     "raw": raw, "top": pred[0] if pred else "", "correct": bool(pred and pred[0] in it["labels"])})
    scores = scores_by_image[it["id"]]
    order = sorted(scores, key=lambda c: -scores[c])
    rows.append({"id": it["id"], "split": it["source_split"], "gold": ", ".join(it["labels"]), "method": "perclass",
                 "raw": json.dumps({k: round(v, 3) for k, v in scores.items()}), "top": order[0],
                 "correct": order[0] in it["labels"]})
with open("disease_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "split", "gold", "method", "raw", "top", "correct"])
    w.writeheader()
    w.writerows(rows)
print(f"Saved disease_results.csv ({len(rows)} rows)")
try:
    from google.colab import files
    files.download("disease_results.csv")
    files.download("disease_checkpoint.json")
except Exception as exc:
    print("Auto-download unavailable; use the Colab file browser instead.", exc)
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": "04_disease_naming.ipynb", "gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT_NB}")
