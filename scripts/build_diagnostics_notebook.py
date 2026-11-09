"""Generates notebooks/03_perception_diagnostics.ipynb.

Same approach as the other notebook builders: the repo's own tested modules are
inlined so Colab runs exactly the code in `baseline/`. Install and model cells are
copied from notebook 01. Run build_baseline_notebook.py first if 01 changed.
"""

import json

import nbformat as nbf

SRC_NB = "notebooks/01_baseline_chexagent.ipynb"
OUT_NB = "notebooks/03_perception_diagnostics.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


def inline_module(relpath):
    keep = []
    for ln in open(relpath).read().splitlines():
        if ln.startswith('if __name__ == "__main__":'):
            break
        if ln.startswith("from __future__") or ln.startswith("from ."):
            continue
        keep.append(ln)
    return "\n".join(keep).strip() + "\n"


def cell_from_01(marker):
    for c in json.load(open(SRC_NB))["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise SystemExit(f"marker not found in {SRC_NB}: {marker}")


md(r"""# Phase 1c: perception diagnostics for CheXagent-2-3b

Run 3 found two perception faults in the same model: a constant **"Right"** on explicit "which side" questions
(26 of 29, whatever the image) and **"no"** to heart-enlargement questions whose answer was "yes". These three cheap
tests ask what could explain them. No training and no retrieval; same model and settings as the other notebooks.

| Test | Question | How |
|---|---|---|
| **A. Flip** | Does the model read left/right from the image at all? | Mirror each image. If it reads laterality its answers should swap; a constant answer will not. |
| **B. Prompt variants** | Is the "Right" habit a wording effect? | Six wordings on 60 side questions, including a forced choice, describe-first, and stating the left/right convention. |
| **C. Heart size** | Can the model ever say "enlarged"? | 47 heart questions (22 with answer "yes") with four wordings, including "normal or enlarged?". |

**What each outcome would mean was written down before running** in `docs/phase1c_diagnostics.md`. Read it first.

**Runtime:** free Colab **T4**. Full run is about 700 short generations (expect roughly 30-60 minutes). Set `QUICK = True`
in the build cell for a ten-minute smoke test first. Progress is saved to `diagnostics_checkpoint.json`, so a dropped session
can resume: re-run the cells and finished answers are skipped.
""")

md("## 1. Setup\n\nIdentical to the other notebooks. If Colab asks you to restart the runtime after this cell, do so and continue from the next cell.")
code(cell_from_01("Install dependencies"))

md("## 2. Build the test items\n\nThe cells below are the repo's own modules, inlined unchanged. Order matters.")
for mod in ("categorize", "data", "evaluate", "stats", "sides", "extra_sets", "diagnostics"):
    code(inline_module(f"baseline/{mod}.py"))

code(r'''import json, os, random
from collections import Counter

QUICK = False   # True = tiny smoke test (a few items per test), then set back to False for the real run

pool = fetch_full_pool()
side_all = attach_images(select_side_records(pool), "images")
flip_items = [r for r in side_all if r["gold_side"] in ("left", "right")]          # 120
diag_items = select_side_diagnostic_records(side_all)                               # 60, a subset of flip_items
heart_items = attach_images(select_heart_size_records(pool), "images")              # 47

if QUICK:
    rng = random.Random(0)
    flip_items = rng.sample(flip_items, 10)
    diag_items = [r for r in diag_items if r["id"] in {x["id"] for x in flip_items}] or flip_items[:6]
    heart_items = rng.sample(heart_items, 8)

os.makedirs("images_flipped", exist_ok=True)
for r in flip_items:
    r["flipped_path"] = flip_image(r["image_path"], f"images_flipped/{os.path.basename(r['image_path'])}")
for r in flip_items + diag_items:
    r["explicit"] = is_explicit_side_question(r["question"])

print("QUICK mode:", QUICK)
print("Flip test items:", len(flip_items), "| explicit side questions among them:", sum(r["explicit"] for r in flip_items))
print("Side prompt-variant items:", len(diag_items), Counter(r["gold_side"] for r in diag_items))
print("Heart items:", len(heart_items), "| answer yes:", sum(r["gold_yes"] for r in heart_items),
      "| sources:", dict(Counter(r["source_dataset"] for r in heart_items)))
''')

md("## 3. Load CheXagent-2-3b\n\nIdentical to the other notebooks (downloads about 12.6 GB the first time).")
code(cell_from_01("snapshot_download(MODEL_NAME"))
code(cell_from_01("def ask_chexagent"))

md("## 4. Run the tests\n\nEvery answer is stored under a key (`test|variant|question id`) and written to disk every 20 answers.")
code(r'''from tqdm.auto import tqdm

CKPT = "diagnostics_checkpoint.json"
store = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
_since_save = 0

def ask_cached(key, image_path, prompt, max_new_tokens=128, force=False):
    global _since_save
    if key in store and not force:
        return store[key]
    store[key] = ask_chexagent(image_path, prompt, max_new_tokens)
    _since_save += 1
    if _since_save % 20 == 0:
        json.dump(store, open(CKPT, "w"))
    return store[key]

# --- Test A: original vs mirrored image, with the same prompt run 3 used for side questions
for r in tqdm(flip_items, desc="A: flip"):
    prompt = SIDE_PROMPTS["suffix"].format(question=r["question"])
    r["pred_original"] = ask_cached(f"A|original|{r['id']}", r["image_path"], prompt)
    r["pred_flipped"] = ask_cached(f"A|flipped|{r['id']}", r["flipped_path"], prompt)

# --- Repeatability check: the same question asked again should give the same answer (greedy decoding)
repeat_items = flip_items[:20]
same = 0
for r in tqdm(repeat_items, desc="A: repeat"):
    again = ask_cached(f"A|repeat|{r['id']}", r["image_path"],
                       SIDE_PROMPTS["suffix"].format(question=r["question"]), force=True)
    same += again == r["pred_original"]
print(f"Repeatability: {same} / {len(repeat_items)} identical answers on a second run")

# --- Test B: side prompt variants
by_id = {r["id"]: r for r in flip_items}
for variant, template in SIDE_PROMPTS.items():
    for r in tqdm(diag_items, desc=f"B: {variant}"):
        if variant == "suffix":
            r["pred_suffix"] = by_id[r["id"]]["pred_original"]
            continue
        r[f"pred_{variant}"] = ask_cached(f"B|{variant}|{r['id']}", r["image_path"],
                                          template.format(question=r["question"]),
                                          SIDE_MAX_TOKENS.get(variant, 128))

# --- Test C: heart-size variants
for variant, template in HEART_PROMPTS.items():
    for r in tqdm(heart_items, desc=f"C: {variant}"):
        r[f"pred_{variant}"] = ask_cached(f"C|{variant}|{r['id']}", r["image_path"],
                                          template.format(question=r["question"]),
                                          HEART_MAX_TOKENS.get(variant, 128))
json.dump(store, open(CKPT, "w"))
print("Done.")
''')

md("""## 5. Results

**A. Flip test.** `swap rate` is the share of items answered left/right on both images where the answer changed
side. A model that reads laterality should be near 100%. A constant answer gives near 0%. Mirroring also moves the heart
to the right of the picture, so anatomy-based cues flip too; that would show up as a swap, not hide one.
""")
code(r'''print(format_flip_report(flip_report(flip_items)))
print(f"\nRepeatability check: {same} / {len(repeat_items)} identical on a second run "
      "(if this is well below 20, differences between runs are partly noise).")
''')

md("**B. Side prompt variants** on the 60 side questions, and again on the explicit ones only.")
code(r'''def rows_for(variant, items):
    return [{"gold_side": r["gold_side"], "prediction": r[f"pred_{variant}"]} for r in items]

print(format_variant_table({v: variant_report(rows_for(v, diag_items)) for v in SIDE_PROMPTS}))
explicit_items = [r for r in diag_items if r["explicit"]]
print(f"\nExplicit 'which side' questions only (n={len(explicit_items)}):")
print(format_variant_table({v: variant_report(rows_for(v, explicit_items)) for v in SIDE_PROMPTS}))
''')

md("""**C. Heart size.** Positive means "heart enlarged". Sensitivity is the share of enlarged hearts the model called
enlarged. `always-no` is the accuracy of answering "no" to everything. `unparsed` counts answers that contain no usable
yes/no or normal/enlarged (counted as wrong). The "describe first" variant is parsed from its last yes/no.
""")
code(r'''def heart_rows(variant):
    return [{"gold_yes": r["gold_yes"], "prediction": r[f"pred_{variant}"]} for r in heart_items]

print(format_heart_table({v: heart_report(heart_rows(v), v) for v in HEART_PROMPTS}))
print("\nExamples of 'describe first' answers:")
for r in heart_items[:5]:
    print(f"  gold={'yes' if r['gold_yes'] else 'no'}: {r['pred_describe_first'][:200]!r}")
''')

md("## 6. Save results\n\nKeep these files. The Colab VM is wiped when the session ends.")
code(r'''import csv

rows = []
for r in flip_items:
    for v, k in (("original", "pred_original"), ("flipped", "pred_flipped")):
        rows.append({"test": "A_flip", "variant": v, "id": r["id"], "source_dataset": r["source_dataset"],
                     "source_split": r["source_split"], "question": r["question"], "gold": r["gold_answer"],
                     "gold_side": r["gold_side"], "explicit": r["explicit"], "prediction": r[k]})
for v in SIDE_PROMPTS:
    for r in diag_items:
        rows.append({"test": "B_side_prompt", "variant": v, "id": r["id"], "source_dataset": r["source_dataset"],
                     "source_split": r["source_split"], "question": r["question"], "gold": r["gold_answer"],
                     "gold_side": r["gold_side"], "explicit": r["explicit"], "prediction": r[f"pred_{v}"]})
for v in HEART_PROMPTS:
    for r in heart_items:
        rows.append({"test": "C_heart", "variant": v, "id": r["id"], "source_dataset": r["source_dataset"],
                     "source_split": r["source_split"], "question": r["question"], "gold": r["gold_answer"],
                     "gold_side": "", "explicit": "", "prediction": r[f"pred_{v}"],
                     "parsed": parse_heart_answer(v, r[f"pred_{v}"])})

fields = ["test", "variant", "id", "source_dataset", "source_split", "question", "gold", "gold_side", "explicit",
          "prediction", "parsed"]
with open("diagnostics_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in fields})
print(f"Saved diagnostics_results.csv ({len(rows)} rows)")
try:
    from google.colab import files
    files.download("diagnostics_results.csv")
    files.download("diagnostics_checkpoint.json")
except Exception as exc:
    print("Auto-download unavailable; use the Colab file browser instead.", exc)
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": "03_perception_diagnostics.ipynb", "gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT_NB}")
