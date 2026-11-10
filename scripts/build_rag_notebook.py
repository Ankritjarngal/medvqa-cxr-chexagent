"""Generates notebooks/05_rag_knowledge_arms.ipynb (Stage 3 of docs/phase2_rag_plan.md).

The repo's own tested modules are inlined so Colab runs exactly the code in `baseline/`. Install and model
cells are copied from notebook 01. Run build_baseline_notebook.py first if 01 changed.
"""

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb_common import GUESS_CELL, YES_NO_CELL, cell_from_01, inline_module  # noqa: E402

OUT_NB = "notebooks/05_rag_knowledge_arms.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(r"""# Phase 2: knowledge-grounded RAG, generation arms

Stage 3 of the Phase 2 plan (`docs/phase2_rag_plan.md`; decision rule: Amendment 5 in `docs/evaluation_protocol.md`).
Same model, greedy decoding and prompt style in every arm; only the disease hint and the retrieved text differ.

| Arm | What the model is given | Questions |
|---|---|---|
| **A1** | the question (style-matched, no hint, no retrieval) | all 130 knowledge questions |
| **A2** | retrieval from the question alone | oracle subset (50) |
| **A3** | the *true* disease named (oracle) | oracle subset |
| **A4** | the *true* disease plus retrieval (oracle) | oracle subset |
| **A5** | the model's *own* disease guess named | all 130 |
| **A6** | the model's own guess plus retrieval: **the system** | all 130 |

A2 to A4 use SLAKE's own disease annotation and are **upper bounds, not results** (Amendment 5). A5 and A6 use only the model's
output.

**Before you run:** run notebook 04 first and paste the `CHOSEN_METHOD` it prints into the build cell below. The retrieval
setting (`topic+textbooks`, 2 passages, 700 characters) was frozen before any model run (plan, section 10).

**Runtime:** free Colab **T4**. About 130 disease guesses (one to 15 passes each) and about 700 short generations. Expect 30 to
60 minutes. Progress is saved to `rag_checkpoint.json` every 20 answers; re-run the cells to resume. `QUICK = True` runs a
16-question smoke test.
""")

md("## 1. Setup\n\nIdentical to the other notebooks. If Colab asks you to restart the runtime after this cell, do so and continue from the next cell.")
code(cell_from_01("Install dependencies"))

md("## 2. Build the questions and the retriever\n\nThe cells below are the repo's own modules, inlined unchanged. Order matters.")
for mod in ("categorize", "data", "evaluate", "stats", "sides", "extra_sets", "knowledge", "disease_labels", "rag"):
    code(inline_module(f"baseline/{mod}.py"))

code(r'''import json, os, random
from collections import Counter

QUICK = False                  # True = 16-question smoke test, then set back to False
DISEASE_METHOD = None          # <-- paste CHOSEN_METHOD from notebook 04: "open", "forced" or "perclass"
assert DISEASE_METHOD in ("open", "forced", "perclass"), "set DISEASE_METHOD from notebook 04 first"

# Frozen before any model run (docs/phase2_rag_plan.md, section 10)
SOURCE, K, MAX_CHARS, WEIGHT = "topic+textbooks", 2, 700, 1

pool = fetch_full_pool()
know_items = attach_images(attach_oracle_disease(select_knowledge_records(pool), pool), "images")
dev_ids = {r["id"] for r in split_dev_heldout(know_items)[0]}
for r in know_items:
    r["is_dev"] = r["id"] in dev_ids
    r["has_oracle"] = bool(r["oracle_disease"]) and r["knowledge_subtype"] != "function"
if QUICK:
    rng = random.Random(0)
    oracle = [r for r in know_items if r["has_oracle"]]
    others = [r for r in know_items if not r["has_oracle"]]
    know_items = rng.sample(oracle, 10) + rng.sample(others, 6)

print("QUICK mode:", QUICK, "| disease method:", DISEASE_METHOD)
print("Knowledge questions:", len(know_items), dict(Counter(r["knowledge_subtype"] for r in know_items)))
print("With a known single disease (oracle subset):", sum(r["has_oracle"] for r in know_items))
print("Retrieval:", SOURCE, "k =", K, "max chars =", MAX_CHARS)
retrievers = build_retrievers()
print("Passages indexed:", {k: v.n for k, v in retrievers.items()})
''')

md("## 3. Load CheXagent-2-3b\n\nIdentical to the other notebooks (downloads about 12.6 GB the first time).")
code(cell_from_01("snapshot_download(MODEL_NAME"))
code(cell_from_01("def ask_chexagent"))
code(cell_from_01("IMAGE CHECK"))
code(YES_NO_CELL)
code(GUESS_CELL)

md("""## 4. Token budget

The model reads at most 2048 tokens, and the image slots are part of that. The check measures the longest prompt including the image
slots (all arms that add retrieved text). If it exceeds the text budget the passages are shortened for **every** arm and the change is printed and
saved in `rag_settings.json`. Note it in the experiment log.
""")
code(r'''TEXT_TOKEN_BUDGET = 1900   # whole prompt including the image slots; 2048 limit minus 64 new tokens and a margin

def prompt_tokens(prompt, image_path):
    tok = globals().get("tokenizer")
    if tok is None:
        return len(prompt.split()) * 2          # rough stand-in when no tokenizer is loaded
    query = tok.from_list_format([{"image": image_path}, {"text": prompt}])
    conv = [{"from": "system", "value": "You are a helpful radiology assistant. Answer the question as concisely as possible based only on the image."},
            {"from": "human", "value": query}]
    return int(tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").size(1))

def longest_prompt(max_chars):
    worst = 0
    for r in know_items:
        if not r["has_oracle"]:
            continue
        ps = retrieve_context(retrievers, SOURCE, r["question"], r["knowledge_subtype"], r["oracle_disease"],
                              K, max_chars, WEIGHT)
        worst = max(worst, prompt_tokens(build_prompt(r["question"], r["oracle_disease"], ps), r["image_path"]))
    return worst

while True:
    worst = longest_prompt(MAX_CHARS)
    print(f"longest prompt at {MAX_CHARS} characters per passage: {worst} tokens (budget {TEXT_TOKEN_BUDGET})")
    if worst <= TEXT_TOKEN_BUDGET or MAX_CHARS <= 300:
        break
    MAX_CHARS -= 100
print("Using MAX_CHARS =", MAX_CHARS)
''')

md("## 5. Run the arms\n\nDisease guesses first (one per image), then every arm. Each answer is stored under a key and written to disk every 20 answers.")
code(r'''from tqdm.auto import tqdm

CKPT = "rag_checkpoint.json"
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

# --- the model's own disease guess, once per image
images = {r["image_path"]: r for r in know_items}
for path in tqdm(images, desc=f"disease guess ({DISEASE_METHOD})"):
    cached(f"guess|{DISEASE_METHOD}|{path}", lambda: guess_disease(path, DISEASE_METHOD)[1])

def disease_top(r):
    g = store[f"guess|{DISEASE_METHOD}|{r['image_path']}"]
    if DISEASE_METHOD == "perclass":
        return sorted(g, key=lambda c: -g[c])[0]
    named = parse_diseases(g)
    return named[0] if named else None

for r in know_items:
    r["guessed_disease"] = disease_top(r)
print("Guessed diseases:", dict(Counter(r["guessed_disease"] for r in know_items)))

# --- arms: (which disease, use retrieval)
ARMS = {"A1": (None, False), "A2": (None, True), "A3": ("oracle", False), "A4": ("oracle", True),
        "A5": ("guess", False), "A6": ("guess", True)}
ORACLE_ONLY = {"A2", "A3", "A4"}

def arm_input(arm, r):
    which, use_retrieval = ARMS[arm]
    disease = {None: None, "oracle": r["oracle_disease"], "guess": r["guessed_disease"]}[which]
    passages = (retrieve_context(retrievers, SOURCE, r["question"], r["knowledge_subtype"], disease, K, MAX_CHARS, WEIGHT)
                if use_retrieval else [])
    return disease, passages, build_prompt(r["question"], disease, passages)

def run_arm(arm):
    records = [r for r in know_items if r["has_oracle"]] if arm in ORACLE_ONLY else know_items
    out = []
    for r in tqdm(records, desc=arm):
        disease, passages, prompt = arm_input(arm, r)
        pred = cached(f"{arm}|{r['id']}", lambda: ask_chexagent(r["image_path"], prompt, 64))
        out.append({"arm": arm, "id": r["id"], "source_dataset": r["source_dataset"], "source_split": r["source_split"],
                    "is_dev": r["is_dev"], "knowledge_subtype": r["knowledge_subtype"], "question": r["question"],
                    "gold_answer": r["gold_answer"], "oracle_disease": r["oracle_disease"] or "",
                    "guessed_disease": r["guessed_disease"] or "", "disease_used": disease or "",
                    "passages": " || ".join(f"{p['title'][:60]} ({p['score']})" for p in passages),
                    "prediction": pred, "image_key": r["image_key"]})
    return out

all_rows = []
for arm in ARMS:
    all_rows += run_arm(arm)
json.dump(store, open(CKPT, "w"))
scored = score_knowledge_rows(all_rows)
print("Done:", len(scored), "answers")
''')

md("""## 6. Quick look

Means only; the formal paired comparisons, intervals and blind rating sheets are produced afterwards from the saved files
(`scripts/analyze_rag.py`). Function questions are shown separately because every gold answer is "Breathe".
""")
code(r'''def mean(rows, key="item_recall"):
    return sum(float(r[key]) for r in rows) / len(rows) if rows else float("nan")

print(f"{'arm':>4} {'n (non-function)':>16} {'item recall':>12} | {'function n':>10} {'said Breathe':>12}")
for arm in ARMS:
    rows = [r for r in scored if r["arm"] == arm]
    nf = [r for r in rows if r["knowledge_subtype"] != "function"]
    fn = [r for r in rows if r["knowledge_subtype"] == "function"]
    breathe = sum("breathe" in r["prediction"].lower() for r in fn)
    print(f"{arm:>4} {len(nf):>16} {mean(nf):>12.2f} | {len(fn):>10} {breathe:>12}")

print("\nOracle subset only (same questions in every arm):")
oracle_ids = {r["id"] for r in know_items if r["has_oracle"]}
for arm in ARMS:
    rows = [r for r in scored if r["arm"] == arm and r["id"] in oracle_ids]
    print(f"{arm:>4} n={len(rows):>3} item recall {mean(rows):.2f}")

known = [r for r in know_items if r["has_oracle"]]
hit = sum(r["guessed_disease"] == r["oracle_disease"] for r in known)
print(f"\nThe model's disease guess matches the true disease on {hit} / {len(known)} oracle-subset questions")
''')

md("## 7. Save results\n\nKeep these files. The Colab VM is wiped when the session ends.")
code(r'''import csv

fields = ["arm", "id", "source_dataset", "source_split", "is_dev", "knowledge_subtype", "question", "gold_answer",
          "oracle_disease", "guessed_disease", "disease_used", "passages", "prediction", "image_key",
          "item_recall", "item_precision", "item_f1", "token_f1"]
with open("rag_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in scored:
        w.writerow({k: row.get(k, "") for k in fields})
json.dump({"disease_method": DISEASE_METHOD, "source": SOURCE, "k": K, "max_chars": MAX_CHARS, "weight": WEIGHT,
           "scorer_version": SCORER_VERSION, "quick": QUICK}, open("rag_settings.json", "w"), indent=1)
print(f"Saved rag_results.csv ({len(scored)} rows) and rag_settings.json")
try:
    from google.colab import files
    for name in ("rag_results.csv", "rag_settings.json", "rag_checkpoint.json"):
        files.download(name)
except Exception as exc:
    print("Auto-download unavailable; use the Colab file browser instead.", exc)
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": "05_rag_knowledge_arms.ipynb", "gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT_NB}")
