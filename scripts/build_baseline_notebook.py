"""Generates notebooks/01_baseline_chexagent.ipynb.

The notebook is intentionally self-contained (every function it needs is
inlined in its own cells) rather than `git clone`-ing this repo, because this
repo lives on a private Origin-backed remote that a fresh Colab runtime can't
authenticate to. The logic inlined here mirrors (and was developed against)
`baseline/categorize.py`, `baseline/data.py`, and `baseline/evaluate.py` in
this repo -- if you change the taxonomy or scoring there, re-sync this file.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(r"""# Phase 1 — Baseline testing: CheXagent-2-3b on chest X-ray VQA

**Goal of this notebook:** run the pretrained, open-source `StanfordAIMI/CheXagent-2-3b`
model zero-shot (no training, no retrieval, no modification) on a small chest-X-ray
question set, and check whether it shows the same weak spot its own paper reports:
strong on general/coarse questions (presence, modality, organ), weaker on fine-grained
**laterality** ("which side") and **severity** ("mild vs. severe") questions.

This is Phase 1 of 3:
1. **Baseline testing** (this notebook) — measure where the off-the-shelf model is weak.
2. **RAG layer** — add knowledge-grounded retrieval targeted at the weak spot(s) found here.
3. **Fine-tuning (if needed)** — only if RAG alone doesn't close the gap.

**Important caveat, stated up front, not hidden:** CheXagent's own training corpus
(CheXinstruct) explicitly includes VQA-RAD, SLAKE, and Open-I among its 32 source
datasets. So this is *not* a clean, held-out, zero-shot generalization test — it's close
to in-distribution. That's normal for this field (every paper we reviewed reuses the same
public benchmarks for the same reason). The point here isn't to claim a novel accuracy
number; it's to get our *own* measured numbers, on data we control end-to-end, for the
specific failure mode we plan to target with RAG in Phase 2.

**Runtime:** Free Google Colab, **T4 GPU** (Runtime → Change runtime type → T4 GPU).
Everything below — model download, inference, evaluation — fits comfortably in the
free tier's 16GB VRAM and disk quota.
""")

md("## 1. Setup")
md("""CheXagent ships its own model code, written for `transformers==4.40.0`. That exact version
cannot be installed on Colab's current Python (its `tokenizers` dependency has no prebuilt
package there), and versions newer than 4.46 break the model's `generate` code. So we use
**`transformers==4.46.3`** and, in section 3, patch the two hard-coded version checks in a local
copy of the model code. This combination was tested end-to-end (tokenizer, image handling,
model build, `generate`) on a small random-weight copy of the model before shipping.

If Colab asks you to **restart the runtime** after this cell, do so, then continue from the next
cell (do not re-run the install cell).
""")
code(r"""#@title Install dependencies (torch is preinstalled on Colab)
!pip install -q "transformers==4.46.3" accelerate einops sentencepiece protobuf pyarrow albumentations huggingface_hub
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("WARNING: no GPU detected. Go to Runtime -> Change runtime type -> T4 GPU, then re-run this cell.")
""")

md("""## 2. Build the chest X-ray test set

Two sources, combined:
- **SLAKE** — has structured `location` + `modality` metadata, so chest X-ray rows are
  filtered precisely (`modality == "X-Ray"` and `location` in the chest/lung/heart labels).
- **VQA-RAD** — has no organ metadata, so chest rows are selected with a keyword heuristic
  (noisier — flagged here, not hidden). Used as a secondary source to backfill categories
  (especially "severity", which SLAKE's test split barely covers).

Every question is tagged `general` / `laterality` / `severity` by simple keyword rules
(see the `categorize_question` cell below) so results can be broken down by category
instead of reported as one aggregate number.
""")

code(r'''import re

LATERALITY_PATTERNS = [
    r"\bleft\b", r"\bright\b", r"\bbilateral\b", r"\bwhich side\b",
    r"\blaterality\b", r"\bunilateral\b",
]
SEVERITY_PATTERNS = [
    r"\bmild\b", r"\bmoderate\b", r"\bsevere\b", r"\benlarge(d|ment)?\b",
    r"\bdegree\b", r"\bextent\b", r"\bhow (much|severe|large|big)\b",
    r"\bsize of\b", r"\bworsen(ed|ing)?\b", r"\bimprov(ed|ing|ement)\b",
    r"\bsmall\b", r"\blarge\b",
]

def categorize_question(question: str) -> str:
    """Laterality checked first: it is the harder, more specifically-flagged
    failure mode in CheXagent's own reported numbers, so a question with both
    kinds of words ("is the left-sided effusion severe?") is tagged laterality."""
    q = question.lower().strip()
    if any(re.search(p, q) for p in LATERALITY_PATTERNS):
        return "laterality"
    if any(re.search(p, q) for p in SEVERITY_PATTERNS):
        return "severity"
    return "general"
''')

code(r'''import hashlib, time, zipfile
from pathlib import Path
import requests

SLAKE_DATASET = "BoKelvin/SLAKE"
SLAKE_IMAGES_ZIP_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip"
VQA_RAD_DATASET = "flaviagiammarino/vqa-rad"
DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"

SLAKE_CHEST_LOCATIONS = {"Chest_heart", "Chest_lung", "Chest_mediastinal", "Lung"}

VQA_RAD_CHEST_KEYWORDS = {
    "lung", "lungs", "chest", "pulmonary", "pleural", "pneumothorax", "cardiac",
    "heart", "mediastin", "rib", "costophrenic", "consolidation", "effusion",
    "atelectasis", "diaphragm", "thora", "cardiomegaly", "pneumonia", "hilar", "bronch",
}
VQA_RAD_EXCLUDE_KEYWORDS = {
    "brain", "skull", "head", "abdomen", "abdominal", "liver", "kidney", "pelvis",
    "pelvic", "spine", "spinal", "bowel", "intestine", "stomach", "bladder", "uterus",
    "ovary", "appendix", "cerebral", "cranial", "sinus",
}


def _get_rows(dataset, split, offset, length=100, retries=5, backoff=1.5):
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(DATASETS_SERVER_ROWS_URL, params={
                "dataset": dataset, "config": "default", "split": split,
                "offset": offset, "length": length,
            }, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(backoff * (attempt + 1))
    if last_exc:
        raise last_exc
    resp.raise_for_status()


def _iter_all_rows(dataset, split, page_size=100):
    offset = 0
    while True:
        data = _get_rows(dataset, split, offset, length=page_size)
        rows = data.get("rows", [])
        if not rows:
            return
        for row in rows:
            yield row["row"]
        offset += len(rows)
        if offset >= data.get("num_rows_total", 0):
            return


def fetch_slake_chest_records(split="test"):
    records = []
    for rr in _iter_all_rows(SLAKE_DATASET, split):
        if (rr.get("modality") == "X-Ray" and rr.get("location") in SLAKE_CHEST_LOCATIONS
                and rr.get("q_lang") == "en"):
            records.append({
                "source_dataset": "SLAKE", "source_split": split,
                "question": rr["question"], "gold_answer": rr["answer"],
                "image_key": rr["img_name"], "category": categorize_question(rr["question"]),
            })
    return records


def fetch_vqa_rad_chest_records(splits=("test", "train")):
    records = []
    for split in splits:
        for rr in _iter_all_rows(VQA_RAD_DATASET, split):
            q = rr["question"].lower()
            if (any(k in q for k in VQA_RAD_CHEST_KEYWORDS)
                    and not any(k in q for k in VQA_RAD_EXCLUDE_KEYWORDS)):
                records.append({
                    "source_dataset": "VQA-RAD", "source_split": split,
                    "question": rr["question"], "gold_answer": rr["answer"],
                    "image_key": rr["image"]["src"], "category": categorize_question(rr["question"]),
                })
    return records


def stratified_sample(records, n_per_category, seed=0):
    import random
    rng = random.Random(seed)
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    sampled = []
    for items in by_cat.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:n_per_category])
    rng.shuffle(sampled)
    return sampled


def download_slake_images(records, dest_dir, zip_cache_path=None):
    dest_dir = Path(dest_dir); dest_dir.mkdir(parents=True, exist_ok=True)
    needed_keys = {r["image_key"] for r in records if r["source_dataset"] == "SLAKE"}
    if not needed_keys:
        return {}
    zip_cache_path = Path(zip_cache_path) if zip_cache_path else dest_dir / "_slake_imgs.zip"
    if not zip_cache_path.exists():
        resp = requests.get(SLAKE_IMAGES_ZIP_URL, stream=True, timeout=180)
        resp.raise_for_status()
        with open(zip_cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    local_paths = {}
    with zipfile.ZipFile(zip_cache_path) as zf:
        for key in needed_keys:
            out_path = dest_dir / key.replace("/", "__")
            if not out_path.exists():
                with zf.open(f"imgs/{key}") as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            local_paths[key] = str(out_path)
    return local_paths


def download_vqa_rad_images(records, dest_dir):
    dest_dir = Path(dest_dir); dest_dir.mkdir(parents=True, exist_ok=True)
    local_paths = {}
    for r in records:
        if r["source_dataset"] != "VQA-RAD":
            continue
        url = r["image_key"]
        if url in local_paths:
            continue
        digest = hashlib.md5(url.encode()).hexdigest()[:12]
        out_path = dest_dir / f"vqarad_{digest}.jpg"
        if not out_path.exists():
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
        local_paths[url] = str(out_path)
    return local_paths


def build_baseline_test_set(images_dir, n_per_category=20, seed=0):
    slake = fetch_slake_chest_records(split="test")
    vqa_rad = fetch_vqa_rad_chest_records(splits=("test", "train"))
    sampled = stratified_sample(slake + vqa_rad, n_per_category=n_per_category, seed=seed)

    slake_records = [r for r in sampled if r["source_dataset"] == "SLAKE"]
    vqa_rad_records = [r for r in sampled if r["source_dataset"] == "VQA-RAD"]
    slake_paths = download_slake_images(slake_records, images_dir)
    vqa_rad_paths = download_vqa_rad_images(vqa_rad_records, images_dir)

    for r in sampled:
        paths = slake_paths if r["source_dataset"] == "SLAKE" else vqa_rad_paths
        r["image_path"] = paths[r["image_key"]]
        digest = hashlib.md5((r["source_dataset"] + r["question"] + r["image_key"]).encode()).hexdigest()[:10]
        r["id"] = f"{r['source_dataset']}_{digest}"
    return sampled
''')

code(r'''# N_PER_CATEGORY questions per category (general / laterality / severity) -> up to 3x that many total.
# 20/category (~45-60 total after dedup) keeps the full run well under 10 minutes on a free T4.
N_PER_CATEGORY = 20

test_set = build_baseline_test_set(images_dir="images", n_per_category=N_PER_CATEGORY, seed=0)

from collections import Counter
print(f"Built {len(test_set)} questions")
print("By category:", Counter(r["category"] for r in test_set))
print("By source:  ", Counter(r["source_dataset"] for r in test_set))
''')

md("""## 3. Load CheXagent-2-3b

First run downloads the model from Hugging Face (not gated, MIT license, no token needed).
The repo stores the weights in float32, so the **download is ~12.6GB** (a few minutes; Colab has
the disk space). It is then loaded in half precision (`float16` on a T4, which has no native
`bfloat16`; `bfloat16` on newer GPUs), which fits a free T4's 16GB VRAM.

We download to a local folder first so we can patch the two `assert transformers.__version__ ==
"4.40.0"` lines in the model's own code (see section 1).
""")

code(r'''import glob, re
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "StanfordAIMI/CheXagent-2-3b"
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print("Using dtype:", DTYPE)

_VERSION_ASSERT = re.compile(r'^assert transformers\.__version__ == "4\.40\.0".*$', flags=re.M)

def patch_version_asserts(model_dir):
    """Neutralize the model code's hard-coded transformers==4.40.0 checks.
    Returns the number of lines patched (expected: 2)."""
    patched = 0
    for path in glob.glob(f"{model_dir}/*.py"):
        text = open(path).read()
        new_text, n = _VERSION_ASSERT.subn("pass", text)
        if n:
            open(path, "w").write(new_text)
            patched += n
    return patched

model_dir = snapshot_download(MODEL_NAME, local_dir="CheXagent-2-3b")
print("Patched version checks:", patch_version_asserts(model_dir))

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="auto", torch_dtype=DTYPE, trust_remote_code=True
)
model.eval()
print("Model loaded.")
''')

code(r'''GENERAL_PROMPT = "{question}"
LATERALITY_SEVERITY_PROMPT = "{question} Answer concisely based only on what is visible in the image."

def prompt_for_category(question, category):
    """Deliberately minimal (no few-shot examples, no retrieval) -- that's the
    whole point of a *baseline*. Phase 2 is where retrieved context gets added."""
    if category in ("laterality", "severity"):
        return LATERALITY_SEVERITY_PROMPT.format(question=question)
    return GENERAL_PROMPT.format(question=question)


def ask_chexagent(image_path, question, max_new_tokens=128):
    query = tokenizer.from_list_format([{"image": image_path}, {"text": question}])
    conv = [
        {"from": "system", "value": "You are a helpful radiology assistant. Answer the question as concisely as possible based only on the image."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids.to(DEVICE), do_sample=False, num_beams=1, use_cache=True,
            max_new_tokens=max_new_tokens,
        )[0]
    return tokenizer.decode(output[input_ids.size(1):-1]).strip()
''')

md("## 4. Run inference over the test set")
code(r'''from tqdm.auto import tqdm

for r in tqdm(test_set):
    prompt = prompt_for_category(r["question"], r["category"])
    r["prediction"] = ask_chexagent(r["image_path"], prompt)

print("Done. Example:")
print(test_set[0]["question"], "->", test_set[0]["prediction"])
''')

md("""## 5. Score predictions

Two automated scores are reported side by side, not blended into one number --
Med-Flamingo's own finding was that automated text-overlap metrics can *disagree sharply*
with what a clinician would call correct, so neither score here should be taken as the
final word (see the qualitative spot-check in section 6).

- `exact_match`: strict, normalized string equality. Fair for yes/no questions, harsh for
  open-ended ones (a correct but differently-worded answer scores 0).
- `loose_match`: substring containment after normalization. More forgiving for open-ended
  answers, more generous than it should be on a few easy cases.
""")

code(r'''import re as _re
from dataclasses import dataclass, field

_YES_SYNONYMS = {"yes", "y", "yeah", "affirmative", "true", "present"}
_NO_SYNONYMS = {"no", "n", "nope", "negative", "false", "absent", "not present"}
_PUNCT_RE = _re.compile(r"[^a-z0-9\s]")
_WS_RE = _re.compile(r"\s+")

def normalize(text):
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if text in _YES_SYNONYMS:
        return "yes"
    if text in _NO_SYNONYMS:
        return "no"
    return text

def exact_match(prediction, gold):
    return normalize(prediction) == normalize(gold)

def loose_match(prediction, gold):
    p, g = normalize(prediction), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
    if len(shorter) < 3:
        return False
    return shorter in longer


@dataclass
class CategoryStats:
    total: int = 0
    exact_correct: int = 0
    loose_correct: int = 0
    @property
    def exact_accuracy(self):
        return self.exact_correct / self.total if self.total else float("nan")
    @property
    def loose_accuracy(self):
        return self.loose_correct / self.total if self.total else float("nan")


per_category, per_source = {}, {}
overall = CategoryStats()
for r in test_set:
    em = exact_match(r["prediction"], r["gold_answer"])
    lm = loose_match(r["prediction"], r["gold_answer"])
    r["exact_match"], r["loose_match"] = em, lm
    for bucket, key in ((per_category, r["category"]), (per_source, r["source_dataset"])):
        s = bucket.setdefault(key, CategoryStats())
        s.total += 1; s.exact_correct += int(em); s.loose_correct += int(lm)
    overall.total += 1; overall.exact_correct += int(em); overall.loose_correct += int(lm)

print("=== Overall ===")
print(f"n={overall.total}  exact_acc={overall.exact_accuracy:.1%}  loose_acc={overall.loose_accuracy:.1%}\n")
print("=== By question category (THE key breakdown -- this is what Phase 2 should move) ===")
for cat, s in sorted(per_category.items()):
    print(f"{cat:>12}: n={s.total:<4} exact_acc={s.exact_accuracy:.1%}  loose_acc={s.loose_accuracy:.1%}")
print("\n=== By source dataset ===")
for src, s in sorted(per_source.items()):
    print(f"{src:>10}: n={s.total:<4} exact_acc={s.exact_accuracy:.1%}  loose_acc={s.loose_accuracy:.1%}")
''')

md("""## 6. Qualitative spot-check (don't skip this)

Med-Flamingo's paper found automated metrics can diverge sharply from what a clinician
would actually call correct. Before trusting the numbers above, eyeball a handful of
cases yourself -- especially any `laterality`/`severity` questions marked wrong.
""")

code(r'''import random
from PIL import Image
import matplotlib.pyplot as plt

sample_for_review = random.Random(1).sample(test_set, min(6, len(test_set)))

for r in sample_for_review:
    img = Image.open(r["image_path"]).convert("RGB")
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.title(r["category"])
    plt.show()
    print(f"Q: {r['question']}")
    print(f"Gold: {r['gold_answer']!r}   Pred: {r['prediction']!r}")
    print(f"exact_match={r['exact_match']}  loose_match={r['loose_match']}")
    print("-" * 60)
''')

md("## 7. Save results")
code(r'''import json, csv

with open("baseline_results.jsonl", "w") as f:
    for r in test_set:
        f.write(json.dumps(r) + "\n")

fieldnames = ["id", "source_dataset", "source_split", "category", "image_path",
              "question", "gold_answer", "prediction", "exact_match", "loose_match"]
with open("baseline_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in test_set:
        writer.writerow({k: r.get(k) for k in fieldnames})

print("Saved baseline_results.jsonl and baseline_results.csv")

try:
    from google.colab import files
    files.download("baseline_results.csv")
    files.download("baseline_results.jsonl")
except Exception as exc:
    print("Auto-download unavailable; use the Colab file browser (left sidebar) instead.", exc)
print("Keep these files: the Colab VM is wiped when the session ends.")
''')

md("""## What's next (Phase 2)

Look at the category breakdown from section 5:
- If `laterality` and/or `severity` accuracy is meaningfully lower than `general`
  (the pattern CheXagent's own paper reports — ~77-79% vs ~90%+), that confirms the target
  for the RAG layer: retrieve **authoritative finding definitions / diagnostic criteria**
  (not just similar patients' reports, which is what RULE's case-based retrieval does) for
  exactly the laterality/severity questions this baseline got wrong.
- Keep `baseline_results.csv` — Phase 2's RAG-augmented run will be scored the same way and
  compared row-by-row against this baseline, specifically on the categories where this
  baseline was weak.
""")

nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": "01_baseline_chexagent.ipynb", "gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("notebooks/01_baseline_chexagent.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote notebooks/01_baseline_chexagent.ipynb")
