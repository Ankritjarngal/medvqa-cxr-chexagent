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


def inline_module(relpath):
    """Return the source of a baseline/ module with package-relative imports
    stripped, so the notebook runs the exact same tested code as the repo."""
    keep = []
    for line in open(relpath).read().splitlines():
        if line.startswith("from __future__") or line.startswith("from ."):
            continue
        keep.append(line)
    return "\n".join(keep).strip() + "\n"


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
- **SLAKE** (test + validation) — has structured `location` + `modality` metadata, so chest
  X-ray rows are filtered precisely.
- **VQA-RAD** — has no organ metadata, so chest rows are selected with a keyword heuristic
  (noisier — flagged here, not hidden; a few non-chest items may still slip through, which is
  why section 6 shows images of wrong answers for you to check).

Every question is tagged by simple keyword rules (the next cell shows them):
- `category`: `laterality` / `severity` / `general`
- `answer_format`: `yes/no` vs `open`
- `is_knowledge`: answered from medical knowledge (symptoms, treatment...) vs by looking at the image

Changes after the first pilot run (60 questions): the radiation-damage template question is
dropped (it is not clinical severity), identical question text is capped at 4 per category, more
questions are sampled, and question IDs are stable so Phase 2 can be compared row by row.

The two cells below are the repo's own `baseline/categorize.py` and `baseline/data.py`,
inlined unchanged (they are unit-tested in the repo).
""")

code(inline_module('baseline/categorize.py'))

code(inline_module('baseline/data.py'))

code(r'''# Up to N_PER_CATEGORY questions per category (general / laterality / severity).
# Severity questions are rare in public CXR VQA sets, so that bucket may come out smaller than
# the others -- the printed pool summary shows exactly how many exist.
N_PER_CATEGORY = 100

test_set = build_baseline_test_set(images_dir="images", n_per_category=N_PER_CATEGORY, seed=0)

from collections import Counter
print(f"Built {len(test_set)} questions")
print("By category:     ", Counter(r["category"] for r in test_set))
print("By answer format:", Counter(r["answer_format"] for r in test_set))
print("By source:       ", Counter(r["source_dataset"] for r in test_set))
print("Knowledge-type:  ", sum(r["is_knowledge"] for r in test_set))
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

_IMAGE_BRANCH_ORIGINAL = "if past_key_values is None and torch.any(input_ids == self.tokenizer.img_start_id):"
_IMAGE_BRANCH_PATCHED = (
    "if (past_key_values is None or (hasattr(past_key_values, 'get_seq_length') "
    "and past_key_values.get_seq_length() == 0)) and torch.any(input_ids == self.tokenizer.img_start_id):"
)

def patch_image_cache_check(model_dir):
    """transformers >= 4.45 passes an empty DynamicCache (not None) on the first generation step, and the model
    code only encodes the image when past_key_values is None. Without this patch the image is silently skipped."""
    path = f"{model_dir}/modeling_chexagent.py"
    text = open(path).read()
    if _IMAGE_BRANCH_PATCHED in text:
        return 1
    if _IMAGE_BRANCH_ORIGINAL not in text:
        raise RuntimeError("modeling_chexagent.py does not contain the expected image-branch condition")
    open(path, "w").write(text.replace(_IMAGE_BRANCH_ORIGINAL, _IMAGE_BRANCH_PATCHED))
    return 1

model_dir = snapshot_download(MODEL_NAME, local_dir="CheXagent-2-3b")
print("Patched version checks:", patch_version_asserts(model_dir))
print("Image-branch patch in place:", patch_image_cache_check(model_dir))

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

code(r'''# IMAGE CHECK: proves the picture reaches the model. The same file path is used with three different pixel patterns,
# so the text the model reads is identical every time; only the pixels differ. If the first-token scores do not change,
# the image is being ignored and nothing below this cell can be trusted.
import numpy as np
from PIL import Image

def _first_token_logprobs(path):
    query = tokenizer.from_list_format([{"image": path}, {"text": "Describe the image."}])
    conv = [
        {"from": "system", "value": "You are a helpful radiology assistant. Answer the question as concisely as possible based only on the image."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(input_ids.to(DEVICE), do_sample=False, num_beams=1, use_cache=True,
                             max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    return torch.log_softmax(out.scores[0][0].float(), dim=-1).cpu(), int(input_ids.size(1))

_rng = np.random.default_rng(0)
_variants = {"black": np.zeros((512, 512, 3), np.uint8), "white": np.full((512, 512, 3), 255, np.uint8),
             "noise": _rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)}
_lp = {}
for _name, _pix in _variants.items():
    Image.fromarray(_pix).save("image_check.png")
    _lp[_name], _n_tokens = _first_token_logprobs("image_check.png")
_diffs = {f"{a} vs {b}": float((_lp[a] - _lp[b]).abs().max()) for a, b in (("black", "white"), ("black", "noise"), ("white", "noise"))}
print("prompt length with image slots:", _n_tokens, "tokens (context limit 2048)")
print("largest change in next-token log-probability when only the pixels change:", {k: round(v, 3) for k, v in _diffs.items()})
if max(_diffs.values()) < 0.01:
    raise RuntimeError("IMAGE CHECK FAILED: changing the pixels does not change the model's output, so the image is not "
                       "reaching the model. Stop and report this.")
print("IMAGE CHECK PASSED: the model's output depends on the pixels.")
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

Several automated scores are reported side by side, not blended into one number --
Med-Flamingo's own finding was that automated text-overlap metrics can *disagree sharply*
with what a clinician would call correct, so none of them is the final word (see section 6).

- `exact`: strict normalized equality. Fair for yes/no, harsh for open answers.
- `loose`: substring containment either way.
- `token_f1`: word overlap, gives partial credit to open answers.
- Every accuracy has a **95% confidence interval**. With a few dozen questions per bucket,
  intervals that overlap mean the difference between buckets is not proven.
- **Yes/no block:** sensitivity (correct when the finding IS present) and specificity (correct
  when it is NOT), plus the accuracy of always answering the majority class. A model that says
  "no" to almost everything looks fine on accuracy but misses real findings.

The next cell is the repo's own `baseline/evaluate.py`, inlined unchanged.
""")

code(inline_module('baseline/evaluate.py'))

code(r'''report = evaluate(test_set)
test_set = report.rows   # same records, now with exact_match / loose_match / token_f1
print(format_report(report))
''')

md("""## 6. Qualitative spot-check (don't skip this)

Automated metrics can diverge from what a clinician would call correct, and the VQA-RAD chest
filter is a keyword heuristic. So look at the **wrong** answers yourself: is the image really a
chest X-ray? Is the model's answer actually wrong, or just worded differently? Laterality and
open-ended mistakes are shown first.
""")

code(r'''import random
from PIL import Image
import matplotlib.pyplot as plt

wrong = [r for r in test_set if not r["exact_match"]]
wrong.sort(key=lambda r: (r["category"] != "laterality", r["answer_format"] != "open"))
to_review = wrong[:12]
print(f"{len(wrong)} wrong answers in total; showing {len(to_review)}")

for r in to_review:
    img = Image.open(r["image_path"]).convert("RGB")
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"{r['category']} / {r['answer_format']} / {r['source_dataset']}", fontsize=8)
    plt.show()
    print(f"Q: {r['question']}")
    print(f"Gold: {r['gold_answer']!r}   Pred: {r['prediction']!r}   token_f1={r['token_f1']}")
    print("-" * 60)
''')

md("## 7. Save results")
code(r'''import json, csv

with open("baseline_results.jsonl", "w") as f:
    for r in test_set:
        f.write(json.dumps({k: v for k, v in r.items() if k != "image_key"}) + "\n")

fieldnames = ["id", "source_dataset", "source_split", "category", "answer_format", "is_knowledge",
              "image_path", "question", "gold_answer", "prediction", "exact_match", "loose_match",
              "token_f1"]
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

Read the report in section 5 with these questions:
1. **Which buckets are lowest, and do their confidence intervals really separate from the others?**
   (the paper's weak spot is fine-grained laterality/severity)
2. **Yes/no block:** is sensitivity far below specificity? That means the model under-calls findings,
   which retrieved text cannot fix because the error is in what the model *sees*.
3. **Knowledge vs perception:** errors on `knowledge` questions (symptoms, treatment, how a finding is
   measured) are what a retrieval layer can help with; perception errors point to confidence scoring
   or targeted fine-tuning instead.

Keep `baseline_results.csv`. Phase 2 results will be scored by the same code and compared row by
row (IDs are stable) against this baseline.
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
