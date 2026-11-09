"""Execute a notebook's code cells on this machine with a stand-in model (no GPU, no download of the model).

  python3 scripts/stub_run_notebook.py notebooks/04_disease_naming.ipynb constant|reader

The install, model-download, `ask_chexagent`, `yes_no_score` cells are skipped; fakes are injected before the first cell
that defines `guess_disease` (04) or `run_arm` (05). `constant` models a system that ignores the image; `reader` models
one that reads the label (04) or the true disease (05) from a hidden table. Run it in an empty scratch directory with
PYTHONPATH set to the repo root and MPLBACKEND=Agg. It checks the notebook's plumbing and reports, not CheXagent.
"""
import hashlib
import json
import os
import sys

nb_path, mode = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
SKIP = ("pip install", "snapshot_download(MODEL_NAME", "def ask_chexagent", "def _single_token_ids")

FAKE = '''
import hashlib as _h
def _bucket(s, m): return int(_h.md5(s.encode()).hexdigest(), 16) % m

def _truth(image_path):
    for it in globals().get("items", []) + globals().get("know_items", []):
        if it["image_path"] == image_path:
            return it.get("labels") or ([it["oracle_disease"]] if it.get("oracle_disease") else [])
    return []

def ask_chexagent(image_path, question, max_new_tokens=128):
    if "What disease is shown" in question or "Which of these is present" in question:
        t = _truth(image_path)
        return (t[0] if (MODE == "reader" and t) else "Pneumonia")
    if "Reference:" in question or "The finding in this image" in question:
        return "Antibiotics, rest, fluids"
    return "Medical treatment, surgical treatment"

def yes_no_score(image_path, question):
    t = _truth(image_path)
    if MODE == "reader" and t and any(n in question.lower() for n in [x.lower() for x in t]):
        return 4.0
    return float(_bucket(image_path + question, 100)) / 100.0 - 1.0 if MODE == "reader" else 0.0
'''
ns = {"__name__": "__main__", "os": os, "MODE": mode}
sys.argv = ["ipykernel_launcher"]
injected = False
for src in cells:
    src = src.replace("DISEASE_METHOD = None ", "DISEASE_METHOD = os.environ.get('STUB_METHOD', 'forced') ")
    if any(m in src for m in SKIP) and "def guess_disease" not in src:
        continue
    if not injected and ("def guess_disease" in src or "def run_arm" in src):
        exec(compile(FAKE, "fake", "exec"), ns)
        injected = True
    exec(compile(src, "nbcell", "exec"), ns)
