"""Thin wrapper around StanfordAIMI/CheXagent-2-3b for zero-shot inference.

This module needs `torch` + `transformers` + a GPU to actually run, so it is
kept separate from `data.py` / `categorize.py` / `evaluate.py` (all GPU-free)
and is only imported where a GPU is available -- in practice that means the
Colab notebook, not this dev sandbox. Everything in here mirrors the
inference pattern documented on the model card.
"""

from __future__ import annotations

import glob
import re

MODEL_NAME = "StanfordAIMI/CheXagent-2-3b"

# CheXagent's remote code hard-codes `assert transformers.__version__ == "4.40.0"`.
# That version cannot be installed on current Colab Pythons (no prebuilt
# `tokenizers` 0.19 wheel), and transformers > 4.46 breaks the model's generate
# code. We run on 4.46.3 and neutralize those two asserts in a local copy.
_VERSION_ASSERT = re.compile(r'^assert transformers\.__version__ == "4\.40\.0".*$', flags=re.M)


def patch_version_asserts(model_dir: str) -> int:
    """Replace the hard-coded transformers-version asserts with `pass`.
    Returns the number of lines patched (2 for the released model code)."""
    patched = 0
    for path in glob.glob(f"{model_dir}/*.py"):
        with open(path) as f:
            text = f.read()
        new_text, n = _VERSION_ASSERT.subn("pass", text)
        if n:
            with open(path, "w") as f:
                f.write(new_text)
            patched += n
    return patched

GENERAL_PROMPT = "{question}"
LATERALITY_SEVERITY_PROMPT = (
    "{question} Answer concisely based only on what is visible in the image."
)


class CheXagentRunner:
    """Loads CheXagent-2-3b once; call `.ask(image_path, question)` per item."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cuda",
                 dtype=None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        # T4 (compute capability 7.5) has no native bfloat16; use float16 there.
        if dtype is None:
            dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
                     else torch.float16)
        self.dtype = dtype

        model_dir = snapshot_download(model_name, local_dir=model_name.split("/")[-1])
        patch_version_asserts(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", torch_dtype=self.dtype, trust_remote_code=True
        )
        self.model.eval()

    def ask(self, image_paths: list[str], question: str, max_new_tokens: int = 128) -> str:
        query = self.tokenizer.from_list_format(
            [*[{"image": p} for p in image_paths], {"text": question}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful radiology assistant. Answer the question as concisely as possible based only on the image."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids.to(self.device),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )[0]
        response = self.tokenizer.decode(output[input_ids.size(1):-1])
        return response.strip()


def prompt_for_category(question: str, category: str) -> str:
    """Category-aware prompt wrapping. Phase 1 deliberately keeps this
    minimal (no few-shot examples, no retrieval) -- that is the whole point
    of a *baseline*. Phase 2 is where retrieved context gets added."""
    if category in ("laterality", "severity"):
        return LATERALITY_SEVERITY_PROMPT.format(question=question)
    return GENERAL_PROMPT.format(question=question)
