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

# The model code only encodes the image when `past_key_values is None`. transformers >= 4.45 passes an empty
# DynamicCache on the first step instead of None, so with transformers 4.46.3 the image was silently skipped and the
# model answered from the text alone. This makes the condition also true for an empty cache.
_IMAGE_BRANCH_ORIGINAL = "if past_key_values is None and torch.any(input_ids == self.tokenizer.img_start_id):"
_IMAGE_BRANCH_PATCHED = (
    "if (past_key_values is None or (hasattr(past_key_values, 'get_seq_length') "
    "and past_key_values.get_seq_length() == 0)) and torch.any(input_ids == self.tokenizer.img_start_id):"
)


def patch_image_cache_check(model_dir: str) -> int:
    """Make the model encode the image on the first step whatever cache object transformers passes.
    Idempotent. Returns 1 when the patched condition is in place; raises if the model code is not as expected."""
    path = f"{model_dir}/modeling_chexagent.py"
    with open(path) as f:
        text = f.read()
    if _IMAGE_BRANCH_PATCHED in text:
        return 1
    if _IMAGE_BRANCH_ORIGINAL not in text:
        raise RuntimeError("modeling_chexagent.py does not contain the expected image-branch condition; "
                           "the image may not reach the model. Inspect the file before running anything.")
    with open(path, "w") as f:
        f.write(text.replace(_IMAGE_BRANCH_ORIGINAL, _IMAGE_BRANCH_PATCHED))
    return 1


def align_visual_dtype(model, dtype):
    """Give CheXagent's visual tower a single dtype.

    The tower registers a float32 position-embedding parameter directly on itself, and its `encode()` casts the input
    pixels to the dtype of the *first* parameter it finds, which is that one. The SigLIP convolution weights are float16,
    so with the image branch actually running, the pixels arrive as float32 and the first convolution fails with
    "Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same".
    Casting the whole tower to `dtype` removes the mismatch. Raises if the tower cannot be found or is still mixed."""
    towers = [m for m in model.modules() if hasattr(m, "encode") and hasattr(m, "attn_pool")]
    if len(towers) != 1:
        raise RuntimeError(f"expected one visual tower, found {len(towers)}; inspect the model code")
    tower = towers[0]
    tower.to(dtype=dtype)
    found = {p.dtype for p in tower.parameters()}
    if found != {dtype}:
        raise RuntimeError(f"visual tower still has mixed dtypes: {found}")
    return tower


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
        patch_image_cache_check(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", torch_dtype=self.dtype, trust_remote_code=True
        )
        align_visual_dtype(self.model, self.dtype)
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
