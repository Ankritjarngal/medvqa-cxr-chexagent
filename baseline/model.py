"""Thin wrapper around StanfordAIMI/CheXagent-2-3b for zero-shot inference.

This module needs `torch` + `transformers` + a GPU to actually run, so it is
kept separate from `data.py` / `categorize.py` / `evaluate.py` (all GPU-free)
and is only imported where a GPU is available -- in practice that means the
Colab notebook, not this dev sandbox. Everything in here mirrors the
inference pattern documented on the model card.
"""

from __future__ import annotations

MODEL_NAME = "StanfordAIMI/CheXagent-2-3b"

GENERAL_PROMPT = "{question}"
LATERALITY_SEVERITY_PROMPT = (
    "{question} Answer concisely based only on what is visible in the image."
)


class CheXagentRunner:
    """Loads CheXagent-2-3b once; call `.ask(image_path, question)` per item."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cuda",
                 dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).to(self.dtype)
        self.model.eval()

    def ask(self, image_paths: list[str], question: str, max_new_tokens: int = 256) -> str:
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
