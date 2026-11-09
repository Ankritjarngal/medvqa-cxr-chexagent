"""Shared pieces for the Phase 2 notebook builders (notebooks 04 and 05)."""

import json

SRC_NB = "notebooks/01_baseline_chexagent.ipynb"


def inline_module(relpath):
    """Module source for a notebook cell: relative imports and __main__ blocks removed."""
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


# Needs `model`, `tokenizer`, `torch`, `DEVICE` from the model cells. generate() is the call that is known to
# work with this model's remote code, so the yes/no score reads the scores of one generated token.
YES_NO_CELL = r'''def _single_token_ids(words):
    ids = set()
    for w in words:
        try:
            t = tokenizer.encode(w, add_special_tokens=False)
        except TypeError:
            t = tokenizer.encode(w)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)

YES_IDS = _single_token_ids(["Yes", "yes", " Yes", " yes", "YES"])
NO_IDS = _single_token_ids(["No", "no", " No", " no", "NO"])
assert YES_IDS and NO_IDS, "could not find single-token ids for yes / no"
print("yes ids:", YES_IDS, "| no ids:", NO_IDS)


def yes_no_score(image_path, question):
    """log P(yes) - log P(no) for the first answer token (higher = more 'yes')."""
    query = tokenizer.from_list_format([{"image": image_path}, {"text": question}])
    conv = [
        {"from": "system", "value": "You are a helpful radiology assistant. Answer the question as concisely as possible based only on the image."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(input_ids.to(DEVICE), do_sample=False, num_beams=1, use_cache=True,
                             max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    logp = torch.log_softmax(out.scores[0][0].float(), dim=-1)
    return float(torch.logsumexp(logp[YES_IDS], 0) - torch.logsumexp(logp[NO_IDS], 0))
'''

GUESS_CELL = r'''def guess_disease(image_path, method):
    """Disease names the model gives for one image, most likely first, plus the raw text or scores.
    method: 'open' | 'forced' (free text, parsed) | 'perclass' (yes-minus-no score per candidate)."""
    if method in ("open", "forced"):
        text = ask_chexagent(image_path, DISEASE_PROMPTS[method], 48)
        return parse_diseases(text), text
    scores = {name: yes_no_score(image_path, perclass_question(name)) for name in CANDIDATES}
    return sorted(scores, key=lambda c: -scores[c]), scores
'''
