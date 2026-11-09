"""Disease labels for SLAKE chest X-rays, and scoring for "can the model name the disease?".

Why this exists (docs/phase2_rag_plan.md): SLAKE's knowledge answers (cause, symptoms, treatment,
prevention) are fixed by the disease on the image, and the question text never names it. SLAKE
states the disease in a separate question on the same image ("What diseases are included in the
picture?"). Those answers are used here for two things only:

1. Scoring a disease-naming test (Phase 1d): the label is the answer key.
2. "Oracle" arms in Phase 2: the true disease is handed to the system to measure an upper bound.
   Oracle arms are never counted as the project's result (protocol rule 3, Amendment 5).

The candidate vocabulary shown to the model is the public ChestX-ray14 pathology list, not the
list of diseases found in SLAKE.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from .evaluate import wilson_interval
from .stats import binom_two_sided

CHESTXRAY14 = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis",
    "Hernia", "Infiltration", "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
]
NO_FINDING = "No Finding"
CANDIDATES = CHESTXRAY14 + [NO_FINDING]

# SLAKE spelling (lowercase) -> canonical name. Includes the dataset's typos ("Noudle", "Cardiomega").
SPELLING_MAP = {
    "pulmonary mass": "Mass", "mass": "Mass",
    "pulmonary nodule": "Nodule", "nodule": "Nodule", "noudle": "Nodule",
    "pleural effusion": "Effusion", "effusion": "Effusion",
    "pulmonary infiltration": "Infiltration", "infiltration": "Infiltration",
    "cardiomegaly": "Cardiomegaly", "cardiomegal": "Cardiomegaly", "cardiomega": "Cardiomegaly",
    "atelectasis": "Atelectasis", "pneumonia": "Pneumonia", "pneumothorax": "Pneumothorax",
}

# Phrases in free-text model answers that count as naming a disease.
_PARSE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Cardiomegaly", re.compile(r"cardiomegaly|enlarged heart|heart (?:is )?enlarged|enlarged cardiac")),
    ("Atelectasis", re.compile(r"atelecta")),
    ("Consolidation", re.compile(r"consolidat")),
    ("Edema", re.compile(r"\bedema\b|\boedema\b")),
    ("Effusion", re.compile(r"effusion")),
    ("Emphysema", re.compile(r"emphysema")),
    ("Fibrosis", re.compile(r"fibrosis|fibrotic")),
    ("Hernia", re.compile(r"hernia")),
    ("Infiltration", re.compile(r"infiltrat")),
    ("Mass", re.compile(r"\bmass(?:es)?\b")),
    ("Nodule", re.compile(r"\bnodul")),
    ("Pleural Thickening", re.compile(r"pleural thickening")),
    ("Pneumonia", re.compile(r"pneumonia")),
    ("Pneumothorax", re.compile(r"pneumothorax")),
    (NO_FINDING, re.compile(r"no finding|no acute|no abnormal|normal chest|no disease|no evidence of")),
]

_DISEASE_Q = re.compile(r"^\s*what diseases?\b", re.IGNORECASE)
_SPLIT_LABELS = re.compile(r"[,;/]|\band\b")


def is_disease_naming_question(question: str) -> bool:
    return bool(_DISEASE_Q.match(question))


def canonical_labels(answer: str) -> tuple[str, ...]:
    """'Cardiomegaly, Infiltration, Effusion' -> ('Cardiomegaly', 'Effusion', 'Infiltration').
    Unknown names are kept (title-cased) so nothing is silently dropped."""
    out = set()
    for part in _SPLIT_LABELS.split(str(answer)):
        p = part.strip().lower()
        if p:
            out.add(SPELLING_MAP.get(p, p.title()))
    return tuple(sorted(out))


def build_image_labels(pool: list[dict]) -> dict[str, dict]:
    """image_key -> {'labels': tuple, 'split': str}, from SLAKE disease-naming questions.
    If an image has several such questions the labels are merged."""
    merged: dict[str, dict] = {}
    for r in pool:
        if r["source_dataset"] != "SLAKE" or not is_disease_naming_question(r["question"]):
            continue
        entry = merged.setdefault(r["image_key"], {"labels": set(), "split": r["source_split"]})
        entry["labels"].update(canonical_labels(r["gold_answer"]))
    return {k: {"labels": tuple(sorted(v["labels"])), "split": v["split"]} for k, v in merged.items()}


def disease_id_items(pool: list[dict]) -> list[dict]:
    """One item per labelled SLAKE chest image, ordered by image key. Needs `attach_images` afterwards."""
    labels = build_image_labels(pool)
    keys = {r["image_key"]: r for r in pool if r["source_dataset"] == "SLAKE" and r["image_key"] in labels}
    items = []
    for key in sorted(labels):
        r = keys[key]
        items.append({
            "id": f"DIS_{key.split('/')[0]}", "uid": f"DIS_{key.split('/')[0]}", "source_dataset": "SLAKE",
            "source_split": labels[key]["split"], "image_key": key, "question": "(disease naming)",
            "gold_answer": ", ".join(labels[key]["labels"]), "labels": list(labels[key]["labels"]),
        })
    return items


def attach_oracle_disease(knowledge_records: list[dict], pool: list[dict]) -> list[dict]:
    """Add `oracle_disease` (single canonical name, or None) to every knowledge record. Multi-label
    images, VQA-RAD rows and images without a disease question get None. Oracle use only."""
    labels = build_image_labels(pool)
    out = []
    for r in knowledge_records:
        lab = labels.get(r["image_key"]) if r["source_dataset"] == "SLAKE" else None
        out.append({**r, "oracle_disease": lab["labels"][0] if lab and len(lab["labels"]) == 1 else None})
    return out


def parse_diseases(text: str) -> list[str]:
    """Diseases named in a free-text answer, in order of first appearance."""
    t = str(text).lower()
    found = [(m.start(), name) for name, pat in _PARSE_PATTERNS if (m := pat.search(t))]
    return [name for _, name in sorted(found)]


# ------------------------------------------------------------------------ prompts

DISEASE_PROMPTS = {
    "open": "What disease is shown in this chest X-ray?",
    "forced": ("Which of these is present in this chest X-ray: " + ", ".join(CANDIDATES) + "? "
               "Answer with one name from the list."),
}
PERCLASS_PROMPT = "Is there {name} in this chest X-ray? Answer yes or no."


# ---------------------------------------------------------------------- scoring

def auc_one_vs_rest(scores: list[float], positives: list[bool]) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _any_match(pred: list[str], gold: list[str]) -> bool:
    return bool(set(pred) & set(gold))


def _strict_match(pred: list[str], gold: list[str]) -> bool:
    return len(pred) == 1 and set(pred) == set(gold)


def disease_report(rows: list[dict], n_shuffles: int = 2000, seed: int = 0) -> dict:
    """rows need: labels (gold list), pred (list of predicted names; first is the top choice),
    split. Reports accuracy on the top choice and with any named disease, against
    always-the-most-common-label and against shuffled labels."""
    n = len(rows)
    top = [r["pred"][:1] for r in rows]
    gold = [r["labels"] for r in rows]
    acc_top = sum(_any_match(p, g) for p, g in zip(top, gold))
    acc_any = sum(_any_match(r["pred"], r["labels"]) for r in rows)
    acc_strict = sum(_strict_match(r["pred"], r["labels"]) for r in rows)

    counts = Counter(l for g in gold for l in set(g))
    majority = counts.most_common(1)[0][0] if counts else None
    maj_acc = sum(majority in g for g in gold) if majority else 0

    rng = random.Random(seed)
    shuffled = []
    for _ in range(n_shuffles):
        perm = gold[:]
        rng.shuffle(perm)
        shuffled.append(sum(_any_match(p, g) for p, g in zip(top, perm)))
    shuffled.sort()

    per_class: dict[str, dict] = {}
    for c in sorted(counts):
        idx = [i for i, g in enumerate(gold) if c in g]
        per_class[c] = {"n": len(idx), "recall": sum(c in top[i] for i in idx) / len(idx),
                        "predicted": sum(c in t for t in top)}
    by_split = {}
    for sp in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == sp]
        by_split[sp] = {"n": len(sub), "acc_top": sum(_any_match(r["pred"][:1], r["labels"]) for r in sub)}
    return {
        "n": n, "acc_top": acc_top / n if n else float("nan"), "acc_top_ci": wilson_interval(acc_top, n),
        "acc_any": acc_any / n if n else float("nan"), "acc_strict": acc_strict / n if n else float("nan"),
        "unparsed": sum(not r["pred"] for r in rows),
        "majority_label": majority, "majority_acc": maj_acc / n if n else float("nan"),
        "p_vs_majority": binom_two_sided(acc_top, n, maj_acc / n) if n and maj_acc else float("nan"),
        "shuffled_mean": sum(shuffled) / len(shuffled) / n if n else float("nan"),
        "shuffled_p95": shuffled[int(0.95 * len(shuffled)) - 1] / n if n else float("nan"),
        "per_class": per_class, "by_split": by_split,
    }


def format_disease_report(by_method: dict[str, dict]) -> str:
    L = ["=== Disease naming (top choice counts as correct if it matches any gold label) ===",
         f"{'method':>16} {'n':>4} {'acc':>6} {'95% CI':>11} {'any named':>9} {'strict':>7} {'majority':>9} "
         f"{'shuffled':>9} {'p vs maj':>9} {'unparsed':>8}"]
    for name, r in by_method.items():
        lo, hi = r["acc_top_ci"]
        L.append(f"{name:>16} {r['n']:>4} {r['acc_top']:>6.0%} {lo:>5.0%}-{hi:<5.0%} {r['acc_any']:>9.0%} "
                 f"{r['acc_strict']:>7.0%} {r['majority_acc']:>9.0%} {r['shuffled_mean']:>9.0%} "
                 f"{r['p_vs_majority']:>9.3f} {r['unparsed']:>8}")
    L.append("majority = always answering the most common label; shuffled = same answers against shuffled gold labels.")
    for name, r in by_method.items():
        L.append(f"\n[{name}] per class (n, recall, times predicted):")
        L.append("   " + "; ".join(f"{c} {v['n']}, {v['recall']:.0%}, {v['predicted']}" for c, v in r["per_class"].items()))
        L.append("   by split: " + "; ".join(f"{sp} {v['acc_top']}/{v['n']}" for sp, v in r["by_split"].items()))
    return "\n".join(L)


def choose_method(by_method_validation: dict[str, dict]) -> str:
    """The method used downstream: highest accuracy on the validation images; ties go to the
    order of the dict (open, forced, perclass), so the choice is fixed before looking at test rows."""
    return max(by_method_validation, key=lambda m: by_method_validation[m]["acc_top"])


def perclass_predictions(scores: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """scores: image id -> {class: yes-minus-no logit}. Predicted list = classes ordered by score."""
    return {i: sorted(s, key=lambda c: -s[c]) for i, s in scores.items()}


def class_table(scores: dict[str, dict[str, float]], labels: dict[str, list[str]]) -> dict[str, float]:
    """One-vs-rest AUC per class over all images that have scores."""
    ids = sorted(scores)
    out = {}
    for c in sorted({c for s in scores.values() for c in s}):
        pos = [c in labels[i] for i in ids]
        out[c] = auc_one_vs_rest([scores[i][c] for i in ids], pos)
    return out


def group_by_label(items: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for it in items:
        for l in it["labels"]:
            g[l].append(it)
    return dict(g)
