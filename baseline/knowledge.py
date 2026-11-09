"""Scoring and review tooling for knowledge-type questions (symptoms, treatment,
prevention, cause, function, 'why'/'how can you tell').

Why this exists: on these questions the gold answer is usually a short list
("Chest pain, dyspnea"), and exact match scores a correct paraphrase
("Chest tightness, shortness of breath" -> shares 'dyspnea') as 0 while
token F1 can reward unrelated shared words. Two additions:

1. `item_scores`: splits the gold and predicted answers into items, maps a small
   hand-written table of synonyms onto one spelling, and measures how many gold
   items the prediction covers (recall) and how many predicted items are
   supported by the gold (precision). The synonym table is a convenience, not
   a clinically validated ontology. It is short enough to read in full.
2. A blind manual review sheet: baseline and retrieval answers are shuffled
   into columns the reviewer cannot attribute to a system, rated 0/1/2, and
   unblinded afterwards. Automated scores are a screening tool; this is the
   check that decides whether an improvement is real.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass

from .evaluate import normalize, token_f1
from .stats import binom_two_sided

# canonical spelling -> variants that mean the same thing (all lowercase).
CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "dyspnea": ["shortness of breath", "short of breath", "difficulty breathing",
                "difficulty in breathing", "breathing difficulty", "breathlessness", "dyspnoea"],
    "hemoptysis": ["coughing up blood", "coughing blood", "bloody sputum", "blood in sputum"],
    "cough": ["coughing"],
    "chest pain": ["chest discomfort"],
    "expectoration": ["sputum production", "phlegm"],
    "fatigue": ["tiredness"],
    "fever": ["pyrexia"],
    "medication": ["medical treatment", "medical therapy", "drug therapy", "drug treatment", "medicine"],
    "surgery": ["surgical treatment", "surgical therapy", "operation", "surgical"],
}

_STOPWORDS = {"the", "a", "an", "of", "in", "on", "and", "to", "is", "are", "with", "by",
              "for", "can", "be", "it", "this", "that", "or", "as", "at"}

_VARIANT_TO_CANON: list[tuple[re.Pattern, str]] = sorted(
    ((re.compile(rf"\b{re.escape(v)}\b"), canon)
     for canon, variants in CONCEPT_SYNONYMS.items() for v in variants),
    key=lambda item: -len(item[0].pattern),
)

_SPLIT_RE = re.compile(r"[,;/]|\band\b|\bor\b", re.IGNORECASE)

SCORER_VERSION = 2  # 1: no stemming (used for the first knowledge-set report); 2: light stemming

RATING_RUBRIC = (
    "2 = correct (same meaning as the gold answer, wording may differ)\n"
    "1 = partly correct (some of the gold content, or correct but incomplete / with wrong extras)\n"
    "0 = wrong, unrelated, contradicts the gold answer, or does not answer the question"
)


def canonicalize(text: str) -> str:
    t = normalize(str(text))
    for pattern, canon in _VARIANT_TO_CANON:
        t = pattern.sub(canon, t)
    return t


def _stem(word: str) -> str:
    """Very light suffix stripping so that 'keeping' matches 'keep' and 'lungs'
    matches 'lung'. Added in scorer version 2 after 'keeping healthy' scored 0
    against the gold 'Keep healthy'. Deliberately crude; applied identically to
    gold and predicted text."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in canonicalize(text).split() if w not in _STOPWORDS)


def split_items(answer: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(str(answer)) if p.strip()]


def _items_match(a: frozenset[str], b: frozenset[str]) -> bool:
    return bool(a) and bool(b) and (a <= b or b <= a)


@dataclass(frozen=True)
class ItemScores:
    recall: float
    precision: float
    f1: float


def item_scores(prediction: str, gold: str) -> ItemScores:
    """Item-level recall / precision / F1 after synonym canonicalization."""
    gold_items = [content_tokens(i) for i in split_items(gold)]
    pred_items = [content_tokens(i) for i in split_items(prediction)]
    gold_all, pred_all = content_tokens(gold), content_tokens(prediction)
    gold_items, pred_items = [g for g in gold_items if g], [p for p in pred_items if p]
    if not gold_items or not pred_items:
        return ItemScores(0.0, 0.0, 0.0)

    covered = sum(1 for g in gold_items if g <= pred_all or any(_items_match(g, p) for p in pred_items))
    supported = sum(1 for p in pred_items if p <= gold_all or any(_items_match(p, g) for g in gold_items))
    recall, precision = covered / len(gold_items), supported / len(pred_items)
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    return ItemScores(recall, precision, f1)


def score_knowledge_rows(rows: list[dict]) -> list[dict]:
    """Add item_recall / item_precision / item_f1 / token_f1 to each row."""
    out = []
    for r in rows:
        s = item_scores(r["prediction"], r["gold_answer"])
        out.append({**r, "item_recall": round(s.recall, 3), "item_precision": round(s.precision, 3),
                    "item_f1": round(s.f1, 3), "token_f1": round(token_f1(r["prediction"], r["gold_answer"]), 3)})
    return out


def summarize_knowledge(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean scores overall and per knowledge_subtype for rows from `score_knowledge_rows`."""
    def mean(rs, key):
        return sum(float(r[key]) for r in rs) / len(rs) if rs else float("nan")

    groups: dict[str, list[dict]] = {"all": rows}
    for r in rows:
        groups.setdefault(r.get("knowledge_subtype", "unknown"), []).append(r)
    return {name: {"n": len(rs), "item_recall": mean(rs, "item_recall"), "item_f1": mean(rs, "item_f1"),
                   "token_f1": mean(rs, "token_f1"), "any_item_hit": sum(float(r["item_recall"]) > 0 for r in rs) / len(rs)}
            for name, rs in groups.items()}


def format_knowledge_summary(summary: dict[str, dict[str, float]]) -> str:
    L = ["=== Knowledge questions: item-level scores ===",
         f"{'subtype':>14} {'n':>4}  {'item_recall':>11}  {'item_f1':>7}  {'token_f1':>8}  {'>=1 item right':>14}"]
    for name, s in sorted(summary.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        L.append(f"{name:>14} {s['n']:>4}  {s['item_recall']:>11.2f}  {s['item_f1']:>7.2f}  "
                 f"{s['token_f1']:>8.2f}  {s['any_item_hit']:>14.0%}")
    return "\n".join(L)


def make_blind_review(baseline_rows: list[dict], candidate_rows: list[dict], seed: int = 0,
                      candidate_name: str = "candidate") -> tuple[list[dict], list[dict]]:
    """Pair two systems' answers by id, randomize which one appears as
    answer_1, and return (sheet, key). Give the reviewer the sheet only; keep
    the key until all ratings are filled in."""
    cand = {r["id"]: r for r in candidate_rows}
    rng = random.Random(seed)
    sheet, key = [], []
    for b in baseline_rows:
        c = cand.get(b["id"])
        if c is None:
            continue
        flip = rng.random() < 0.5
        first, second = (c, b) if flip else (b, c)
        sheet.append({"id": b["id"], "question": b["question"], "gold_answer": b["gold_answer"],
                      "answer_1": first["prediction"], "answer_2": second["prediction"],
                      "rating_1": "", "rating_2": "", "notes": ""})
        key.append({"id": b["id"], "answer_1_is": candidate_name if flip else "baseline"})
    return sheet, key


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize_blind_review(sheet: list[dict], key: list[dict], candidate_name: str = "candidate") -> dict:
    """Unblind completed ratings: mean rating per system, and wins / losses /
    ties for the candidate with an exact sign test (ties excluded)."""
    key_by_id = {k["id"]: k["answer_1_is"] for k in key}
    ratings = {"baseline": [], candidate_name: []}
    wins = losses = ties = 0
    for row in sheet:
        if str(row["rating_1"]).strip() == "" or str(row["rating_2"]).strip() == "":
            continue
        r1, r2 = int(row["rating_1"]), int(row["rating_2"])
        if key_by_id[row["id"]] == candidate_name:
            cand_r, base_r = r1, r2
        else:
            cand_r, base_r = r2, r1
        ratings[candidate_name].append(cand_r)
        ratings["baseline"].append(base_r)
        wins += cand_r > base_r
        losses += cand_r < base_r
        ties += cand_r == base_r
    n = len(ratings["baseline"])
    return {
        "n_rated": n,
        "mean_baseline": sum(ratings["baseline"]) / n if n else float("nan"),
        f"mean_{candidate_name}": sum(ratings[candidate_name]) / n if n else float("nan"),
        "wins": wins, "losses": losses, "ties": ties,
        "sign_test_p": binom_two_sided(wins, wins + losses, 0.5) if wins + losses else float("nan"),
    }


def constant_floor(rows: list[dict]) -> dict[str, float]:
    """Item recall of a system that ignores the image and always writes the most common gold answer for the
    question type (leave-one-out, so a question's own answer is never used). Amendment 2 of the protocol.
    rows need: knowledge_subtype, gold_answer. Returns {'all': ..., subtype: ...}."""
    from collections import Counter
    by_sub: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_sub.setdefault(r["knowledge_subtype"], []).append(i)
    scores: dict[str, list[float]] = {}
    for sub, idx in by_sub.items():
        for i in idx:
            others = Counter(rows[j]["gold_answer"] for j in idx if j != i)
            if not others:
                continue
            best = sorted(others.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            scores.setdefault(sub, []).append(item_scores(best, rows[i]["gold_answer"]).recall)
    out = {sub: sum(v) / len(v) for sub, v in scores.items()}
    flat = [x for v in scores.values() for x in v]
    out["all"] = sum(flat) / len(flat) if flat else float("nan")
    return out


def distinct_answers(rows: list[dict], key: str = "prediction") -> dict[str, int]:
    """Number of different answers (after normalisation) per knowledge subtype and overall."""
    seen: dict[str, set] = {}
    for r in rows:
        seen.setdefault(r["knowledge_subtype"], set()).add(normalize(str(r[key])))
    out = {sub: len(v) for sub, v in seen.items()}
    out["all"] = len({normalize(str(r[key])) for r in rows})
    return out
