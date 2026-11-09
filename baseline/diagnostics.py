"""Cheap perception diagnostics for CheXagent-2-3b (no training, no retrieval).

Run 3 found two perception faults: a constant "Right" on explicit "which side"
questions, and "no" to every heart-enlargement question whose answer was "yes".
These tests ask what could explain them, using the same model:

A. Flip test. Mirror each image left-right. A model that reads laterality from
   the image should swap its left/right answers; a model giving a constant answer
   will not. (Mirroring also moves the heart to the right side of the picture, so
   anatomy-based cues flip as well; that is consistent with a swap, not against it.)
B. Prompt variants on side questions. Does wording, a forced choice, describing
   first, or stating the left/right convention change the answers, and do the
   answers then depend on the true side?
C. Heart-size variants. Does describing the heart first, or asking "normal or
   enlarged?" instead of a yes/no question, get the model to ever say "enlarged"?

`docs/phase1c_diagnostics.md` states, before any result exists, what each outcome
would mean.
"""

from __future__ import annotations

import random
import re

from .sides import side_of
from .stats import fisher_exact_2x2
from .evaluate import wilson_interval

OPPOSITE = {"left": "right", "right": "left"}

# ---------------------------------------------------------------- side prompts

SIDE_PROMPTS = {
    "suffix": "{question} Answer concisely based only on what is visible in the image.",
    "plain": "{question}",
    "one_word": "{question} Answer with exactly one word: Left or Right.",
    "describe_first": ("Describe in one sentence where the abnormality or finding is in this chest X-ray, "
                       "then answer the question. {question}"),
    "patient_side": ("{question} Left and right mean the patient's own left and right "
                     "(the patient's right side appears on the left of the picture)."),
    "image_side": "{question} Left and right mean the left and right of the picture as you look at it.",
}
SIDE_MAX_TOKENS = {"describe_first": 160}

_EXPLICIT_SIDE_RE = re.compile(r"left or right|which side")


def is_explicit_side_question(question: str) -> bool:
    """'Which side ...' or '... left or right?' questions, where the answer is
    a choice of side (as opposed to 'where is the abnormality?')."""
    return bool(_EXPLICIT_SIDE_RE.search(question.lower()))


def select_side_diagnostic_records(side_records: list[dict], n_other: int = 31, seed: int = 0) -> list[dict]:
    """Left/right-gold side questions: every explicit one plus `n_other` random
    others. Bilateral gold answers are excluded (they have no mirror image)."""
    lr = [r for r in side_records if r.get("gold_side") in ("left", "right")]
    explicit = [r for r in lr if is_explicit_side_question(r["question"])]
    others = [r for r in lr if not is_explicit_side_question(r["question"])]
    rng = random.Random(seed)
    rng.shuffle(others)
    return explicit + others[:n_other]


# --------------------------------------------------------------------- flip test

def flip_image(src_path: str, dest_path: str) -> str:
    """Write a left-right mirrored copy of an image."""
    from PIL import Image, ImageOps
    with Image.open(src_path) as im:
        ImageOps.mirror(im.convert("RGB")).save(dest_path)
    return dest_path


def flip_report(rows: list[dict]) -> dict:
    """Rows need: gold_side (left/right), pred_original, pred_flipped (model texts)
    and optionally `explicit` (bool). Reports how answers move when the image is mirrored."""
    def summarize(rs: list[dict]) -> dict:
        po = [side_of(r["pred_original"]) for r in rs]
        pf = [side_of(r["pred_flipped"]) for r in rs]
        both = [(a, b) for a, b in zip(po, pf) if a in OPPOSITE and b in OPPOSITE]
        swapped = sum(b == OPPOSITE[a] for a, b in both)
        same = sum(b == a for a, b in both)
        identical = sum(r["pred_original"].strip().lower() == r["pred_flipped"].strip().lower() for r in rs)
        acc_o = sum(a == r["gold_side"] for a, r in zip(po, rs))
        acc_f = sum(b == OPPOSITE[r["gold_side"]] for b, r in zip(pf, rs))
        return {
            "n": len(rs), "named_both": len(both), "swapped": swapped, "same": same,
            "swap_rate": swapped / len(both) if both else float("nan"),
            "swap_ci": wilson_interval(swapped, len(both)) if both else (float("nan"), float("nan")),
            "identical_text": identical,
            "acc_original": acc_o / len(rs) if rs else float("nan"),
            "acc_flipped_vs_flipped_gold": acc_f / len(rs) if rs else float("nan"),
            "right_rate_original": sum(a == "right" for a in po) / len(rs) if rs else float("nan"),
            "right_rate_flipped": sum(b == "right" for b in pf) / len(rs) if rs else float("nan"),
        }
    out = {"all": summarize(rows)}
    explicit = [r for r in rows if r.get("explicit")]
    if explicit:
        out["explicit"] = summarize(explicit)
    return out


def format_flip_report(rep: dict) -> str:
    L = ["=== Flip test: do left/right answers swap when the image is mirrored? ==="]
    for name, s in rep.items():
        lo, hi = s["swap_ci"]
        L.append(f"[{name}] n={s['n']}  answered left/right on both images: {s['named_both']}")
        L.append(f"   swapped: {s['swapped']}  stayed the same: {s['same']}   swap rate = {s['swap_rate']:.0%} "
                 f"(95% CI {lo:.0%}-{hi:.0%})")
        L.append(f"   byte-identical answers on both images: {s['identical_text']} / {s['n']}")
        L.append(f"   share answering 'right': original {s['right_rate_original']:.0%}, flipped {s['right_rate_flipped']:.0%}")
        L.append(f"   correct on original: {s['acc_original']:.0%}   correct on flipped (vs mirrored gold): "
                 f"{s['acc_flipped_vs_flipped_gold']:.0%}")
    return "\n".join(L)


# ------------------------------------------------------------ prompt variants (B)

def variant_report(rows: list[dict]) -> dict:
    """Rows need: gold_side (left/right) and prediction. `depends_on_truth_p` is the
    Fisher p for 'is the model's left/right answer related to the true side?'"""
    preds = [side_of(r["prediction"]) for r in rows]
    gl = [(p, r) for p, r in zip(preds, rows) if r["gold_side"] == "left"]
    gr = [(p, r) for p, r in zip(preds, rows) if r["gold_side"] == "right"]
    l_right = sum(p == "right" for p, _ in gl)
    l_left = sum(p == "left" for p, _ in gl)
    r_right = sum(p == "right" for p, _ in gr)
    r_left = sum(p == "left" for p, _ in gr)
    return {
        "n": len(rows),
        "named_side": sum(p in OPPOSITE for p in preds) / len(rows) if rows else float("nan"),
        "answered_right": sum(p == "right" for p in preds) / len(rows) if rows else float("nan"),
        "acc_gold_left": l_left / len(gl) if gl else float("nan"),
        "acc_gold_right": r_right / len(gr) if gr else float("nan"),
        "acc_overall": (l_left + r_right) / len(rows) if rows else float("nan"),
        "depends_on_truth_p": (fisher_exact_2x2(l_right, l_left, r_right, r_left)
                               if (l_right + l_left) and (r_right + r_left) else float("nan")),
    }


def format_variant_table(by_variant: dict[str, dict]) -> str:
    L = ["=== Side prompt variants ===",
         f"{'variant':>15} {'n':>4} {'named side':>10} {'said right':>10} {'acc gold-L':>10} {'acc gold-R':>10} "
         f"{'acc all':>8} {'depends on truth p':>19}"]
    for name, v in by_variant.items():
        L.append(f"{name:>15} {v['n']:>4} {v['named_side']:>10.0%} {v['answered_right']:>10.0%} "
                 f"{v['acc_gold_left']:>10.0%} {v['acc_gold_right']:>10.0%} {v['acc_overall']:>8.0%} "
                 f"{v['depends_on_truth_p']:>19.3f}")
    L.append("'depends on truth p' small (< 0.05) means the answer follows the true side; "
             "near 1 means it does not.")
    return "\n".join(L)


# ------------------------------------------------------------------ heart size (C)

_ENLARGE_POS = re.compile(r"enlarg|cardiomegaly|abnormally large|too big|too large|larger than normal|heart.*\blarge\b")
_HEART = re.compile(r"heart|cardiac|cardiomegaly")
_REVERSED = re.compile(r"normal|within|small|stretched|decreas|reduc")
_NOT_HEART = re.compile(r"pulmonary|hilar|bronch|vascul|aort|atri|ventric|pericard|mediastin")
_ON_LUNG = re.compile(r"\bon the (upper|lower|left|right|middle)[^?]*\blung")


def select_heart_size_records(pool: list[dict]) -> list[dict]:
    """Yes/no questions where 'yes' means the heart is enlarged: 'Is the heart
    enlarged?', 'Is there cardiomegaly?', 'Can cardiomegaly be observed in this
    picture?'. Excludes reversed wording ('Is the size normal?'), other structures
    (hilum, vessels, atria), and location-specific cardiomegaly questions, whose
    'no' answers are about the location."""
    out, seen = [], set()
    for r in pool:
        q = r["question"].lower()
        if r["answer_format"] != "yes/no" or not _HEART.search(q) or not _ENLARGE_POS.search(q):
            continue
        if _REVERSED.search(q) or _NOT_HEART.search(q) or _ON_LUNG.search(q):
            continue
        key = (r["source_dataset"], r["uid"])
        if key in seen:
            continue
        seen.add(key)
        out.append({**r, "gold_yes": r["gold_answer"].strip().lower().rstrip(".") == "yes"})
    return out


HEART_PROMPTS = {
    "run2_suffix": "{question} Answer concisely based only on what is visible in the image.",
    "plain": "{question}",
    "describe_first": ("{question} First describe the size of the heart compared with the width of the chest, "
                       "then finish with 'Answer: yes' or 'Answer: no'."),
    "normal_or_enlarged": "Is the heart size normal or enlarged? Answer with one word: normal or enlarged.",
}
HEART_MAX_TOKENS = {"describe_first": 160}

_ANSWER_RE = re.compile(r"answer\s*:\s*(yes|no)\b")
_YESNO_RE = re.compile(r"\b(yes|no)\b")
_NEGATED_ENLARGED = re.compile(r"\b(not|no|without|nor)\b[^.;]{0,25}\b(enlarged|enlargement|cardiomegaly|large)\b")


def parse_yes_no(text: str) -> str | None:
    """Final yes/no in a free-text answer: an 'Answer: yes/no' line if present,
    otherwise the last standalone yes or no."""
    t = text.lower()
    m = _ANSWER_RE.findall(t)
    if m:
        return m[-1]
    m = _YESNO_RE.findall(t)
    return m[-1] if m else None


def parse_normal_or_enlarged(text: str) -> str | None:
    """Map a 'normal / enlarged' answer to 'yes' (enlarged) or 'no' (not enlarged)."""
    t = text.lower()
    if _NEGATED_ENLARGED.search(t):
        return "no"
    if re.search(r"enlarg|cardiomegaly|\blarge\b|\bbig\b|increased", t):
        return "yes"
    if re.search(r"\bnormal\b|\bunremarkable\b", t):
        return "no"
    return None


def parse_heart_answer(variant: str, text: str) -> str | None:
    return parse_normal_or_enlarged(text) if variant == "normal_or_enlarged" else parse_yes_no(text)


def heart_report(rows: list[dict], variant: str) -> dict:
    """Rows need: gold_yes (bool) and prediction (model text)."""
    tp = fn = tn = fp = unparsed = 0
    for r in rows:
        p = parse_heart_answer(variant, r["prediction"])
        if p is None:
            unparsed += 1
        if r["gold_yes"]:
            tp += p == "yes"
            fn += p != "yes"
        else:
            tn += p == "no"
            fp += p != "no"
    pos, neg, n = tp + fn, tn + fp, len(rows)
    return {
        "n": n, "gold_yes": pos, "gold_no": neg, "tp": tp, "fn": fn, "tn": tn, "fp": fp, "unparsed": unparsed,
        "sensitivity": tp / pos if pos else float("nan"),
        "specificity": tn / neg if neg else float("nan"),
        "accuracy": (tp + tn) / n if n else float("nan"),
        "always_no_accuracy": neg / n if n else float("nan"),
        "predicted_yes_rate": (tp + fp) / n if n else float("nan"),
        # Youden's J = sensitivity + specificity - 1: 0 for a system that ignores the image (including one
        # that answers "yes" to everything), 1 for a perfect one.
        "youden_j": (tp / pos + tn / neg - 1) if pos and neg else float("nan"),
    }


def format_heart_table(by_variant: dict[str, dict]) -> str:
    L = ["=== Heart-size variants (positive = heart enlarged) ===",
         f"{'variant':>20} {'n':>3} {'yes in gold':>11} {'sensitivity':>12} {'specificity':>12} {'accuracy':>9} "
         f"{'always-no':>9} {'said yes':>9} {'Youden J':>9} {'unparsed':>8}"]
    for name, h in by_variant.items():
        sens = f"{h['tp']}/{h['gold_yes']} ({h['sensitivity']:.0%})"
        spec = f"{h['tn']}/{h['gold_no']} ({h['specificity']:.0%})"
        L.append(f"{name:>20} {h['n']:>3} {h['gold_yes']:>11} {sens:>12} {spec:>12} {h['accuracy']:>9.0%} "
                 f"{h['always_no_accuracy']:>9.0%} {h['predicted_yes_rate']:>9.0%} {h['youden_j']:>9.2f} {h['unparsed']:>8}")
    L.append("Youden J = sensitivity + specificity - 1. About 0 means no skill, even if the model says 'yes' more often.")
    return "\n".join(L)
