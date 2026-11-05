"""Scoring for 'which side' questions.

Gold answers to side questions range from "Right" to "Lower Left Lung", and a
model that answers "left lung" to "Lower Left Lung" has the side right even
though exact match scores it 0. So side questions are scored on the extracted
side (left / right / both), in addition to the usual metrics.

The same extraction supports the specific follow-up question from run 2: when
the correct side is left or both, does the model still answer "right"?
"""

from __future__ import annotations

import re

from .stats import binom_two_sided, fisher_exact_2x2

_BOTH_RE = re.compile(r"\b(bilateral|bilaterally|both)\b")
_LEFT_RE = re.compile(r"\bleft\b")
_RIGHT_RE = re.compile(r"\bright\b")


def side_of(text: str) -> str | None:
    """'left', 'right' or 'both' if the text names a side, else None.

    Text naming both words ("left and right lungs") counts as 'both'. Answers
    such as "No" or "Lung" name no side and return None."""
    t = str(text).lower()
    if _BOTH_RE.search(t):
        return "both"
    has_l, has_r = bool(_LEFT_RE.search(t)), bool(_RIGHT_RE.search(t))
    if has_l and has_r:
        return "both"
    if has_l:
        return "left"
    if has_r:
        return "right"
    return None


def score_side_rows(rows: list[dict]) -> list[dict]:
    """Add gold_side / pred_side / side_correct / side_given to each row."""
    out = []
    for r in rows:
        gold, pred = side_of(r["gold_answer"]), side_of(r["prediction"])
        out.append({**r, "gold_side": gold, "pred_side": pred,
                    "side_given": pred is not None, "side_correct": pred == gold})
    return out


def side_report(rows: list[dict]) -> dict:
    """Confusion table and bias statistics for rows scored by `score_side_rows`
    (rows whose gold names no side are ignored)."""
    rows = [r for r in rows if r.get("gold_side")]
    sides = ("left", "right", "both")
    confusion = {g: {p: 0 for p in (*sides, "none")} for g in sides}
    for r in rows:
        confusion[r["gold_side"]][r["pred_side"] or "none"] += 1

    per_gold = {}
    for g in sides:
        n = sum(confusion[g].values())
        per_gold[g] = {"n": n, "correct": confusion[g][g],
                       "accuracy": confusion[g][g] / n if n else float("nan")}

    # Asymmetry: among gold-left items that got a left/right answer, how often
    # was it "right"; among gold-right items, how often "left"? Equal rates
    # mean errors are symmetric; a higher left->right rate is a right default.
    l_to_r, l_to_l = confusion["left"]["right"], confusion["left"]["left"]
    r_to_l, r_to_r = confusion["right"]["left"], confusion["right"]["right"]
    asym_p = (fisher_exact_2x2(l_to_r, l_to_l, r_to_l, r_to_r)
              if (l_to_r + l_to_l) and (r_to_l + r_to_r) else float("nan"))

    # Among left/right-gold items answered left or right, is accuracy above 50%?
    lr = [r for r in rows if r["gold_side"] in ("left", "right") and r["pred_side"] in ("left", "right")]
    lr_correct = sum(r["side_correct"] for r in lr)

    pred_right = sum(1 for r in rows if r["pred_side"] == "right")
    gold_right = sum(1 for r in rows if r["gold_side"] == "right")
    return {
        "n": len(rows),
        "confusion": confusion,
        "per_gold": per_gold,
        "left_to_right_rate": l_to_r / (l_to_r + l_to_l) if (l_to_r + l_to_l) else float("nan"),
        "right_to_left_rate": r_to_l / (r_to_l + r_to_r) if (r_to_l + r_to_r) else float("nan"),
        "asymmetry_p": asym_p,
        "lr_answered": len(lr),
        "lr_correct": lr_correct,
        "lr_vs_chance_p": binom_two_sided(lr_correct, len(lr), 0.5) if lr else float("nan"),
        "predicted_right_rate": pred_right / len(rows) if rows else float("nan"),
        "gold_right_rate": gold_right / len(rows) if rows else float("nan"),
        "no_side_answer": sum(1 for r in rows if r["pred_side"] is None),
    }


def format_side_report(rep: dict) -> str:
    L = [f"=== Side questions (n={rep['n']}) ===", "gold \\ predicted      left  right  both  none  accuracy"]
    for g in ("left", "right", "both"):
        c, pg = rep["confusion"][g], rep["per_gold"][g]
        L.append(f"{g:>20}  {c['left']:5d}  {c['right']:5d}  {c['both']:4d}  {c['none']:4d}"
                 f"  {pg['correct']}/{pg['n']} ({pg['accuracy']:.0%})" if pg["n"] else f"{g:>20}  (none)")
    L.append("")
    L.append(f"model answered 'right' on {rep['predicted_right_rate']:.0%} of questions; "
             f"'right' was correct on {rep['gold_right_rate']:.0%}")
    L.append(f"gold-left answered 'right': {rep['left_to_right_rate']:.0%}   "
             f"gold-right answered 'left': {rep['right_to_left_rate']:.0%}   "
             f"(Fisher p for asymmetry = {rep['asymmetry_p']:.3f})")
    L.append(f"left-vs-right accuracy among items answered left/right: "
             f"{rep['lr_correct']}/{rep['lr_answered']}   (exact binomial p vs 50% chance = {rep['lr_vs_chance_p']:.3f})")
    L.append(f"questions where the model named no side: {rep['no_side_answer']}")
    return "\n".join(L)
