"""Analysis of the Phase 2 arms (rag_results.csv) against the pre-registered rules.

Rules: docs/phase2_rag_plan.md and Amendments 1 to 5 in docs/evaluation_protocol.md. This module only
reads result files; nothing here calls the model.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .compare import compare_continuous, compare_binary, pair_by_id
from .evaluate import exact_match, normalize
from .knowledge import constant_floor, distinct_answers, make_blind_review, summarize_blind_review, write_csv
from .stats import cluster_bootstrap_mean_diff

ARMS = ("A1", "A2", "A3", "A4", "A5", "A6")
ARM_NAMES = {
    "A1": "style-matched, no retrieval", "A2": "retrieval from the question alone (oracle set)",
    "A3": "true disease named (oracle)", "A4": "true disease + retrieval (oracle)",
    "A5": "model's disease guess named", "A6": "model's guess + retrieval (the system)",
}

# (label, arm_a, arm_b, selection): all reported, in this order (plan, section 6)
COMPARISONS = [
    ("A1 -> A6  the system", "A1", "A6", "nonfunction"),
    ("A5 -> A6  what retrieval adds to the guess", "A5", "A6", "nonfunction"),
    ("A1 -> A5  what the guess alone adds", "A1", "A5", "nonfunction"),
    ("A1 -> A3  oracle: knowing the disease", "A1", "A3", "oracle"),
    ("A3 -> A4  oracle: what retrieval adds", "A3", "A4", "oracle"),
    ("A1 -> A4  oracle ceiling", "A1", "A4", "oracle"),
    ("A1 -> A2  retrieval from the question alone", "A1", "A2", "oracle"),
]


def load_arm_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("item_recall", "item_precision", "item_f1", "token_f1"):
            r[k] = float(r[k])
        r["is_dev"] = str(r["is_dev"]).strip().lower() in ("true", "1")
        r["image_path"] = r["image_key"]          # cluster key for the image-level bootstrap
        r["group"] = "train" if r["source_split"] == "train" else "validation/test"
        r["exact_match"] = exact_match(r["prediction"], r["gold_answer"])
    return rows


SELECTIONS = {
    "nonfunction": lambda r: r["knowledge_subtype"] != "function",
    "heldout": lambda r: r["knowledge_subtype"] != "function" and not r["is_dev"],
    "oracle": lambda r: r["knowledge_subtype"] != "function" and bool(r["oracle_disease"]),
    "heldout_oracle": lambda r: r["knowledge_subtype"] != "function" and bool(r["oracle_disease"]) and not r["is_dev"],
    "function": lambda r: r["knowledge_subtype"] == "function",
}


def arm(rows: list[dict], name: str, selection: str = "nonfunction") -> list[dict]:
    sel = SELECTIONS[selection]
    return [r for r in rows if r["arm"] == name and sel(r)]


def mean_recall(rows: list[dict]) -> float:
    return sum(r["item_recall"] for r in rows) / len(rows) if rows else float("nan")


def paired(rows: list[dict], arm_a: str, arm_b: str, selection: str):
    return compare_continuous(pair_by_id(arm(rows, arm_a, selection), arm(rows, arm_b, selection)), "item_recall")


def minimum_detectable_difference(rows: list[dict], arm_a: str, arm_b: str, selection: str) -> float:
    """Rough 80%-power, 5%-level minimum detectable mean difference, from the spread of per-image
    differences actually observed (2.8 x standard error). A planning aid, not a test."""
    pairs = pair_by_id(arm(rows, arm_a, selection), arm(rows, arm_b, selection))
    per_image: dict[str, list[float]] = defaultdict(list)
    for a, b in pairs:
        per_image[a["image_path"]].append(b["item_recall"] - a["item_recall"])
    means = [sum(v) / len(v) for v in per_image.values()]
    if len(means) < 2:
        return float("nan")
    m = sum(means) / len(means)
    sd = (sum((x - m) ** 2 for x in means) / (len(means) - 1)) ** 0.5
    return 2.8 * sd / len(means) ** 0.5


def arm_table(rows: list[dict]) -> str:
    nonfn = {a: arm(rows, a, "nonfunction") for a in ARMS}
    floor = constant_floor([r for r in nonfn["A1"]]) if nonfn["A1"] else {}
    L = ["=== Item recall by arm (non-function questions; scorer version 2) ===",
         f"{'arm':>4} {'n':>4} {'recall':>7} {'distinct answers':>17}  {'on oracle subset':>17}   description"]
    for a in ARMS:
        rs = nonfn[a]
        if not rs:
            continue
        orc = arm(rows, a, "oracle")
        L.append(f"{a:>4} {len(rs):>4} {mean_recall(rs):>7.2f} {distinct_answers(rs)['all']:>17} "
                 f"{mean_recall(orc):>11.2f} (n={len(orc)})   {ARM_NAMES[a]}")
    if floor:
        L.append(f"constant-answer floor (most common gold answer per subtype, leave-one-out): {floor['all']:.2f}")
    L.append("\nBy subtype (recall, floor):")
    subtypes = sorted({r["knowledge_subtype"] for r in nonfn["A1"]})
    L.append(f"{'subtype':>12} {'n':>3} " + " ".join(f"{a:>5}" for a in ARMS if nonfn[a]) + f" {'floor':>6}")
    for st in subtypes:
        cells = []
        for a in ARMS:
            if nonfn[a]:
                sub = [r for r in nonfn[a] if r["knowledge_subtype"] == st]
                cells.append(f"{mean_recall(sub):>5.2f}" if sub else f"{'-':>5}")
        n = len([r for r in nonfn["A1"] if r["knowledge_subtype"] == st])
        L.append(f"{st:>12} {n:>3} " + " ".join(cells) + f" {floor.get(st, float('nan')):>6.2f}")
    return "\n".join(L)


def comparison_report(rows: list[dict]) -> str:
    L = ["=== Paired comparisons of item recall (95% CI, resampled by image) ==="]
    for label, a, b, sel in COMPARISONS:
        if not arm(rows, a, sel) or not arm(rows, b, sel):
            continue
        L.append(f"\n{label}   [{sel}]")
        for name, selection in ((f"{sel} (all rows)", sel),) + (
                (("held-out only", "heldout" if sel == "nonfunction" else "heldout_oracle"),)):
            c = paired(rows, a, b, selection)
            L.append(f"  {name:>24}: n={c.n:<3} {a}={c.mean_a:.3f} {b}={c.mean_b:.3f} diff={c.diff:+.3f} "
                     f"(CI {c.ci_low:+.3f} to {c.ci_high:+.3f})")
        for group in ("train", "validation/test"):
            pairs = pair_by_id([r for r in arm(rows, a, sel) if r["group"] == group],
                               [r for r in arm(rows, b, sel) if r["group"] == group])
            if pairs:
                c = compare_continuous(pairs, "item_recall")
                L.append(f"  {group:>24}: n={c.n:<3} diff={c.diff:+.3f} (CI {c.ci_low:+.3f} to {c.ci_high:+.3f})")
    L.append(f"\nSubtype split for A1 -> A6 (all non-function rows):")
    for st in sorted({r["knowledge_subtype"] for r in arm(rows, "A1", "nonfunction")}):
        pairs = pair_by_id([r for r in arm(rows, "A1") if r["knowledge_subtype"] == st],
                           [r for r in arm(rows, "A6") if r["knowledge_subtype"] == st])
        if pairs:
            cc = compare_continuous(pairs, "item_recall")
            L.append(f"  {st:>12}: n={cc.n:<3} diff={cc.diff:+.3f} (CI {cc.ci_low:+.3f} to {cc.ci_high:+.3f})")
    mdd = minimum_detectable_difference(rows, "A1", "A6", "heldout")
    L.append(f"\nPower note: with the spread observed in the held-out A1 -> A6 differences, the smallest mean "
             f"item-recall difference this set could reliably detect is about {mdd:.2f}.")
    return "\n".join(L)


def guess_accuracy(rows: list[dict]) -> dict:
    """How often the model's disease guess equals the true disease, on the oracle subset (A6 rows)."""
    rs = arm(rows, "A6", "oracle")
    hit = sum(r["guessed_disease"] == r["oracle_disease"] for r in rs)
    return {"n": len(rs), "correct": hit}


# --------------------------------------------------------------------------- blind rating sheets

RATING_PAIRS = [("A1", "A6", "heldout"), ("A5", "A6", "heldout"), ("A3", "A4", "heldout_oracle")]


def make_rating_sheets(rows: list[dict], out_dir: str, seed: int = 0) -> dict[str, dict]:
    """Write a blind sheet and key per pair. Pairs whose two answers are identical are tied by definition and left
    off the sheet (their number is stored in the meta file), so the reviewer rates only differing pairs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for a, b, sel in RATING_PAIRS:
        ra, rb = arm(rows, a, sel), arm(rows, b, sel)
        by_b = {r["id"]: r for r in rb}
        diff_a = [r for r in ra if r["id"] in by_b and normalize(r["prediction"]) != normalize(by_b[r["id"]]["prediction"])]
        n_pairs = sum(r["id"] in by_b for r in ra)
        sheet, key = make_blind_review(diff_a, [by_b[r["id"]] for r in diff_a], seed=seed, candidate_name=b)
        stem = f"ratings_{a}_vs_{b}"
        write_csv(str(out / f"{stem}_sheet.csv"), sheet)
        write_csv(str(out / f"{stem}_key.csv"), key)
        meta = {"pair": f"{a} vs {b}", "selection": sel, "pairs": n_pairs, "identical": n_pairs - len(diff_a),
                "to_rate": len(diff_a)}
        (out / f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))
        info[f"{a}_vs_{b}"] = meta
    return info


def summarize_rating_files(directory: str, a: str, b: str) -> dict | None:
    """Unblind a completed sheet. None if the sheet is missing or has no ratings yet."""
    d = Path(directory)
    meta_p = d / f"ratings_{a}_vs_{b}_meta.json"
    if not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text())
    if meta["to_rate"] == 0:      # every pair identical: all ties, nothing to rate
        return {"n_rated": 0, "mean_baseline": float("nan"), f"mean_{b}": float("nan"), "wins": 0, "losses": 0,
                "ties": 0, "sign_test_p": 1.0, "identical_pairs_tied": meta["identical"], "to_rate": 0}
    with open(d / f"ratings_{a}_vs_{b}_sheet.csv", newline="") as f:
        sheet = list(csv.DictReader(f))
    with open(d / f"ratings_{a}_vs_{b}_key.csv", newline="") as f:
        key = list(csv.DictReader(f))
    s = summarize_blind_review(sheet, key, candidate_name=b)
    if s["n_rated"] == 0:
        return None
    s["identical_pairs_tied"] = meta["identical"]
    s["to_rate"] = meta["to_rate"]
    return s


# ------------------------------------------------------------------------------ decision rule

def decision_rule(rows: list[dict], ratings: dict[str, dict | None], run2_rows: list[dict] | None = None,
                  total_baseline_questions: int = 219) -> list[dict]:
    """The conditions of Amendment 5 (the protocol's four, plus A6 beating A5), each PASS / FAIL / PENDING."""
    checks = []

    def add(name, status, detail):
        checks.append({"condition": name, "status": status, "detail": detail})

    c = paired(rows, "A1", "A6", "heldout")
    add("1. item recall A6 > A1 on held-out non-function, 95% CI excludes 0",
        "PASS" if c.ci_low > 0 else "FAIL", f"diff {c.diff:+.3f} (CI {c.ci_low:+.3f} to {c.ci_high:+.3f}, n={c.n})")

    r = ratings.get("A1_vs_A6")
    if r is None:
        add("2. blind rating favours A6 over A1 (sign test p < 0.05)", "PENDING", "no completed rating sheet")
    else:
        ok = r["sign_test_p"] < 0.05 and r["wins"] > r["losses"]
        add("2. blind rating favours A6 over A1 (sign test p < 0.05)", "PASS" if ok else "FAIL",
            f"wins {r['wins']}, losses {r['losses']}, ties {r['ties']} (+{r['identical_pairs_tied']} identical), p={r['sign_test_p']:.3f}")

    diffs = {}
    for g in ("train", "validation/test"):
        pairs = pair_by_id([x for x in arm(rows, "A1", "heldout") if x["group"] == g],
                           [x for x in arm(rows, "A6", "heldout") if x["group"] == g])
        diffs[g] = compare_continuous(pairs, "item_recall").diff if pairs else None
    same = all(v is not None and v > 0 for v in diffs.values())
    add("3. same direction (A6 above A1) on train rows and on validation/test rows",
        "PASS" if same else "FAIL", ", ".join(f"{g}: {('n/a' if v is None else f'{v:+.3f}')}" for g, v in diffs.items()))

    fn = arm(rows, "A6", "function")
    breathe = sum("breathe" in x["prediction"].lower() for x in fn)
    harm_ok = bool(fn) and breathe == len(fn)
    detail = f"function questions still answered 'Breathe': {breathe}/{len(fn)}"
    if run2_rows is not None:
        ids = {x["id"] for x in arm(rows, "A6", "nonfunction") + fn}
        base = {x["id"]: x for x in run2_rows if x["id"] in ids}
        sys_rows = {x["id"]: x for x in arm(rows, "A6", "nonfunction") + fn if x["id"] in base}
        pairs = [({"exact_match": str(base[i]["exact_match"]).lower() in ("true", "1", "1.0")},
                  {"exact_match": sys_rows[i]["exact_match"]}) for i in sys_rows]
        bc = compare_binary(pairs, "exact_match")
        points = (bc.b_correct - bc.a_correct) / total_baseline_questions * 100
        worse = bc.only_a > bc.only_b and bc.p_value < 0.05
        detail += (f"; routed baseline questions: run 2 right {bc.a_correct}, A6 right {bc.b_correct} "
                   f"(McNemar p={bc.p_value:.3f}); exact-match change over all {total_baseline_questions}: {points:+.1f} points")
        harm_ok = harm_ok and not worse and points >= -3
    add("4. no harm (function answers unchanged; routed baseline questions not significantly worse, < 3 points lost)",
        "PASS" if harm_ok else "FAIL", detail)

    c5 = paired(rows, "A5", "A6", "heldout")
    r5 = ratings.get("A5_vs_A6")
    rec_ok = c5.ci_low > 0
    if r5 is None:
        add("5. retrieval beats A5 (recall CI excludes 0, and blind rating p < 0.05)", "PENDING" if rec_ok else "FAIL",
            f"recall diff {c5.diff:+.3f} (CI {c5.ci_low:+.3f} to {c5.ci_high:+.3f}); no completed rating sheet")
    else:
        ok = rec_ok and r5["sign_test_p"] < 0.05 and r5["wins"] > r5["losses"]
        add("5. retrieval beats A5 (recall CI excludes 0, and blind rating p < 0.05)", "PASS" if ok else "FAIL",
            f"recall diff {c5.diff:+.3f} (CI {c5.ci_low:+.3f} to {c5.ci_high:+.3f}); rating wins {r5['wins']}, "
            f"losses {r5['losses']}, p={r5['sign_test_p']:.3f}")
    return checks


def format_decision(checks: list[dict]) -> str:
    L = ["=== Decision rule (Amendment 5) ==="]
    for c in checks:
        L.append(f"[{c['status']:>7}] {c['condition']}\n          {c['detail']}")
    statuses = {c["status"] for c in checks}
    if statuses == {"PASS"}:
        verdict = "All conditions met: retrieval is reported as helping on knowledge questions (bounded, SLAKE template questions)."
    elif "FAIL" in statuses:
        c1 = checks[0]["status"] == "PASS"
        c5 = checks[4]["status"] == "PASS"
        if c1 and not c5:
            verdict = "A6 beats A1 but not A5: the gain comes from naming the disease, not from retrieval."
        elif checks[0]["status"] == "PASS" and checks[2]["status"] == "FAIL":
            verdict = "Improvement on rows the model probably saw, not shown elsewhere."
        else:
            verdict = "No demonstrated improvement from retrieval under the pre-registered rule."
    else:
        verdict = "Not decided yet: blind rating sheets still to be completed (PENDING above)."
    L.append(f"\nVerdict: {verdict}")
    return "\n".join(L)
