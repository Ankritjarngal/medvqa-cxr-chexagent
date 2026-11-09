"""Stage 2 of docs/phase2_rag_plan.md: does the corpus hold the gold answer's items? (no model)

  python3 scripts/retrieval_check.py tune      # grid on DEV oracle questions only; prints the setting to freeze
  python3 scripts/retrieval_check.py report K SOURCE MAXCHARS WEIGHT   # frozen setting, dev + held-out + controls

Run `tune` first, record the chosen setting in docs/experiment_log.md, then run `report`.
"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.disease_labels import attach_oracle_disease  # noqa: E402
from baseline.extra_sets import select_knowledge_records, split_dev_heldout  # noqa: E402
from baseline.rag import (  # noqa: E402
    MAX_CONTEXT_CHARS, RETRIEVAL_SOURCES, build_retrievers, choose_setting, coverage_for_setting, mean_coverage,
)

POOL = "data/cache/pools_full.json"
SOURCES = RETRIEVAL_SOURCES
KS = (1, 2, 3)
CHARS = (400, 700)
WEIGHTS = (1, 2)


def load_records():
    pool = json.load(open(POOL))
    recs = attach_oracle_disease(select_knowledge_records(pool), pool)
    nonfn = [r for r in recs if r["knowledge_subtype"] != "function"]
    dev, held = split_dev_heldout(nonfn)
    return [r for r in dev if r["oracle_disease"]], [r for r in held if r["oracle_disease"]]


def grid():
    for s, k, c, w in itertools.product(SOURCES, KS, CHARS, WEIGHTS):
        if k * c <= MAX_CONTEXT_CHARS:
            yield s, k, c, w


def tune():
    dev, _ = load_records()
    print(f"dev oracle questions: {len(dev)} (held-out questions are not touched)")
    retrievers = build_retrievers()
    results = {}
    for setting in grid():
        s, k, c, w = setting
        results[setting] = mean_coverage(coverage_for_setting(dev, retrievers, s, k, c, w))
    print(f"{'source':>12} {'k':>2} {'chars':>5} {'wt':>2} {'strict':>7} {'loose':>7}")
    for setting, m in sorted(results.items(), key=lambda kv: (-kv[1]["strict"], -kv[1]["loose"])):
        print(f"{setting[0]:>12} {setting[1]:>2} {setting[2]:>5} {setting[3]:>2} {m['strict']:>7.2f} {m['loose']:>7.2f}")
    best = choose_setting(results)
    print("\nchosen by the pre-stated rule:", best, results[best])
    if results[best]["strict"] < 0.30:
        print("GATE: strict coverage under 0.30 even for the best setting. Fix the corpus or the query "
              "(and log it) before any GPU run.")


def report(k, source, max_chars, weight):
    dev, held = load_records()
    retriever = build_retrievers()
    diseases = sorted({r["oracle_disease"] for r in dev + held})
    rng = random.Random(0)

    def wrong(r):
        return rng.choice([d for d in diseases if d != r["oracle_disease"]])

    out = {}
    for name, recs in (("dev", dev), ("held-out", held), ("all", dev + held)):
        rows = coverage_for_setting(recs, retriever, source, k, max_chars, weight)
        wrong_rows = coverage_for_setting(recs, retriever, source, k, max_chars, weight, disease_of=wrong)
        qonly = coverage_for_setting(recs, retriever, source, k, max_chars, weight, disease_of=lambda r: None)
        out[name] = {"true disease": mean_coverage(rows), "wrong disease": mean_coverage(wrong_rows),
                     "question only": mean_coverage(qonly)}
        subtypes = sorted({r["subtype"] for r in rows})
        out[name]["by_subtype_true_disease"] = {st: mean_coverage([r for r in rows if r["subtype"] == st]) for st in subtypes}
    print(f"setting: source={source} k={k} max_chars={max_chars} disease_weight={weight}")
    for name, d in out.items():
        print(f"\n[{name}]")
        for arm in ("true disease", "wrong disease", "question only"):
            m = d[arm]
            print(f"  {arm:>14}: n={m['n']} strict coverage {m['strict']:.2f}  loose {m['loose']:.2f}")
        for st, m in d["by_subtype_true_disease"].items():
            print(f"     {st:>12} (true disease): n={m['n']} strict {m['strict']:.2f} loose {m['loose']:.2f}")
    Path("results/phase2").mkdir(parents=True, exist_ok=True)
    json.dump({"setting": [source, k, max_chars, weight], **out}, open("results/phase2/retrieval_check.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune()
    else:
        report(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
