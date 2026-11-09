"""Stage 4 of docs/phase2_rag_plan.md: which baseline-set questions would the router send to retrieval?

Only questions matched by `is_knowledge_question` go to retrieval; everything else keeps the baseline path unchanged.
This lists the routed ones among the 219 questions of run 2 (minus the 5 excluded ids) and checks that none of the
questions in the knowledge set escapes the router.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.categorize import is_knowledge_question, knowledge_subtype  # noqa: E402

base = pd.read_csv("results/run_2/baseline_results.csv")
excluded = set(Path("results/run_2/excluded_ids.txt").read_text().split())
base = base[~base["id"].isin(excluded)]
routed = base[base["question"].map(is_knowledge_question)]
print(f"baseline set: {len(base)} questions; routed to retrieval: {len(routed)}")
print(routed.assign(subtype=routed["question"].map(knowledge_subtype))
      .groupby("subtype").size().to_string())
print("\nrouted questions that are in the knowledge test set (ids shared with run 3):")
know = pd.read_csv("results/run_3/knowledge_results.csv")
shared = routed[routed["id"].isin(set(know["id"]))]
print(f"  {len(shared)} of {len(routed)} routed baseline questions are also knowledge-set questions")
print("  routed but NOT in the knowledge set:", sorted(set(routed["id"]) - set(know["id"])))
print("\ncheck: every knowledge-set question passes the router:",
      bool(know["question"].map(is_knowledge_question).all()))
print("check: function questions routed:", int((know["question"].map(knowledge_subtype) == "function").sum()))
summary = {"baseline_n": int(len(base)), "routed": int(len(routed)), "routed_in_knowledge_set": int(len(shared)),
           "routed_ids_not_in_knowledge_set": sorted(set(routed["id"]) - set(know["id"]))}
Path("results/phase2").mkdir(parents=True, exist_ok=True)
json.dump(summary, open("results/phase2/router_check.json", "w"), indent=1)
