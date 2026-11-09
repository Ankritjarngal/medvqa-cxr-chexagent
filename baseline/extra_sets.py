"""Two supplementary test sets built from the same public sources as the 224-question baseline.

1. Knowledge set: every symptoms / treatment / prevention / cause / function /
   'why' question available in SLAKE (all splits) and VQA-RAD. Run 2 had only 24,
   too few to detect a retrieval improvement. The pool is small (about 130) and
   made of a few fixed templates, so analysis must treat questions about the
   same image or template as correlated (see compare.py).

2. Side set: open questions whose gold answer names a side ('Right', 'Lower Left
   Lung', 'Both Lung'), sampled with equal numbers of left and right answers so
   a model that favours one side cannot score well by habit. Run 2 hinted at a
   right-side default from only 13 such questions.

Both sets include SLAKE train rows. CheXagent was very likely trained on those
(CheXinstruct includes SLAKE), so every record carries `source_split` and
results should be reported for train and test/validation rows separately.
Train rows make the set larger, not cleaner.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .categorize import knowledge_subtype
from .data import download_slake_images, download_vqa_rad_images, fetch_slake_chest_records, fetch_vqa_rad_chest_records
from .sides import side_of

KNOWLEDGE_PROMPT = "{question}"
SIDE_PROMPT = "{question} Answer concisely based only on what is visible in the image."


def prompt_for_set(set_name: str, question: str, category: str) -> str:
    """Prompt used when running a supplementary set.

    knowledge: the plain question for every row. Run 2 appended "based only on
    what is visible in the image" to laterality/severity-tagged rows, which
    included 20 of its 24 knowledge questions, a confound for a question that
    is answered from knowledge.
    side: the visible-in-image suffix for every row (what run 2 used for
    laterality questions).
    """
    if set_name == "knowledge":
        return KNOWLEDGE_PROMPT.format(question=question)
    if set_name == "side":
        return SIDE_PROMPT.format(question=question)
    raise ValueError(f"unknown set: {set_name}")


def fetch_full_pool(include_train: bool = True, cache_path: str | Path | None = None) -> list[dict]:
    """All chest candidates from SLAKE (test, validation, train) and VQA-RAD.
    Cached as JSON because the full fetch takes about a minute."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())
    splits = ["test", "validation"] + (["train"] if include_train else [])
    pool: list[dict] = []
    for split in splits:
        pool += fetch_slake_chest_records(split=split)
    pool += fetch_vqa_rad_chest_records(splits=("test", "train"))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(pool))
    return pool


def _image_identity(r: dict) -> str:
    """Stable identity of the image behind a question. VQA-RAD rows carry a
    content hash, so questions about one image are recognised as sharing it."""
    return f"{r['source_dataset']}:{r.get('image_id') or r['image_key'].split('?')[0]}"


def select_knowledge_records(pool: list[dict]) -> list[dict]:
    """Knowledge questions, deduplicated on (source, image, question text)."""
    seen: set[tuple[str, str, str]] = set()
    out = []
    for r in pool:
        if not r["is_knowledge"]:
            continue
        key = (r["source_dataset"], _image_identity(r), r["question"].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append({**r, "knowledge_subtype": knowledge_subtype(r["question"])})
    return out


def select_side_records(pool: list[dict], n_per_side: int = 60, n_both: int = 20,
                        max_per_question_per_side: int = 8, seed: int = 0) -> list[dict]:
    """Open questions whose short gold answer names a side, balanced left/right.

    `max_per_question_per_side` stops one phrasing ("Where is the abnormality
    in this image?") from dominating a side.
    """
    candidates: dict[str, list[dict]] = {"left": [], "right": [], "both": []}
    for r in pool:
        if r["answer_format"] != "open" or r["is_knowledge"]:
            continue
        gold = r["gold_answer"]
        side = side_of(gold)
        if side is None or len(gold.split()) > 4:
            continue
        candidates[side].append({**r, "gold_side": side,
                                 "side_granularity": "side_only" if len(gold.split()) == 1 else "region"})

    rng = random.Random(seed)
    chosen: list[dict] = []
    for side, limit in (("left", n_per_side), ("right", n_per_side), ("both", n_both)):
        rows = candidates[side][:]
        rng.shuffle(rows)
        per_q: dict[str, int] = {}
        taken = []
        for r in rows:
            k = r["question"].lower().strip()
            if per_q.get(k, 0) >= max_per_question_per_side:
                continue
            per_q[k] = per_q.get(k, 0) + 1
            taken.append(r)
            if len(taken) >= limit:
                break
        chosen += taken
    rng.shuffle(chosen)
    return chosen


def split_dev_heldout(records: list[dict], dev_fraction: float = 0.3) -> tuple[list[dict], list[dict]]:
    """Deterministic split by image so no image appears on both sides.

    Retrieval design (what to retrieve, how many passages, prompt wording) may
    be tuned on the dev part only; the held-out part is scored once, at the end.
    Without this, a small test set gets tuned until it shows an improvement.
    """
    dev, held = [], []
    for r in records:
        bucket = int(hashlib.md5(_image_identity(r).encode()).hexdigest(), 16) % 1000
        (dev if bucket < dev_fraction * 1000 else held).append(r)
    return dev, held


def attach_images(records: list[dict], images_dir: str | Path) -> list[dict]:
    slake = download_slake_images([r for r in records if r["source_dataset"] == "SLAKE"], images_dir)
    vqa = download_vqa_rad_images([r for r in records if r["source_dataset"] == "VQA-RAD"], images_dir)
    for r in records:
        paths = slake if r["source_dataset"] == "SLAKE" else vqa
        r["image_path"] = paths[r["image_key"]]
        r["id"] = r["uid"]
    return records


def describe_set(records: list[dict]) -> dict:
    """Counts used in the documentation of how each set is composed."""
    def count(key):
        out: dict[str, int] = {}
        for r in records:
            out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
        return dict(sorted(out.items()))

    return {
        "n": len(records),
        "by_source": count("source_dataset"),
        "by_split": count("source_split"),
        "by_subtype": count("knowledge_subtype") if records and "knowledge_subtype" in records[0] else None,
        "by_gold_side": count("gold_side") if records and "gold_side" in records[0] else None,
        "unique_questions": len({r["question"].lower().strip() for r in records}),
        "unique_gold_answers": len({r["gold_answer"].lower().strip() for r in records}),
        "unique_images": len({_image_identity(r) for r in records}),
    }


def build_knowledge_test_set(images_dir: str | Path, include_train: bool = True,
                             cache_path: str | Path | None = None, pool: list[dict] | None = None) -> list[dict]:
    """Pass `pool` (from `fetch_full_pool`) to build several sets from one fetch."""
    pool = pool if pool is not None else fetch_full_pool(include_train=include_train, cache_path=cache_path)
    return attach_images(select_knowledge_records(pool), images_dir)


def build_side_test_set(images_dir: str | Path, n_per_side: int = 60, n_both: int = 20, seed: int = 0,
                        include_train: bool = True, cache_path: str | Path | None = None,
                        pool: list[dict] | None = None) -> list[dict]:
    pool = pool if pool is not None else fetch_full_pool(include_train=include_train, cache_path=cache_path)
    return attach_images(select_side_records(pool, n_per_side=n_per_side, n_both=n_both, seed=seed), images_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the knowledge and side test sets (no GPU needed).")
    parser.add_argument("--out", default="data")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--skip-images", action="store_true", help="metadata only, no image download")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = fetch_full_pool(cache_path=out / "cache" / "pools_full.json")
    sets = {"knowledge": select_knowledge_records(pool), "side": select_side_records(pool)}
    summary = {}
    for name, records in sets.items():
        if not args.skip_images:
            attach_images(records, args.images_dir)
        else:
            for r in records:
                r["id"] = r["uid"]
        with open(out / f"{name}_test_set.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summary[name] = describe_set(records)
        print(name, json.dumps(summary[name], indent=2))
    (out / "extra_sets_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
