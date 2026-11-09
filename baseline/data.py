"""Build a small, chest-X-ray-only VQA test set for Phase 1 baseline testing.

Important caveat (documented here, not hidden): CheXagent's own training
corpus (CheXinstruct) explicitly rolls in VQA-RAD, SLAKE, and Open-I/IU X-ray
among its 32 source datasets. So evaluating CheXagent on these test splits is
*not* a clean held-out/zero-shot measurement -- it is close to in-distribution.
That is simply the state of public CXR VQA benchmarks today (virtually every
other paper in our review reuses the same splits for the same reason).

The point of Phase 1 is not to claim a novel generalization number; it is to
check whether our own baseline run reproduces the *specific, already-known*
weak spot from CheXagent's paper (strong on coarse/general questions, weaker
on laterality/severity) on data we control end-to-end, so Phase 2 (RAG) has a
concrete, measured gap to target rather than a hypothetical one.

Two sources are combined:
- SLAKE (test + validation): has structured `location` + `modality` metadata, so
  chest X-ray rows can be filtered precisely (no guessing from question text).
- VQA-RAD: has no organ metadata, so chest rows are selected with a keyword
  heuristic (noisier, clearly flagged below -- not a silent assumption).

Non-clinical template questions (e.g. SLAKE's radiation-damage question) are
dropped, repeated question text is capped per category, and every record gets a
stable `id` taken from the source dataset's own row id so later runs (RAG) can
be compared with this baseline row by row.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import zipfile
from pathlib import Path

import requests

from .categorize import answer_format, categorize_question, is_knowledge_question, is_non_clinical_question

SLAKE_DATASET = "BoKelvin/SLAKE"
SLAKE_IMAGES_ZIP_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip"
VQA_RAD_DATASET = "flaviagiammarino/vqa-rad"
DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"

# Preferred data path: download each dataset's own files once (5 requests).
# The paging API (`datasets-server`, ~150 requests) rate-limits Colab with HTTP
# 429, so it is kept only as a fallback. Row ids are identical on both paths,
# checked against all 224 question ids of run 2.
HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SLAKE_SPLIT_FILES = {"test": "test.json", "validation": "validation.json", "train": "train.json"}
VQA_RAD_PARQUET_FILES = {
    "test": "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet",
    "train": "data/train-00000-of-00001-eb8844602202be60.parquet",
}
VQA_RAD_PARQUET_KEY_PREFIX = "vqarad-parquet"
RAW_CACHE_DIR = Path("raw_cache")

# SLAKE's `location` field uses these labels for chest X-ray images.
SLAKE_CHEST_LOCATIONS = {"Chest_heart", "Chest_lung", "Chest_mediastinal", "Lung"}

# VQA-RAD keyword heuristic: a question must mention a chest-related term and
# must NOT mention a clearly non-chest body region. This is approximate --
# some true negatives/positives will slip through -- which is why SLAKE
# (precise metadata) is the primary source and VQA-RAD is secondary/noisier.
VQA_RAD_CHEST_KEYWORDS = {
    "lung", "lungs", "chest", "pulmonary", "pleural", "pneumothorax",
    "cardiac", "heart", "mediastin", "rib", "costophrenic", "consolidation",
    "effusion", "atelectasis", "diaphragm", "thora", "cardiomegaly",
    "pneumonia", "hilar", "bronch",
}
# Keywords that must match as a whole word. "rib" is the one that matters: as a
# substring it matched "describe", "circumscribed", "distribution" and
# "tributaries", which let brain/abdominal "describe the lesion" questions into
# the chest set (found by looking at the images of the side set).
VQA_RAD_WHOLE_WORD_KEYWORDS = {"rib", "ribs"}
_WHOLE_WORD_RE = re.compile(r"\b(" + "|".join(sorted(VQA_RAD_WHOLE_WORD_KEYWORDS)) + r")\b")

VQA_RAD_EXCLUDE_KEYWORDS = {
    "brain", "skull", "head", "abdomen", "abdominal", "liver", "kidney",
    "pelvis", "pelvic", "spine", "spinal", "bowel", "intestine", "stomach",
    "bladder", "uterus", "ovary", "appendix", "cerebral", "cranial", "sinus",
    # Added after the pilot run surfaced non-chest items (e.g. "vascular
    # distribution of these infarcts"). Still a heuristic -- review the images.
    "infarct", "ventric", "sulci", "gyri", "white matter", "gray matter",
    "grey matter", "falx", "midline", "mass effect", "vascular distribution",
    "hepatic", "splenic", "spleen", "adrenal", "pancrea", "gallbladder",
    "renal", "psoas", "femur", "knee", "shoulder", "orbit",
}


def mentions_chest_term(question: str) -> bool:
    """True if the question text contains a chest keyword (substring match,
    except for short words listed in VQA_RAD_WHOLE_WORD_KEYWORDS)."""
    q = question.lower()
    if _WHOLE_WORD_RE.search(q):
        return True
    return any(k in q for k in VQA_RAD_CHEST_KEYWORDS if k not in VQA_RAD_WHOLE_WORD_KEYWORDS)


def _retry_wait(resp, attempt: int, backoff: float) -> float:
    """Seconds to wait before retrying. Honours Retry-After on HTTP 429 and
    otherwise backs off exponentially (capped at 60 s)."""
    if resp is not None and resp.status_code == 429:
        try:
            return min(120.0, max(float(resp.headers.get("Retry-After", 0)), backoff * 4 * (attempt + 1)))
        except ValueError:
            pass
    return min(60.0, backoff * (2 ** attempt))


def _download_file(url: str, dest: str | Path, retries: int = 8, backoff: float = 2.0) -> Path:
    """Download `url` to `dest` once (skipped if present), streaming, with
    retries on 429 / 5xx / connection errors. Written via a temp file so an
    interrupted download is never mistaken for a complete one."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(retries):
        resp = None
        try:
            resp = requests.get(url, stream=True, timeout=180)
            if resp.status_code == 200:
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return dest
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
        except requests.RequestException as exc:
            last_exc = exc
            if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
                raise
        time.sleep(_retry_wait(resp, attempt, backoff))
    raise last_exc or RuntimeError(f"could not download {url}")


def _get_rows(dataset: str, split: str, offset: int, length: int = 100,
              retries: int = 8, backoff: float = 1.5) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                DATASETS_SERVER_ROWS_URL,
                params={
                    "dataset": dataset, "config": "default", "split": split,
                    "offset": offset, "length": length,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            resp = None
        time.sleep(_retry_wait(resp, attempt, backoff))
    if last_exc:
        raise last_exc
    resp.raise_for_status()
    return {}


def _iter_all_rows(dataset: str, split: str, page_size: int = 100):
    offset = 0
    while True:
        data = _get_rows(dataset, split, offset, length=page_size)
        rows = data.get("rows", [])
        if not rows:
            return
        for row in rows:
            yield {**row["row"], "_row_idx": row["row_idx"]}
        offset += len(rows)
        if offset >= data.get("num_rows_total", 0):
            return


def _iter_slake_rows(split: str):
    """SLAKE rows from the dataset's own JSON file; paging API as fallback."""
    try:
        url = HF_RESOLVE_URL.format(repo=SLAKE_DATASET, path=SLAKE_SPLIT_FILES[split])
        path = _download_file(url, RAW_CACHE_DIR / f"slake_{split}.json")
        rows = json.loads(path.read_text())
    except Exception as exc:
        print(f"[data] SLAKE {split}: file download failed ({exc!r}); falling back to the paging API")
        yield from _iter_all_rows(SLAKE_DATASET, split)
        return
    yield from rows


def fetch_slake_chest_records(split: str = "test") -> list[dict]:
    """English chest-X-ray rows from SLAKE, filtered via structured metadata."""
    records = []
    for rr in _iter_slake_rows(split):
        if (
            rr.get("modality") == "X-Ray"
            and rr.get("location") in SLAKE_CHEST_LOCATIONS
            and rr.get("q_lang") == "en"
            and not is_non_clinical_question(rr["question"])
        ):
            records.append({
                "source_dataset": "SLAKE",
                "source_split": split,
                "question": rr["question"],
                "gold_answer": rr["answer"],
                "answer_type": rr.get("answer_type"),
                "uid": f"SLAKE_{rr['qid']}",
                "image_key": rr["img_name"],  # e.g. "xmlab102/source.jpg"
                "category": categorize_question(rr["question"]),
                "answer_format": answer_format(rr["answer"]),
                "is_knowledge": is_knowledge_question(rr["question"]),
            })
    return records


def _vqa_rad_parquet_path(split: str) -> Path:
    url = HF_RESOLVE_URL.format(repo=VQA_RAD_DATASET, path=VQA_RAD_PARQUET_FILES[split])
    return _download_file(url, RAW_CACHE_DIR / f"vqa_rad_{split}.parquet")


def _iter_vqa_rad_rows(split: str):
    """VQA-RAD rows from the dataset's parquet file (images are embedded, so
    there are no expiring signed URLs, and an image's content hash gives a real
    image identity); paging API as fallback if pyarrow or the download fails.

    Yields dicts with question, answer, _row_idx, _image_key, _image_id."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(_vqa_rad_parquet_path(split))
        images, questions, answers = table.column("image"), table.column("question"), table.column("answer")
        rows = []
        for idx in range(table.num_rows):
            img_bytes = images[idx].as_py()["bytes"]
            rows.append({
                "question": questions[idx].as_py(), "answer": answers[idx].as_py(), "_row_idx": idx,
                "_image_key": f"{VQA_RAD_PARQUET_KEY_PREFIX}:{split}:{idx}",
                "_image_id": hashlib.md5(img_bytes).hexdigest()[:12],
            })
    except Exception as exc:
        print(f"[data] VQA-RAD {split}: parquet path failed ({exc!r}); falling back to the paging API")
        for rr in _iter_all_rows(VQA_RAD_DATASET, split):
            yield {"question": rr["question"], "answer": rr["answer"], "_row_idx": rr["_row_idx"],
                   "_image_key": rr["image"]["src"], "_image_id": None}
        return
    yield from rows


def fetch_vqa_rad_chest_records(splits: tuple[str, ...] = ("test", "train")) -> list[dict]:
    """Chest-ish rows from VQA-RAD, filtered via keyword heuristic (noisier)."""
    records = []
    for split in splits:
        for rr in _iter_vqa_rad_rows(split):
            q = rr["question"].lower()
            if (
                mentions_chest_term(q)
                and not any(k in q for k in VQA_RAD_EXCLUDE_KEYWORDS)
                and not is_non_clinical_question(rr["question"])
            ):
                records.append({
                    "source_dataset": "VQA-RAD",
                    "source_split": split,
                    "question": rr["question"],
                    "gold_answer": rr["answer"],
                    "answer_type": None,
                    "uid": f"VQA-RAD_{split}_{rr['_row_idx']}",
                    "image_key": rr["_image_key"],
                    "image_id": rr["_image_id"],  # content hash of the image (None on the fallback path)
                    "category": categorize_question(rr["question"]),
                    "answer_format": answer_format(rr["answer"]),
                    "is_knowledge": is_knowledge_question(rr["question"]),
                })
    return records


def stratified_sample(records: list[dict], n_per_category: int, seed: int = 0,
                      max_per_question: int = 4) -> list[dict]:
    """Sample up to `n_per_category` records per category.

    `max_per_question` caps how many rows with identical question text can
    be drawn (the pilot run had "is the heart enlarged?" ~12 times in 60 rows,
    which lets one question dominate a category). Deterministic for a given
    input order and seed.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    sampled = []
    for cat in sorted(by_cat):
        shuffled = by_cat[cat][:]
        rng.shuffle(shuffled)
        per_question: dict[str, int] = {}
        taken = []
        for r in shuffled:
            key = r["question"].lower().strip()
            if per_question.get(key, 0) >= max_per_question:
                continue
            per_question[key] = per_question.get(key, 0) + 1
            taken.append(r)
            if len(taken) >= n_per_category:
                break
        sampled.extend(taken)
    rng.shuffle(sampled)
    return sampled


def pool_summary(records: list[dict]) -> dict[str, dict[str, int]]:
    """Counts per category / source / answer format, so it is visible how many
    questions of each kind actually exist (severity questions are rare in
    public CXR VQA sets -- that is itself a finding)."""
    summary: dict[str, dict[str, int]] = {"category": {}, "source": {}, "answer_format": {}}
    for r in records:
        for field, key in (("category", r["category"]), ("source", r["source_dataset"]),
                           ("answer_format", r["answer_format"])):
            summary[field][key] = summary[field].get(key, 0) + 1
    return summary


def download_slake_images(records: list[dict], dest_dir: str | Path,
                           zip_cache_path: str | Path | None = None) -> dict[str, str]:
    """Download SLAKE's imgs.zip once (cached) and extract only needed files."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    needed_keys = {r["image_key"] for r in records if r["source_dataset"] == "SLAKE"}
    if not needed_keys:
        return {}

    zip_cache_path = Path(zip_cache_path) if zip_cache_path else dest_dir / "_slake_imgs.zip"
    _download_file(SLAKE_IMAGES_ZIP_URL, zip_cache_path)

    local_paths = {}
    with zipfile.ZipFile(zip_cache_path) as zf:
        for key in needed_keys:
            member = f"imgs/{key}"
            out_path = dest_dir / key.replace("/", "__")
            if not out_path.exists():
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            local_paths[key] = str(out_path)
    return local_paths


def download_vqa_rad_images(records: list[dict], dest_dir: str | Path) -> dict[str, str]:
    """Write each needed VQA-RAD image to `dest_dir`. Records from the parquet
    path are extracted from the parquet file and named by image content hash, so
    questions about the same image share one file. Records from the paging
    fallback are fetched from their (expiring) signed URLs."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_paths: dict[str, str] = {}
    tables: dict[str, object] = {}
    for r in records:
        if r["source_dataset"] != "VQA-RAD":
            continue
        key = r["image_key"]
        if key in local_paths:
            continue
        if key.startswith(VQA_RAD_PARQUET_KEY_PREFIX + ":"):
            _, split, idx = key.split(":")
            out_path = dest_dir / f"vqarad_{r['image_id']}.jpg"
            if not out_path.exists():
                if split not in tables:
                    import pyarrow.parquet as pq
                    tables[split] = pq.read_table(_vqa_rad_parquet_path(split), columns=["image"])
                out_path.write_bytes(tables[split].column("image")[int(idx)].as_py()["bytes"])
        else:
            digest = hashlib.md5(key.encode()).hexdigest()[:12]
            out_path = dest_dir / f"vqarad_{digest}.jpg"
            if not out_path.exists():
                resp = requests.get(key, timeout=60)
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(resp.content)
        local_paths[key] = str(out_path)
    return local_paths


def build_baseline_test_set(images_dir: str | Path, n_per_category: int = 100,
                             seed: int = 0, max_per_question: int = 4,
                             verbose: bool = True) -> list[dict]:
    """End-to-end: fetch metadata, stratify-sample per category, download
    images, return records ready for model inference. Each record has an
    `image_path` and a stable `id` (derived from the source dataset's own row
    id, so baseline and later RAG runs can be compared row by row)."""
    slake = (fetch_slake_chest_records(split="test")
             + fetch_slake_chest_records(split="validation"))
    vqa_rad = fetch_vqa_rad_chest_records(splits=("test", "train"))
    all_records = slake + vqa_rad
    if verbose:
        print("Available pool before sampling:", pool_summary(all_records))

    sampled = stratified_sample(all_records, n_per_category=n_per_category, seed=seed,
                                max_per_question=max_per_question)

    slake_records = [r for r in sampled if r["source_dataset"] == "SLAKE"]
    vqa_rad_records = [r for r in sampled if r["source_dataset"] == "VQA-RAD"]
    slake_paths = download_slake_images(slake_records, images_dir)
    vqa_rad_paths = download_vqa_rad_images(vqa_rad_records, images_dir)

    for r in sampled:
        paths = slake_paths if r["source_dataset"] == "SLAKE" else vqa_rad_paths
        r["image_path"] = paths[r["image_key"]]
        r["id"] = r["uid"]

    return sampled
