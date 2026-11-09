"""Retrieval for Phase 2: corpus loading, BM25, query and prompt construction, retrieval scoring.

Design and rules: docs/phase2_rag_plan.md and Amendment 5 in docs/evaluation_protocol.md.

Corpora (both independent of SLAKE; see the plan, section 8):
  * MedRAG textbooks on Hugging Face (18 textbooks chunked to <= 1000 characters). Research use only,
    downloaded at run time, never committed.
  * MedlinePlus Health Topics (National Library of Medicine, public domain), split into passages at
    each section heading ("What causes pneumonia?").

BM25 is implemented here (about 40 lines, no dependencies) so that the index is auditable and the
unit tests do not need extra packages.
"""

from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import requests

from .data import _download_file
from .disease_labels import NO_FINDING
from .knowledge import _STOPWORDS, _stem, content_tokens, split_items

RAG_CACHE_DIR = Path("rag_cache")

MEDRAG_TEXTBOOK_URL = "https://huggingface.co/datasets/MedRAG/textbooks/resolve/main/chunk/{name}.jsonl"
MEDRAG_TEXTBOOK_NAMES = [
    "Anatomy_Gray", "Biochemistry_Lippincott", "Cell_Biology_Alberts", "First_Aid_Step1", "First_Aid_Step2",
    "Gynecology_Novak", "Histology_Ross", "Immunology_Janeway", "InternalMed_Harrison", "Neurology_Adams",
    "Obstentrics_Williams", "Pathology_Robbins", "Pathoma_Husain", "Pediatrics_Nelson", "Pharmacology_Katzung",
    "Physiology_Levy", "Psichiatry_DSM-5", "Surgery_Schwartz",
]
MEDLINEPLUS_INDEX_URL = "https://medlineplus.gov/xml.html"
MEDLINEPLUS_XML_URL = "https://medlineplus.gov/xml/{name}"

# A textbook chunk is kept only if it mentions a chest disease or a chest structure. This keeps the index
# small enough to build in about a minute. It uses the ChestX-ray14 names, not SLAKE's labels.
CHEST_FILTER = re.compile(
    r"atelecta|cardiomegal|cardiac enlargement|consolidat|\bedema\b|effusion|emphysema|fibrosis|hernia|"
    r"infiltrat|\bmass(?:es)?\b|nodul|pleur|pneumonia|pneumothora|lung|pulmonary|bronch|thora|\bchest\b|"
    r"cardiac|heart|cardio", re.IGNORECASE)

# ------------------------------------------------------------------------------------ BM25


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a list of passages ({'text', 'title', ...}); title and text both count."""

    def __init__(self, passages: list[dict], k1: float = 1.5, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, p in enumerate(passages):
            toks = tokenize(f"{p.get('title', '')} {p['text']}")
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n = len(passages)
        self.avg_len = sum(self.doc_len) / self.n if self.n else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 3, topics: set[str] | None = None) -> list[tuple[float, dict]]:
        """`topics`, when given, restricts the search to passages whose `topic` field is in the set."""
        scores: dict[int, float] = defaultdict(float)
        # A term repeated in the query counts proportionally more, which is how a query puts extra weight on
        # the disease name (build_query, disease_weight).
        for term, qtf in Counter(tokenize(query)).items():
            idf = self.idf(term)
            for i, tf in self.postings.get(term, ()):
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                if topics is None or self.passages[i].get("topic") in topics:
                    scores[i] += qtf * idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(s, self.passages[i]) for i, s in ranked]


# --------------------------------------------------------------------------------- corpora


def load_textbook_passages(cache_dir: str | Path = RAG_CACHE_DIR, chest_only: bool = True) -> list[dict]:
    """MedRAG textbook chunks, optionally restricted to chest-related chunks (CHEST_FILTER)."""
    cache_dir = Path(cache_dir)
    out_path = cache_dir / ("textbooks_chest.jsonl" if chest_only else "textbooks_all.jsonl")
    if out_path.exists() and out_path.stat().st_size > 0:
        return [json.loads(line) for line in out_path.read_text().splitlines() if line]
    passages = []
    for name in MEDRAG_TEXTBOOK_NAMES:
        path = _download_file(MEDRAG_TEXTBOOK_URL.format(name=name), cache_dir / "raw" / f"{name}.jsonl")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("content") or row.get("contents") or ""
            if chest_only and not CHEST_FILTER.search(text):
                continue
            passages.append({"source": "textbooks", "id": row.get("id", ""), "title": row.get("title", name),
                             "text": " ".join(text.split())})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(p) for p in passages))
    return passages


def discover_medlineplus_xml_name(timeout: int = 60) -> str:
    """Name of the newest bulk file listed on MedlinePlus's XML page (the file name carries a date)."""
    page = requests.get(MEDLINEPLUS_INDEX_URL, timeout=timeout).text
    names = sorted(set(re.findall(r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml", page)))
    if not names:
        raise RuntimeError("no mplus_topics_*.xml file listed on " + MEDLINEPLUS_INDEX_URL)
    return names[-1]


def _clean_html(fragment: str) -> str:
    t = re.sub(r"</li>", "; ", fragment)
    t = re.sub(r"<[^>]+>", " ", t)
    return " ".join(html.unescape(t).replace("\xa0", " ").split())


def medlineplus_passages_from_xml(xml_text: str) -> list[dict]:
    """One passage per section of each English health topic. Section = text under one <h3> heading."""
    root = ET.fromstring(xml_text)
    out = []
    for topic in root:
        if topic.attrib.get("language") != "English":
            continue
        title = topic.attrib.get("title", "")
        summary = topic.findtext("full-summary") or ""
        parts = re.split(r"(?i)<h3>(.*?)</h3>", summary)
        sections = [("", parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
        for heading, body in sections:
            text = _clean_html(body)
            if len(text) < 40:
                continue
            head = _clean_html(heading)
            out.append({"source": "medlineplus", "id": f"{topic.attrib.get('id', '')}:{head[:40]}",
                        "title": f"{title}: {head}" if head else title, "text": text,
                        "topic": title, "url": topic.attrib.get("url", "")})
    return out


def load_medlineplus_passages(cache_dir: str | Path = RAG_CACHE_DIR) -> list[dict]:
    cache_dir = Path(cache_dir)
    existing = sorted(cache_dir.glob("mplus_topics_*.xml"))
    if existing:
        path = existing[-1]
    else:
        name = discover_medlineplus_xml_name()
        path = _download_file(MEDLINEPLUS_XML_URL.format(name=name), cache_dir / name)
    return medlineplus_passages_from_xml(path.read_text(encoding="utf-8"))


def build_retrievers(cache_dir: str | Path = RAG_CACHE_DIR) -> dict[str, BM25]:
    """One index per corpus: {'textbooks': BM25, 'medlineplus': BM25}."""
    return {"textbooks": BM25(load_textbook_passages(cache_dir)),
            "medlineplus": BM25(load_medlineplus_passages(cache_dir))}


# ------------------------------------------------------------------- queries and prompts

# Words added to the disease name per question type. Chosen from how textbooks and MedlinePlus
# headings are worded ("What causes ...", "Symptoms", "Treatment"), before any result was seen.
SUBTYPE_QUERY_WORDS = {
    "cause": "causes etiology",
    "symptoms": "symptoms signs clinical presentation",
    "treatment": "treatment management therapy",
    "prevention": "prevention prevent risk factors",
}

# canonical ChestX-ray14 name -> words used in the query. Revised once after a dev-only diagnosis (experiment log,
# entry 14): the first version used the radiology name alone and missed diseases that reference texts file under
# another name (cardiomegaly under cardiomyopathy / enlarged heart, atelectasis under collapsed lung).
QUERY_NAMES = {
    "Atelectasis": "atelectasis collapsed lung", "Cardiomegaly": "cardiomegaly enlarged heart cardiomyopathy",
    "Consolidation": "lung consolidation pneumonia", "Edema": "pulmonary edema", "Effusion": "pleural effusion",
    "Emphysema": "emphysema", "Fibrosis": "pulmonary fibrosis", "Hernia": "hiatal hernia",
    "Infiltration": "pulmonary infiltrates", "Mass": "lung mass tumor cancer",
    "Nodule": "pulmonary nodule lung cancer", "Pleural Thickening": "pleural thickening",
    "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax collapsed lung",
}
# MedlinePlus health-topic titles that cover each disease, chosen from the list of topic titles and from medical
# knowledge, not from any answer in the test sets. Used by the "medlineplus_topic" source.
MEDLINEPLUS_TOPICS = {
    "Atelectasis": {"Collapsed Lung"}, "Cardiomegaly": {"Cardiomyopathy", "Heart Failure"},
    "Consolidation": {"Pneumonia"}, "Edema": {"Edema"}, "Effusion": {"Pleural Disorders"},
    "Emphysema": {"Emphysema"}, "Fibrosis": {"Pulmonary Fibrosis"}, "Hernia": {"Hiatal Hernia"},
    "Infiltration": {"Pneumonia", "Interstitial Lung Diseases"}, "Mass": {"Lung Cancer"},
    "Nodule": {"Lung Cancer"}, "Pleural Thickening": {"Pleural Disorders"}, "Pneumonia": {"Pneumonia"},
    "Pneumothorax": {"Collapsed Lung"},
}
HINT_NAMES = {
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
    "Edema": "pulmonary edema", "Effusion": "pleural effusion", "Emphysema": "emphysema",
    "Fibrosis": "pulmonary fibrosis", "Hernia": "hiatal hernia", "Infiltration": "pulmonary infiltration",
    "Mass": "pulmonary mass", "Nodule": "pulmonary nodule", "Pleural Thickening": "pleural thickening",
    "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax",
}

STYLE_INSTRUCTION = "Answer with a short comma-separated list of 2 to 4 items."


def usable_disease(disease: str | None) -> bool:
    return bool(disease) and disease != NO_FINDING and disease in QUERY_NAMES


def build_query(question: str, subtype: str, disease: str | None, disease_weight: int = 1) -> str:
    """Disease plus question-type words when both are known; otherwise the question text itself.
    `disease_weight` repeats the disease name so that it outweighs the question-type words."""
    if usable_disease(disease) and subtype in SUBTYPE_QUERY_WORDS:
        name = " ".join([QUERY_NAMES[disease]] * max(1, disease_weight))
        return f"{name} {SUBTYPE_QUERY_WORDS[subtype]}"
    return question


def truncate_passage(text: str, max_chars: int) -> str:
    """Cut at a sentence end before max_chars when there is one in the second half, else at a word."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("; "))
    if stop > max_chars // 2:
        return cut[:stop + 1]
    return cut.rsplit(" ", 1)[0] + " ..."


def retrieve_passages(retriever: BM25, query: str, k: int, max_chars: int) -> list[dict]:
    hits = retriever.search(query, k=k)
    return [{**p, "score": round(s, 3), "text_used": truncate_passage(p["text"], max_chars)} for s, p in hits]


RETRIEVAL_SOURCES = ("textbooks", "medlineplus", "medlineplus_topic", "topic+textbooks")


def retrieve_context(retrievers: dict[str, BM25], source: str, question: str, subtype: str,
                     disease: str | None, k: int, max_chars: int, disease_weight: int = 1) -> list[dict]:
    """Passages for one question.
    textbooks / medlineplus: BM25 over that corpus with the query from `build_query`.
    medlineplus_topic: BM25 restricted to the MedlinePlus topics for the disease (plain BM25 when no disease).
    topic+textbooks: the best topic passage first, the rest from the textbooks."""
    query = build_query(question, subtype, disease, disease_weight)
    topics = MEDLINEPLUS_TOPICS.get(disease) if usable_disease(disease) else None

    def top(name: str, n: int, restrict=None) -> list[dict]:
        return [{**p, "score": round(sc, 3), "text_used": truncate_passage(p["text"], max_chars)}
                for sc, p in retrievers[name].search(query, k=n, topics=restrict)]

    if source in ("textbooks", "medlineplus"):
        return top(source, k)
    if source == "medlineplus_topic":
        return top("medlineplus", k, topics)
    if source == "topic+textbooks":
        return top("medlineplus", 1, topics) + (top("textbooks", k - 1) if k > 1 else [])
    raise ValueError(f"unknown source {source!r}")


def build_prompt(question: str, disease: str | None = None, passages: list[dict] | None = None) -> str:
    """The prompt for every Phase 2 arm. No disease and no passages gives the style-matched baseline (A1)."""
    parts = []
    if passages:
        parts.append("Reference: " + " ".join(p["text_used"] for p in passages))
    if usable_disease(disease):
        parts.append(f"The finding in this image is {HINT_NAMES[disease]}.")
    parts += [question, STYLE_INSTRUCTION]
    return "\n".join(parts)


# ----------------------------------------------------------------------- retrieval scoring

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def gold_item_coverage(passages: list[dict], gold_answer: str) -> dict[str, float]:
    """How much of the gold answer's item list the retrieved text holds, with no model involved.

    loose:  an item counts if all its content words appear anywhere in the retrieved text.
    strict: an item counts only if all its content words appear within one sentence.
    Both use the same synonym table and stemming as the answer scorer (scorer version 2)."""
    items = [content_tokens(i) for i in split_items(gold_answer)]
    items = [i for i in items if i]
    if not items:
        return {"loose": 0.0, "strict": 0.0}
    texts = [p.get("text_used", p["text"]) for p in passages]
    all_tokens = content_tokens(" ".join(texts))
    sentences = [content_tokens(s) for t in texts for s in _SENT_SPLIT.split(t)]
    loose = sum(i <= all_tokens for i in items) / len(items)
    strict = sum(any(i <= s for s in sentences) for i in items) / len(items)
    return {"loose": loose, "strict": strict}


# ------------------------------------------------------------- Stage 2: retrieval without a model

MAX_CONTEXT_CHARS = 1400  # about 350 tokens; the model's context is 2048 tokens including the image


def coverage_for_setting(records: list[dict], retrievers: dict[str, BM25], source: str, k: int, max_chars: int,
                         disease_weight: int = 1, disease_of=lambda r: r.get("oracle_disease")) -> list[dict]:
    """For each record: retrieve with the query built from `disease_of(record)` and score how much of the
    gold answer's item list the retrieved text holds. No model, no gold access except for scoring."""
    out = []
    for r in records:
        disease = disease_of(r)
        passages = retrieve_context(retrievers, source, r["question"], r["knowledge_subtype"], disease,
                                    k, max_chars, disease_weight)
        cov = gold_item_coverage(passages, r["gold_answer"])
        out.append({"id": r.get("id") or r.get("uid"), "subtype": r["knowledge_subtype"],
                    "query": build_query(r["question"], r["knowledge_subtype"], disease, disease_weight), **cov})
    return out


def mean_coverage(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    return {"n": n, "strict": sum(r["strict"] for r in rows) / n if n else float("nan"),
            "loose": sum(r["loose"] for r in rows) / n if n else float("nan")}


def choose_setting(results: dict[tuple, dict]) -> tuple:
    """results: (source, k, max_chars, disease_weight) -> mean_coverage on the dev questions.
    Highest strict coverage, then highest loose coverage, then the shortest context, then grid order."""
    return min(results, key=lambda s: (-results[s]["strict"], -results[s]["loose"], s[1] * s[2]))
