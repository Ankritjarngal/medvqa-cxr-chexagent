"""Answer scoring for the baseline run.

Med-Flamingo's own finding (clinician ratings disagreeing sharply with
automated text-overlap metrics) is the reason this module reports *two*
different automated scores side by side rather than one. Neither is a
substitute for a human spot-check, which `run_baseline.py` also samples for.

- `exact_match`: strict, normalized string equality. Good for closed
  yes/no questions, harsh/unfair for open-ended ones.
- `loose_match`: substring-containment either direction, after
  normalization. More forgiving for open-ended questions, more generous
  than it should be on a few easy cases. Reported as a second column, not a
  replacement for exact_match, so the gap between the two is itself a
  signal worth looking at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_YES_SYNONYMS = {"yes", "y", "yeah", "affirmative", "true", "present"}
_NO_SYNONYMS = {"no", "n", "nope", "negative", "false", "absent", "not present"}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if text in _YES_SYNONYMS:
        return "yes"
    if text in _NO_SYNONYMS:
        return "no"
    return text


def exact_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def loose_match(prediction: str, gold: str) -> bool:
    p, g = normalize(prediction), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Guard against trivially short strings matching almost anything
    # ("a" inside "a tiny hazy opacity") by requiring the shorter string
    # to be at least 3 characters before allowing substring containment.
    shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
    if len(shorter) < 3:
        return False
    return shorter in longer


@dataclass
class CategoryStats:
    total: int = 0
    exact_correct: int = 0
    loose_correct: int = 0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.total if self.total else float("nan")

    @property
    def loose_accuracy(self) -> float:
        return self.loose_correct / self.total if self.total else float("nan")


@dataclass
class EvaluationReport:
    per_category: dict[str, CategoryStats] = field(default_factory=dict)
    per_source: dict[str, CategoryStats] = field(default_factory=dict)
    overall: CategoryStats = field(default_factory=CategoryStats)
    rows: list[dict] = field(default_factory=list)


def evaluate(records: list[dict]) -> EvaluationReport:
    """Score a list of records.

    Each record must have: 'question', 'gold_answer', 'prediction',
    'category', 'source_dataset'. Returns an EvaluationReport with overall
    and per-category / per-source breakdowns, plus the raw per-row scoring
    so a human can spot-check individual disagreements.
    """
    report = EvaluationReport()

    for rec in records:
        em = exact_match(rec["prediction"], rec["gold_answer"])
        lm = loose_match(rec["prediction"], rec["gold_answer"])

        for bucket_dict, key in (
            (report.per_category, rec["category"]),
            (report.per_source, rec["source_dataset"]),
        ):
            stats = bucket_dict.setdefault(key, CategoryStats())
            stats.total += 1
            stats.exact_correct += int(em)
            stats.loose_correct += int(lm)

        report.overall.total += 1
        report.overall.exact_correct += int(em)
        report.overall.loose_correct += int(lm)

        report.rows.append({**rec, "exact_match": em, "loose_match": lm})

    return report


def format_report(report: EvaluationReport) -> str:
    lines = []
    lines.append("=== Overall ===")
    lines.append(
        f"n={report.overall.total}  exact_acc={report.overall.exact_accuracy:.1%}"
        f"  loose_acc={report.overall.loose_accuracy:.1%}"
    )
    lines.append("")
    lines.append("=== By question category (the key breakdown) ===")
    for cat, stats in sorted(report.per_category.items()):
        lines.append(
            f"{cat:>12}: n={stats.total:<4} exact_acc={stats.exact_accuracy:.1%}"
            f"  loose_acc={stats.loose_accuracy:.1%}"
        )
    lines.append("")
    lines.append("=== By source dataset ===")
    for src, stats in sorted(report.per_source.items()):
        lines.append(
            f"{src:>10}: n={stats.total:<4} exact_acc={stats.exact_accuracy:.1%}"
            f"  loose_acc={stats.loose_accuracy:.1%}"
        )
    return "\n".join(lines)
