"""Answer scoring for the baseline run.

Med-Flamingo's own finding (clinician ratings disagreeing sharply with
automated text-overlap metrics) is why several automated scores are reported
side by side rather than one. None is a substitute for a human spot-check.

- `exact_match`: strict, normalized string equality. Fair for yes/no, harsh for
  open-ended answers.
- `loose_match`: substring containment either direction after normalization.
- `token_f1`: word-overlap F1 (SQuAD style). Gives partial credit to open
  answers like "Chest tightness, shortness of breath" vs "Chest tightness,
  fatigue", which exact_match scores 0.

Every accuracy is reported with a Wilson 95% confidence interval, because with
a few dozen questions per bucket most differences between buckets are not
statistically meaningful, and the report should say so.

Yes/no questions get extra treatment (sensitivity/specificity and the
"always answer the majority class" baseline): a model that answers "no" to
almost everything scores well when most true answers are "no", but misses
real findings.
"""

from __future__ import annotations

import math
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
    # Guard against trivially short strings matching almost anything.
    shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
    if len(shorter) < 3:
        return False
    return shorter in longer


def token_f1(prediction: str, gold: str) -> float:
    p_tokens, g_tokens = normalize(prediction).split(), normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    remaining = list(g_tokens)
    common = 0
    for tok in p_tokens:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p_tokens), common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class Stats:
    total: int = 0
    exact_correct: int = 0
    loose_correct: int = 0
    f1_sum: float = 0.0

    def add(self, em: bool, lm: bool, f1: float) -> None:
        self.total += 1
        self.exact_correct += int(em)
        self.loose_correct += int(lm)
        self.f1_sum += f1

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.total if self.total else float("nan")

    @property
    def loose_accuracy(self) -> float:
        return self.loose_correct / self.total if self.total else float("nan")

    @property
    def mean_f1(self) -> float:
        return self.f1_sum / self.total if self.total else float("nan")

    @property
    def exact_ci(self) -> tuple[float, float]:
        return wilson_interval(self.exact_correct, self.total)


# Backwards-compatible name used by earlier notebook versions and tests.
CategoryStats = Stats


@dataclass
class YesNoStats:
    """Confusion counts for yes/no questions, treating 'yes' as the positive
    class (i.e. 'there is a finding')."""
    tp: int = 0   # gold yes, predicted yes
    fn: int = 0   # gold yes, predicted not-yes  (missed finding)
    tn: int = 0   # gold no,  predicted no
    fp: int = 0   # gold no,  predicted not-no   (false alarm)

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.tn + self.fp

    @property
    def sensitivity(self) -> float:
        pos = self.tp + self.fn
        return self.tp / pos if pos else float("nan")

    @property
    def specificity(self) -> float:
        neg = self.tn + self.fp
        return self.tn / neg if neg else float("nan")

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else float("nan")

    @property
    def majority_baseline(self) -> float:
        """Accuracy of always answering the more common gold class."""
        return max(self.tp + self.fn, self.tn + self.fp) / self.total if self.total else float("nan")

    @property
    def predicted_yes_rate(self) -> float:
        return (self.tp + self.fp) / self.total if self.total else float("nan")

    @property
    def gold_yes_rate(self) -> float:
        return (self.tp + self.fn) / self.total if self.total else float("nan")


@dataclass
class EvaluationReport:
    per_category: dict[str, Stats] = field(default_factory=dict)
    per_source: dict[str, Stats] = field(default_factory=dict)
    per_format: dict[str, Stats] = field(default_factory=dict)
    per_category_format: dict[str, Stats] = field(default_factory=dict)
    per_kind: dict[str, Stats] = field(default_factory=dict)
    overall: Stats = field(default_factory=Stats)
    yes_no: YesNoStats = field(default_factory=YesNoStats)
    yes_no_by_category: dict[str, YesNoStats] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)


def _gold_format(rec: dict) -> str:
    if rec.get("answer_format"):
        return rec["answer_format"]
    return "yes/no" if normalize(rec["gold_answer"]) in ("yes", "no") else "open"


def evaluate(records: list[dict]) -> EvaluationReport:
    """Score a list of records.

    Required keys: 'question', 'gold_answer', 'prediction', 'category',
    'source_dataset'. Optional: 'answer_format' ('yes/no'|'open', inferred if
    absent) and 'is_knowledge' (bool). Returns overall and per-bucket stats plus
    per-row scoring so a human can inspect individual disagreements.
    """
    report = EvaluationReport()

    for rec in records:
        em = exact_match(rec["prediction"], rec["gold_answer"])
        lm = loose_match(rec["prediction"], rec["gold_answer"])
        f1 = token_f1(rec["prediction"], rec["gold_answer"])
        fmt = _gold_format(rec)
        kind = "knowledge" if rec.get("is_knowledge") else "perception"

        for bucket, key in (
            (report.per_category, rec["category"]),
            (report.per_source, rec["source_dataset"]),
            (report.per_format, fmt),
            (report.per_category_format, f"{rec['category']} / {fmt}"),
            (report.per_kind, kind),
        ):
            bucket.setdefault(key, Stats()).add(em, lm, f1)
        report.overall.add(em, lm, f1)

        if fmt == "yes/no":
            gold_yes = normalize(rec["gold_answer"]) == "yes"
            pred = normalize(rec["prediction"])
            for yn in (report.yes_no, report.yes_no_by_category.setdefault(rec["category"], YesNoStats())):
                if gold_yes:
                    yn.tp += int(pred == "yes")
                    yn.fn += int(pred != "yes")
                else:
                    yn.tn += int(pred == "no")
                    yn.fp += int(pred != "no")

        report.rows.append({**rec, "answer_format": fmt, "exact_match": em,
                            "loose_match": lm, "token_f1": round(f1, 3)})

    return report


def _fmt_stats(s: Stats) -> str:
    lo, hi = s.exact_ci
    return (f"n={s.total:<4} exact={s.exact_accuracy:6.1%} (95% CI {lo:.0%}-{hi:.0%})"
            f"  loose={s.loose_accuracy:6.1%}  token_f1={s.mean_f1:.2f}")


def format_report(report: EvaluationReport) -> str:
    L = []
    L.append("=== Overall ===")
    L.append(_fmt_stats(report.overall))

    def section(title, bucket):
        L.append("")
        L.append(f"=== {title} ===")
        for key, s in sorted(bucket.items()):
            L.append(f"{key:>24}: {_fmt_stats(s)}")

    section("By question category (laterality / severity / general)", report.per_category)
    section("By answer format (yes/no vs open-ended)", report.per_format)
    section("By category and answer format", report.per_category_format)
    section("By question kind (knowledge = symptoms/treatment/etc; perception = look at the image)", report.per_kind)
    section("By source dataset", report.per_source)

    yn = report.yes_no
    L.append("")
    L.append("=== Yes/no questions: does the model miss findings? ===")
    if yn.total:
        L.append(f"n={yn.total}  accuracy={yn.accuracy:.1%}  "
                 f"always-answer-majority baseline={yn.majority_baseline:.1%}")
        L.append(f"sensitivity (correct on gold 'yes')={yn.sensitivity:.1%}  "
                 f"specificity (correct on gold 'no')={yn.specificity:.1%}")
        L.append(f"gold 'yes' rate={yn.gold_yes_rate:.1%}  model predicted 'yes' rate={yn.predicted_yes_rate:.1%}")
        for cat, c in sorted(report.yes_no_by_category.items()):
            L.append(f"  {cat:>12}: n={c.total:<3} acc={c.accuracy:.0%}  sens={c.sensitivity:.0%}  spec={c.specificity:.0%}")
    else:
        L.append("(no yes/no questions in this run)")
    L.append("")
    L.append("Note: buckets with n < ~50 have wide confidence intervals; treat differences that "
             "the intervals overlap on as unproven.")
    return "\n".join(L)
