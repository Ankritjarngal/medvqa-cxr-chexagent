"""Heuristic question-type categorization.

The literature review (CheXagent in particular) found that CXR VLMs score
~90-99% on coarse perception questions (presence/absence, modality, view,
organ identification) but drop to ~77-79% on fine-grained laterality and
severity questions ("which side", "mild vs. severe"). To test whether our
baseline shows the same weak spot, every question in the baseline test set
is tagged with a category up front so results can be broken down by category
rather than reported as one aggregate accuracy number.

This is a deliberately simple keyword heuristic, not a trained classifier.
It will mis-tag some edge cases (e.g. "what is to the left of the heart" is
positional, not a laterality-of-finding question) -- that is an accepted,
documented limitation, not a bug to silently paper over.
"""

from __future__ import annotations

import re

LATERALITY_PATTERNS = [
    r"\bleft\b",
    r"\bright\b",
    r"\bbilateral\b",
    r"\bwhich side\b",
    r"\blaterality\b",
    r"\bunilateral\b",
]

SEVERITY_PATTERNS = [
    r"\bmild\b",
    r"\bmoderate\b",
    r"\bsevere\b",
    r"\benlarge(d|ment)?\b",
    r"\bdegree\b",
    r"\bextent\b",
    r"\bhow (much|severe|large|big)\b",
    r"\bsize of\b",
    r"\bworsen(ed|ing)?\b",
    r"\bimprov(ed|ing|ement)\b",
    r"\bsmall\b",
    r"\blarge\b",
]

CATEGORY_LATERALITY = "laterality"
CATEGORY_SEVERITY = "severity"
CATEGORY_GENERAL = "general"

ALL_CATEGORIES = (CATEGORY_LATERALITY, CATEGORY_SEVERITY, CATEGORY_GENERAL)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def categorize_question(question: str) -> str:
    """Tag a question as 'laterality', 'severity', or 'general'.

    Laterality is checked before severity because a question can contain
    both kinds of words ("is the left-sided effusion severe?") -- in our
    taxonomy laterality takes priority since it is the harder, more
    specifically-flagged failure mode in CheXagent's own reported numbers.
    """
    q = question.lower().strip()
    if _matches_any(q, LATERALITY_PATTERNS):
        return CATEGORY_LATERALITY
    if _matches_any(q, SEVERITY_PATTERNS):
        return CATEGORY_SEVERITY
    return CATEGORY_GENERAL
