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

KNOWLEDGE_PATTERNS = [
    r"\bsymptoms?\b",
    r"\bhow to (treat|prevent|cure)\b",
    r"\b(treatment|therapy|prevention)\b",
    r"\bfunction of\b",
    r"\bcause[sd]?\b",
    r"\bhow do you (determine|diagnose|assess|measure)\b",
    # Added with the knowledge test set: 'why ...?' and 'how can you tell ...?'
    # are answered from anatomy/physiology knowledge. Run 2's saved tags were
    # produced before these two were added.
    r"\bwhy\b",
    r"\bhow can you tell\b",
]

# Questions that are templates about the imaging procedure, not about the
# patient's findings (e.g. SLAKE's "How much damage will be caused to the body
# taking such an image?"). They match the severity keywords ("how much") but are
# not clinical severity, so the data pipeline drops them.
NON_CLINICAL_PATTERNS = [
    r"\bdamage\b.*\btaking such an image\b",
    r"\bradiation\b",
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


def is_knowledge_question(question: str) -> bool:
    """True for questions answered mostly from medical knowledge (symptoms,
    treatment, prevention, how a finding is measured) rather than from looking
    at the image. These are what a retrieval layer can plausibly help with;
    perception questions (is X present, which side) are not.

    Independent of `categorize_question`: a question can be both a laterality
    question and a knowledge question ("How to treat the disease on the lower
    left of lung?")."""
    q = question.lower().strip()
    return _matches_any(q, KNOWLEDGE_PATTERNS)


_SUBTYPE_RULES = [
    ("symptoms", r"\bsymptoms?\b"),
    ("treatment", r"\b(how to treat|treatment|therapy)\b"),
    ("prevention", r"\b(prevent|prevention)\b"),
    ("cause", r"\bcause[sd]?\b"),
    ("function", r"\bfunction of\b"),
    ("explanation", r"\b(why|how can you tell|how do you (determine|diagnose|assess|measure))\b"),
]


def knowledge_subtype(question: str) -> str:
    """Which kind of knowledge question this is (symptoms / treatment /
    prevention / cause / function / explanation / other). Reported separately
    because the SLAKE knowledge questions come from a few fixed templates and
    a model can score well on one template and not another."""
    q = question.lower().strip()
    for name, pattern in _SUBTYPE_RULES:
        if re.search(pattern, q):
            return name
    return "other"


def is_non_clinical_question(question: str) -> bool:
    q = question.lower().strip()
    return _matches_any(q, NON_CLINICAL_PATTERNS)


def answer_format(gold_answer: str) -> str:
    """'yes/no' for closed questions, 'open' otherwise. Reported separately
    because a model that mostly answers 'no' looks decent on yes/no questions
    when most true answers are 'no'."""
    g = gold_answer.lower().strip().rstrip(".")
    return "yes/no" if g in ("yes", "no") else "open"
