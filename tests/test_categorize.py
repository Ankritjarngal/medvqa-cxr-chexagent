from baseline.categorize import categorize_question


def test_laterality_keywords():
    assert categorize_question("Which side is the effusion on?") == "laterality"
    assert categorize_question("Is there a left-sided pneumothorax?") == "laterality"
    assert categorize_question("Are the findings bilateral?") == "laterality"


def test_severity_keywords():
    assert categorize_question("Is the cardiomegaly mild or severe?") == "severity"
    assert categorize_question("Is the heart enlarged?") == "severity"
    assert categorize_question("How large is the lesion?") == "severity"


def test_general_fallback():
    assert categorize_question("What modality is this image?") == "general"
    assert categorize_question("Is there a fracture?") == "general"


def test_laterality_takes_priority_over_severity():
    # Contains both a laterality word ("left") and a severity word ("severe").
    assert categorize_question("Is the left-sided effusion severe?") == "laterality"
