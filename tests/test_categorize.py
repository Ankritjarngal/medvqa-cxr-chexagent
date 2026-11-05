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


from baseline.categorize import answer_format, is_knowledge_question, is_non_clinical_question


def test_non_clinical_template_is_detected():
    assert is_non_clinical_question("How much damage will be caused to the body taking such an image?")
    assert not is_non_clinical_question("Is the heart enlarged?")


def test_knowledge_vs_perception():
    assert is_knowledge_question("What kind of symptoms will the disease on the upper right lung bring?")
    assert is_knowledge_question("How to treat the disease located on the lower left of lung?")
    assert is_knowledge_question("how do you determine cardiomegaly?")
    assert not is_knowledge_question("Which side of lung is abnormal in this image, left or right?")
    assert not is_knowledge_question("Is the heart enlarged?")


def test_answer_format():
    assert answer_format("Yes") == "yes/no"
    assert answer_format("no.") == "yes/no"
    assert answer_format("right") == "open"
