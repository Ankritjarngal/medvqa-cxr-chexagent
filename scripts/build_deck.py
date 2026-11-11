"""Builds slides/Medical_VQA_CXR_Project_Presentation.pptx (12 slides, 16:9) from report/figures/*.png.

  python3 scripts/make_figures.py && python3 scripts/build_deck.py

Every slide: a title that states the message, one diagram, one short take-away. Title-slide names are placeholders
until the team supplies the official format.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

FIG = Path("report/figures")
OUT = Path("slides/Medical_VQA_CXR_Project_Presentation.pptx")
FONT = "Calibri"
NAVY, TEAL, ORANGE, GREY, LIGHT, WHITE = (RGBColor.from_string(c) for c in ("12263A", "0F8B8D", "E4572E", "6B7785", "EEF3F6", "FFFFFF"))

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, size=18, color=NAVY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = para if isinstance(para, list) else [(para, bold)]
        for t, b in parts:
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size, r.font.bold, r.font.italic = FONT, Pt(size), b, italic
            r.font.color.rgb = color
    return tb


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def content_slide(n, title, fig, takeaway, source=None, note="", max_h=4.55, max_w=12.3):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, 13.333, 0.12, NAVY)
    text(s, 0.5, 0.28, 12.3, 1.0, title, size=26, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, 0.52, 1.37, 0.9, 0.06, TEAL)
    w, h = Image.open(FIG / f"{fig}.png").size
    scale = min(max_w / w, max_h / h)
    pw, ph = w * scale, h * scale
    top = 1.6 + (max_h - ph) / 2
    s.shapes.add_picture(str(FIG / f"{fig}.png"), Inches((13.333 - pw) / 2), Inches(top), Inches(pw), Inches(ph))
    rect(s, 0.5, 6.3, 12.33, 0.62, LIGHT)
    rect(s, 0.5, 6.3, 0.09, 0.62, TEAL)
    text(s, 0.75, 6.3, 11.95, 0.62, takeaway, size=17, anchor=MSO_ANCHOR.MIDDLE)
    if source:
        text(s, 0.5, 7.05, 11.0, 0.3, source, size=10, color=GREY)
    text(s, 12.0, 7.05, 0.83, 0.3, str(n), size=11, color=GREY, align=PP_ALIGN.RIGHT)
    notes(s, note)
    return s


def bold_lead(lead, rest):
    return [[(lead, True), (rest, False)]]


# 1 ---------------------------------------------------------------------------------------------- title
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 5.2, 7.5, NAVY)
rect(s, 5.2, 0, 0.1, 7.5, TEAL)
text(s, 0.6, 0.7, 4.2, 0.4, "PROJECT PRESENTATION", size=14, color=RGBColor.from_string("9FD6D7"), bold=True)
text(s, 0.6, 5.3, 4.2, 1.4, ["[Institution name]", "[Department / programme]", "[Month Year]"], size=15, color=RGBColor.from_string("C9D4DC"))
text(s, 5.9, 1.2, 6.9, 2.5, "Medical Visual Question Answering for Chest X-rays", size=40, bold=True, anchor=MSO_ANCHOR.TOP)
text(s, 5.9, 3.35, 6.9, 0.9, "A knowledge-grounded approach with CheXagent-2-3b: literature review, baseline tests and retrieval-augmented design",
     size=18, color=GREY)
rect(s, 5.92, 4.5, 0.9, 0.06, TEAL)
text(s, 5.9, 4.75, 6.9, 1.9, [[("Supervisor:  ", True), ("[Supervisor name, designation]", False)],
                              [("Team members:  ", True), ("[Name 1, roll no.]", False)],
                              [("", False), ("[Name 2, roll no.]", False)],
                              [("", False), ("[Name 3, roll no.]", False)]], size=17, color=NAVY)
notes(s, "Title slide. Names, institution and date are placeholders until the official format is supplied.")

# 2 ---------------------------------------------------------------------------------------------- CXR
content_slide(2, "A chest X-ray is a shadow picture: dark means air, white means dense tissue", "fig_cxr_anatomy",
              bold_lead("Four findings used throughout this project: ", "heart size, fluid, air outside the lung, and collapse."),
              "Sources: radiology teaching material (cardiothoracic ratio, costophrenic angle, pleural line); schematic drawn for this report, not a real image.",
              "Plain-language explanation of a frontal chest X-ray. The patient's right is on the image's left, which matters later for the left/right questions. "
              "The four cards are the findings that appear most in our question sets.")

# 3 ---------------------------------------------------------------------------------------------- VQA
content_slide(3, "Medical VQA: show the model an X-ray and a question, then check its free-text answer", "fig_vqa_task",
              bold_lead("Two kinds of difficulty: ", "seeing the finding (perception) and knowing about the disease (knowledge)."),
              "Datasets: SLAKE and VQA-RAD, chest X-ray questions only. Model: StanfordAIMI/CheXagent-2-3b.",
              "We keep perception and knowledge separate on purpose. Most of the later results depend on that split.")

# 4 ---------------------------------------------------------------------------------------------- papers
content_slide(4, "Of six papers reviewed, three shaped the project; none combines retrieval with chest X-rays", "fig_papers_matrix",
              bold_lead("CheXagent ", "is our baseline, RULE warns about retrieval, Med-Flamingo says to rate answers by hand."),
              "Chen et al. 2024 (CheXagent); Xia et al., EMNLP 2024 (RULE); Moor et al. 2023 (Med-Flamingo); Eslami et al., EACL 2023 (PubMedCLIP); "
              "Li et al. 2023 (LLaVA-Med); Zhang et al. 2023 (PMC-VQA).",
              "Highlighted rows are the three papers that changed our plan. The others gave context: the image encoder matters most, "
              "GPT-written training data can hallucinate, and caption-made questions can be answered from text alone. "
              "The bottom row states the gap we address, among the papers we reviewed. Hollow circle = planned, not yet done.", max_h=4.4)

# 5 ---------------------------------------------------------------------------------------------- gaps
content_slide(5, "Each finding in the literature became a concrete design decision", "fig_gaps_to_plan",
              bold_lead("Research question: ", "can retrieved textbook knowledge make a chest X-ray model answer knowledge questions better?"),
              "Findings from CheXagent (77-79% on fine-grained reasoning), RULE (naive retrieval over-trust) and Med-Flamingo (clinician rating).",
              "This slide is the bridge between the literature review and the experiments.")

# 6 ---------------------------------------------------------------------------------------------- evaluation
content_slide(6, "Rules are written before any run, and negative results are reported", "fig_eval_protocol",
              bold_lead("Pre-registration: ", "test sets, scorer and success conditions were frozen first, then amended in writing when needed."),
              "Documented in docs/evaluation_protocol.md (6 amendments) and docs/phase2_rag_plan.md.",
              "Item recall is used because exact match is too strict for free text. The constant-answer floor asks: would a system that ignores the image "
              "and always writes the common answer do just as well?")

# 7 ---------------------------------------------------------------------------------------------- text-only
content_slide(7, "Our baseline tests: 58% overall, but the left/right and knowledge scores do not hold up", "fig_text_only_results",
              bold_lead("Left/right: ", "answers on a mirrored image were unchanged in 112 of 120 pairs. Knowledge: no better than a fixed answer."),
              "Runs 2-4 (219 baseline questions, 105 knowledge questions, 120 mirror-image pairs). Error bars: 95% Wilson intervals.",
              "This is the first thing we found when we tested the model ourselves. The identical answers on mirrored images were the first clue "
              "that something was wrong. The next slide explains what it was.", max_h=4.55)

# 8 ---------------------------------------------------------------------------------------------- bug
content_slide(8, "Turning point: the image never reached the model, so earlier results are a text-only control", "fig_bug_flow",
              bold_lead("What stays valid: ", "how much of these benchmarks a text-only model solves. What is withdrawn: every conclusion about perception."),
              "transformers 4.46.3 passes an empty cache; CheXagent's code only encodes the image when the cache is None. Recorded in experiment_log.md, entry 17.",
              "Found when disease naming (Phase 1d) gave one answer for all 130 images. A patch and an IMAGE CHECK were added to every notebook. "
              "All earlier documents carry a correction banner. Reruns of notebooks 04, 05, then 01-03 are scheduled.", max_h=4.55)

# 9 ---------------------------------------------------------------------------------------------- pipeline
content_slide(9, "Phase 2 design: first name the disease from the image, then retrieve what is known about it", "fig_rag_pipeline",
              bold_lead("Why: ", "SLAKE's knowledge answers are a lookup keyed by the disease, and the question never names it."),
              "Corpora: MedlinePlus (US National Library of Medicine, public domain) and medical textbooks (MedRAG), searched with our own BM25.",
              "Because retrieval is useless without the right disease, the disease-naming step is the real bottleneck. "
              "The textbooks are copyrighted, so they are downloaded at run time and never committed to the repository.")

# 10 --------------------------------------------------------------------------------------------- arms
content_slide(10, "Six arms separate what retrieval adds from what knowing the disease adds", "fig_arms",
              bold_lead("Success needs more than a score gain: ", "the system (A6) must beat both the fair baseline (A1) and the guess alone (A5)."),
              "Defined in docs/phase2_rag_plan.md and Amendment 5 of the evaluation protocol. Oracle arms are upper bounds only.",
              "Oracle arms use SLAKE's own disease label, which a real system would not have, so they are never reported as the result.", max_h=4.2)

# 11 --------------------------------------------------------------------------------------------- retrieval
content_slide(11, "Retrieval check, run without any model: the corpus holds the answer only when the disease is right", "fig_retrieval_results",
              bold_lead("Held-out: ", "29% of gold items found with the true disease, 17% with a wrong disease, 3% from the question alone."),
              "Strict coverage = gold item words inside one sentence of the retrieved text. 50 oracle-evaluable questions; setting frozen on 16 dev questions only.",
              "This is an upper limit on what retrieval can give. Prevention answers such as 'keep healthy' are generic and almost never match a source.", max_h=4.5)

# 12 --------------------------------------------------------------------------------------------- status
content_slide(12, "Where we stand: pipeline and design are ready, the image-visible model runs come next", "fig_roadmap",
              bold_lead("Next: ", "rerun notebooks 04 and 05 (then 01-03) with the image fix; then blind rating. No clinical claims are made."),
              "Limits: zero-shot only, small test sets, one model, SLAKE answers are templates, and no clinician review yet.",
              "Honest status: the retrieval ceiling is measured, but no result with the image visible exists yet. "
              "Phase 3 (fine-tuning) is optional and only considered if Phase 2 shows a gap worth closing.", max_h=3.9)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print("saved", OUT)
