"""Builds deliverables/presentation/Medical_VQA_CXR_Project_Presentation.pptx (12 slides, 16:9, minimalist).

  python3 scripts/make_figures.py && MMDC=... python3 scripts/render_diagrams.py && python3 scripts/build_deck.py

Design: white background, near-black text, one teal accent, left-aligned, generous margins, one idea per slide.
Block diagrams are Mermaid (deliverables/diagrams/*.mmd); the chest X-ray is a CC0 photograph. 
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

FIG = Path("deliverables/report/figures")
OUT = Path("deliverables/presentation/Medical_VQA_CXR_Project_Presentation.pptx")
FONT = "Calibri"
DARK, TEAL, GREY, WHITE = (RGBColor.from_string(c) for c in ("1F2933", "0F8B8D", "6B7785", "FFFFFF"))
M = 0.7                      # page margin (inches)
CW = 13.333 - 2 * M          # content width

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, paras, size=18, color=DARK, bold=False, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, space=0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = paras if isinstance(paras, list) else [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        parts = para if isinstance(para, list) else [(para, bold, color, size)]
        for part in parts:
            t, b, c, s = (list(part) + [bold, color, size][len(part) - 1:])[:4] if len(part) < 4 else part
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size, r.font.bold = FONT, Pt(s), b
            r.font.color.rgb = c
    return tb


def rule(slide, x, y, w, color=TEAL, h=0.04):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False


def picture(slide, name, x, y, max_w, max_h, align="center"):
    w, h = Image.open(FIG / f"{name}.png").size
    k = min(max_w / w, max_h / h)
    pw, ph = w * k, h * k
    px = x + (max_w - pw) / 2 if align == "center" else x
    slide.shapes.add_picture(str(FIG / f"{name}.png"), Inches(px), Inches(y), Inches(pw), Inches(ph))
    return pw, ph


def base(n, title, source=None, note=""):
    s = prs.slides.add_slide(BLANK)
    text(s, M, 0.5, CW, 0.95, title, size=28, bold=True, anchor=MSO_ANCHOR.TOP)
    if source:
        text(s, M, 7.0, 10.8, 0.3, source, size=10, color=GREY)
    text(s, 12.0, 7.0, 0.63, 0.3, str(n), size=11, color=GREY, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = note
    return s


def takeaway(s, lead, rest, y=6.12):
    text(s, M, y, CW, 0.8, [[(lead, True, TEAL, 20), (rest, False, DARK, 20)]], anchor=MSO_ANCHOR.TOP)


# 1 ---------------------------------------------------------------------------------------------- title
s = prs.slides.add_slide(BLANK)
text(s, M, 0.9, 8, 0.4, "PROJECT PRESENTATION", size=14, bold=True, color=TEAL)
text(s, M, 1.6, 10.5, 2.4, "Medical Visual Question Answering for Chest X-rays", size=46, bold=True)
text(s, M, 3.85, 9.5, 1.0, "A knowledge-grounded approach with CheXagent-2-3b: literature review, baseline tests and retrieval-augmented design",
     size=20, color=GREY)
rule(s, M, 5.0, 1.0)
text(s, M, 5.25, 11, 1.8, [[("Supervisor   ", True, DARK, 18), ("Dr. Snober Mushtaq", False, DARK, 18)],
                           [("Students   ", True, DARK, 18), ("Ankrit (Enrollment 2023BCSE028)     Hirdyansh (Enrollment 2023BCSE036)", False, DARK, 18)],
                           [("Department of Computer Science and Engineering", False, GREY, 16)]], space=8)
s.notes_slide.notes_text_frame.text = "Title slide. The institution name and date can be added once the official format is supplied."

# 2 ---------------------------------------------------------------------------------------------- CXR
s = base(2, "A chest X-ray is a shadow picture: dark means air, white means dense tissue",
         "Photograph: M. Haggstrom, Wikimedia Commons, CC0 (normal PA chest X-ray). Findings: standard radiology teaching material.",
         "Real, normal frontal (PA) chest radiograph. Dx and Sin are Italian for right and left: the patient's right appears on the image's left. "
         "The four findings are the ones that appear most in the question sets.")
pw, ph = picture(s, "fig_cxr_real", M, 1.6, 7.3, 5.1, align="left")
finds = [("Cardiomegaly", "Heart wider than half the chest"), ("Pleural effusion", "Fluid blunts the corner at the lung base"),
         ("Pneumothorax", "Dark area with no lung markings"), ("Atelectasis", "Collapsed lung looks whiter")]
text(s, 8.5, 1.75, 4.2, 0.4, "Four findings used in this project", size=14, color=GREY)
for i, (a, b) in enumerate(finds):
    text(s, 8.5, 2.3 + i * 1.12, 4.2, 1.0, [[(a, True, TEAL, 22)], [(b, False, DARK, 17)]], space=2)

# 3 ---------------------------------------------------------------------------------------------- VQA
s = base(3, "Medical VQA: show the model an X-ray and a question, then check its free-text answer",
         "Datasets: SLAKE and VQA-RAD, chest X-ray questions only. Model: StanfordAIMI/CheXagent-2-3b.",
         "Two kinds of difficulty are kept separate on purpose; most later results depend on that split.")
picture(s, "fig_vqa_flow", M, 1.7, CW, 2.7)
text(s, M, 4.75, 5.6, 1.9, [[("Perception", True, TEAL, 22)], [("Can the model see the finding?", False, DARK, 18)],
                            [("Is there an effusion? Which side? Is the heart enlarged?", False, GREY, 16)]], space=4)
text(s, 7.1, 4.75, 5.5, 1.9, [[("Knowledge", True, TEAL, 22)], [("Does the model know the disease?", False, DARK, 18)],
                              [("What causes it? What are the symptoms? How is it treated?", False, GREY, 16)]], space=4)

# 4 ---------------------------------------------------------------------------------------------- papers
s = base(4, "Of six papers reviewed, three shaped the project; none combines retrieval with chest X-rays",
         "Chen et al. 2024 (CheXagent); Xia et al. 2024 (RULE); Moor et al. 2023 (Med-Flamingo); Eslami et al. 2023 (PubMedCLIP); "
         "Li et al. 2023 (LLaVA-Med); Zhang et al. 2023 (PMC-VQA).",
         "Bold rows are the three papers that changed our plan. The others gave context. The bottom row states the gap among the papers we reviewed; "
         "the hollow circle means the blind rating is planned, not yet done.")
picture(s, "fig_papers_matrix", M, 1.65, CW, 4.45)
takeaway(s, "CheXagent ", "is our baseline, RULE warns about retrieval, Med-Flamingo says to rate answers by hand.")

# 5 ---------------------------------------------------------------------------------------------- gaps
s = base(5, "Each finding in the literature became a concrete design decision",
         "CheXagent: 77-79% on fine-grained reasoning. RULE: naive retrieval over-trust. PMC-VQA: many questions answerable from text.",
         "This slide bridges the literature review and the experiments.")
text(s, M, 1.55, 5.5, 0.3, "What the papers show", size=14, color=GREY)
text(s, 6.9, 1.55, 5.5, 0.3, "What we decided to do", size=14, color=GREY)
picture(s, "fig_gaps_to_plan", M, 1.9, CW, 4.0)
takeaway(s, "Research question: ", "can retrieved textbook knowledge make a chest X-ray model answer knowledge questions better?")

# 6 ---------------------------------------------------------------------------------------------- evaluation
s = base(6, "Rules are written before any run, and negative results are reported",
         "docs/evaluation_protocol.md and docs/phase2_rag_plan.md. Scorer: item recall v2; floor: most common gold answer.",
         "Item recall is used because exact match is too strict for free text. The constant-answer floor asks whether a system that ignores the image "
         "and always writes the common answer would do just as well.")
picture(s, "fig_eval_flow", M, 1.75, 6.5, 3.6, align="left")
tiles = [("219", "baseline questions"), ("130", "knowledge questions"), ("140", "left / right questions"),
         ("120", "mirror-image pairs"), ("47", "heart-size questions"), ("130", "images with a disease label")]
for i, (n, t) in enumerate(tiles):
    x = 8.0 + (i % 2) * 2.5
    y = 1.8 + (i // 2) * 1.3
    text(s, x, y, 2.3, 0.7, n, size=36, bold=True, color=TEAL)
    text(s, x, y + 0.68, 2.3, 0.4, t, size=14, color=GREY)
takeaway(s, "Pre-registration: ", "test sets, scorer and success conditions were frozen first, then amended in writing when needed.")

# 7 ---------------------------------------------------------------------------------------------- perception
s = base(7, "Reading the X-ray: the model's left/right and heart-size answers follow the image",
         "Colab T4, 13 Nov 2026, notebook 03, 788 generations. Data: results/run_6_with_image/. Plain wording; 120 mirror pairs, 60 side questions, 47 heart questions.",
         "Mirror test: each question asked on the original X-ray and on its mirror image; a model that reads laterality should change its answer. "
         "Heart size: 17 of 22 enlarged hearts found and 25 of 25 normal hearts called normal. CheXagent's training data includes SLAKE and VQA-RAD, "
         "so these are not clean held-out tests; the heart sensitivity rests on 22 positive questions.")
picture(s, "fig_perception_slide", M, 1.65, CW, 4.1)
takeaway(s, "Mirroring the X-ray flips the answer in 78% of pairs (95% CI 69-85%). ",
         "Left/right 90% correct; enlarged hearts found in 17 of 22, normal hearts called normal 25 of 25.")

# 8 ------------------------------------------------------------------------------------------ pipeline
s = base(8, "Phase 2 design: first name the disease from the image, then retrieve what is known about it",
         "Corpora: MedlinePlus (US National Library of Medicine, public domain) and medical textbooks (MedRAG); our own BM25 search.",
         "SLAKE's knowledge answers are a lookup keyed by the disease, and the question never names it. The textbooks are copyrighted, so they are downloaded "
         "at run time and never committed to the repository.")
picture(s, "fig_rag_pipeline", M, 1.6, CW, 2.6)
text(s, M, 4.45, 8, 0.35, "Why two steps: the answer depends only on the disease", size=18, bold=True, color=TEAL)
picture(s, "fig_lookup_example", M, 4.9, 8.0, 1.6, align="left")

# 9 ------------------------------------------------------------------------------------------ arms
s = base(9, "Six arms separate what retrieval adds from what knowing the disease adds",
         "docs/phase2_rag_plan.md and Amendment 5 of the evaluation protocol.",
         "Oracle arms use SLAKE's own disease label, which a real system would not have, so they are never reported as the result.")
picture(s, "fig_arms", M, 1.65, CW, 4.1)
takeaway(s, "Success needs more than a score gain: ", "the system (A6) must beat both the fair baseline (A1) and the guess alone (A5).")

# 10 ------------------------------------------------------------------------------------------ retrieval
s = base(10, "Retrieval check, run without any model: the corpus holds the answer mainly when the disease is right",
         "Strict coverage: gold item words inside one sentence of the retrieved text. 50 oracle-evaluable questions; setting frozen on 16 dev questions.",
         "This is an upper limit on what retrieval can give. Prevention answers such as 'keep healthy' are generic and almost never match a source.")
picture(s, "fig_retrieval_results", M, 1.65, CW, 4.1)
takeaway(s, "Held-out: ", "29% of gold items found with the true disease, 17% with a wrong disease, 3% from the question alone.")

# 11 ------------------------------------------------------------------------------------------ references
s = base(11, "References", None, "Full reference list; the same entries are in the project report.")
refs = [
    "Chen Z., Varma M., Delbrouck J.-B., et al. CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation. arXiv:2401.12208, 2024.",
    "Xia P., Zhu K., Li H., et al. RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models. EMNLP, 2024.",
    "Moor M., Huang Q., Wu S., et al. Med-Flamingo: a Multimodal Medical Few-shot Learner. ML4H, PMLR, 2023.",
    "Eslami S., Meinel C., de Melo G. PubMedCLIP: How Much Does CLIP Benefit VQA in the Medical Domain? Findings of EACL, 2023.",
    "Li C., Wong C., Zhang S., et al. LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. NeurIPS D&B, 2023.",
    "Zhang X., Wu C., Zhao Z., et al. PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering. arXiv:2305.10415, 2023.",
    "Liu B., Zhan L.-M., Xu L., et al. SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical VQA. IEEE ISBI, 2021.",
    "Lau J. J., Gayen S., Ben Abacha A., Demner-Fushman D. A dataset of clinically generated visual questions and answers about radiology images. Scientific Data, 2018.",
    "Xiong G., Jin Q., Lu Z., Zhang A. Benchmarking Retrieval-Augmented Generation for Medicine (MedRAG). Findings of ACL, 2024.",
    "Robertson S., Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR, 2009.",
    "Haggstrom M. Normal posteroanterior (PA) chest radiograph (X-ray). Wikimedia Commons, CC0.",
]
text(s, M, 1.45, CW, 5.4, [[(f"[{i + 1}]  ", True, TEAL, 14), (r, False, DARK, 14)] for i, r in enumerate(refs)], space=7)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT)
