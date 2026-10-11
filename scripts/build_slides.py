"""
Builds the 5-slide literature-review presentation for the
Medical VQA for Chest X-ray project.

Run: python3 scripts/build_slides.py
Output: slides/Medical_VQA_CXR_Literature_Review.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import copy

# ---- palette -----------------------------------------------------------
NAVY = RGBColor(0x0F, 0x2A, 0x43)
TEAL = RGBColor(0x14, 0x7D, 0x7A)
TEAL_LIGHT = RGBColor(0xE6, 0xF3, 0xF2)
ACCENT = RGBColor(0xE8, 0x7A, 0x41)
GREY = RGBColor(0x4A, 0x55, 0x61)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF6, 0xF8, 0xF9)
ROW_ALT = RGBColor(0xEE, 0xF4, 0xF4)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_background(slide, color=WHITE):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    # send to back
    sp = bg._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)
    return bg


def add_top_bar(slide, title, kicker=None, bar_color=NAVY, text_color=WHITE):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = bar_color
    bar.line.fill.background()
    bar.shadow.inherit = False

    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.15), SLIDE_W, Pt(4))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()
    accent.shadow.inherit = False

    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.14), Inches(12.3), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    if kicker:
        p0 = tf.paragraphs[0]
        p0.text = kicker
        p0.font.size = Pt(13)
        p0.font.color.rgb = RGBColor(0xBF, 0xD6, 0xDB)
        p0.font.bold = True
        p0.font.name = "Calibri"
        p1 = tf.add_paragraph()
    else:
        p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = text_color
    p1.font.name = "Calibri"
    return tb


def add_footer(slide, text, page_num):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.12), Inches(10), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(9)
    p.font.color.rgb = GREY
    p.font.name = "Calibri"

    tb2 = slide.shapes.add_textbox(Inches(12.6), Inches(7.12), Inches(0.6), Inches(0.3))
    tf2 = tb2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = str(page_num)
    p2.font.size = Pt(9)
    p2.font.color.rgb = GREY
    p2.alignment = PP_ALIGN.RIGHT


def set_cell(cell, text, size=11, bold=False, color=NAVY, bg=None, align=PP_ALIGN.LEFT,
             font="Calibri"):
    cell.margin_left = Pt(6)
    cell.margin_right = Pt(6)
    cell.margin_top = Pt(4)
    cell.margin_bottom = Pt(4)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    tf.clear()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font
        p.alignment = align
    if bg is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE


def add_bullets(slide, left, top, width, height, items, size=14, color=NAVY,
                 bullet_color=TEAL, line_spacing=1.08, space_after=8, bold_lead=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = "\u25B8  " + lead
            r1.font.bold = True
            r1.font.size = Pt(size)
            r1.font.color.rgb = bullet_color
            r1.font.name = "Calibri"
            if rest:
                r2 = p.add_run()
                r2.text = "  " + rest
                r2.font.size = Pt(size)
                r2.font.color.rgb = color
                r2.font.name = "Calibri"
        else:
            r1 = p.add_run()
            r1.text = "\u25B8  " + item
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
            r1.font.name = "Calibri"
    return tb


def add_pill(slide, left, top, width, height, text, fill=TEAL, text_color=WHITE, size=12):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shp.adjustments[0] = 0.5
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = text_color
    p.alignment = PP_ALIGN.CENTER
    return shp


# =========================================================================
# SLIDE 1 — Title
# =========================================================================
s1 = prs.slides.add_slide(BLANK)
add_background(s1, NAVY)

# decorative stripe
stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.65), SLIDE_W, Pt(5))
stripe.fill.solid(); stripe.fill.fore_color.rgb = ACCENT; stripe.line.fill.background()
stripe.shadow.inherit = False

kicker = s1.shapes.add_textbox(Inches(0.9), Inches(1.5), Inches(11.5), Inches(0.5))
p = kicker.text_frame.paragraphs[0]
p.text = "LITERATURE REVIEW  \u2022  WEEK 1 DELIVERABLE"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(0x8F, 0xC9, 0xC6)
p.font.name = "Calibri"

title = s1.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.9))
tf = title.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Medical Visual Question Answering"
p.font.size = Pt(40)
p.font.bold = True
p.font.color.rgb = WHITE
p.font.name = "Calibri"
p2 = tf.add_paragraph()
p2.text = "for Chest X-ray Analysis"
p2.font.size = Pt(40)
p2.font.bold = True
p2.font.color.rgb = ACCENT
p2.font.name = "Calibri"

sub = s1.shapes.add_textbox(Inches(0.9), Inches(3.95), Inches(11.0), Inches(0.9))
p = sub.text_frame.paragraphs[0]
p.text = "6-paper review \u2014 baseline VLM landscape, dataset survey, and the gaps we can realistically close"
p.font.size = Pt(18)
p.font.color.rgb = RGBColor(0xD6, 0xE4, 0xE8)
p.font.name = "Calibri"
sub.text_frame.word_wrap = True

foot = s1.shapes.add_textbox(Inches(0.9), Inches(6.0), Inches(11.0), Inches(1.0))
tf = foot.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Papers reviewed: PubMedCLIP (EACL'23) \u00b7 LLaVA-Med \u00b7 Med-Flamingo  (team)   +   CheXagent \u00b7 PMC-VQA/MedVInT \u00b7 RULE  (new)"
p.font.size = Pt(13)
p.font.color.rgb = RGBColor(0xA9, 0xBE, 0xC6)
p.font.italic = True

add_footer_title = s1.shapes.add_textbox(Inches(0.9), Inches(6.9), Inches(8), Inches(0.3))
p = add_footer_title.text_frame.paragraphs[0]
p.text = "Slide 1 / 5"
p.font.size = Pt(10)
p.font.color.rgb = RGBColor(0x8F, 0x9F, 0xA6)

# =========================================================================
# SLIDE 2 — Problem framing + the 6-paper landscape
# =========================================================================
s2 = prs.slides.add_slide(BLANK)
add_background(s2, WHITE)
add_top_bar(s2, "The Landscape: How Medical VQA Has Evolved", kicker="WHY THIS MATTERS")

add_bullets(s2, Inches(0.5), Inches(1.4), Inches(12.3), Inches(1.0), [
    ("Goal:", "a system that looks at a chest X-ray + a free-text question \u2192 gives a grounded, natural-language answer (not just a disease label)."),
    ("Three waves in the literature:", "discriminative classification \u2192 generative instruction-tuned chatbots \u2192 CXR-specialized + retrieval-augmented models."),
], size=13.5, space_after=6)

rows = [
    ("Paper", "Approach", "Formulation", "CXR-specific?"),
    ("PubMedCLIP (EACL'23)", "Domain-tuned CLIP visual encoder plugged into classifier", "Discriminative", "No"),
    ("LLaVA-Med", "GPT-4 self-instruct data + curriculum tuning of LLaVA", "Generative chatbot", "No (1 of 5 modalities)"),
    ("Med-Flamingo", "Few-shot in-context learning on textbook corpus", "Generative, few-shot", "No"),
    ("CheXagent  \u2605 new", "8.5M-sample CXR-only instruction dataset + 3.1B model", "Generative + discriminative", "Yes \u2014 fully"),
    ("PMC-VQA / MedVInT  \u2605 new", "227K ChatGPT-generated QA pairs, pretrain\u2192fine-tune", "Generative", "No (mixed modality)"),
    ("RULE  \u2605 new", "Retrieval-augmented generation + factuality-risk fixes", "Generative + RAG", "Partial (MIMIC-CXR, IU X-ray)"),
]
n_rows, n_cols = len(rows), len(rows[0])
tbl_left, tbl_top, tbl_w, tbl_h = Inches(0.5), Inches(2.55), Inches(12.3), Inches(4.3)
gframe = s2.shapes.add_table(n_rows, n_cols, tbl_left, tbl_top, tbl_w, tbl_h)
table = gframe.table
col_widths = [Inches(2.9), Inches(4.2), Inches(2.6), Inches(2.6)]
for i, w in enumerate(col_widths):
    table.columns[i].width = w
for r in range(n_rows):
    table.rows[r].height = Inches(4.3/n_rows)
    for c in range(n_cols):
        text = rows[r][c]
        if r == 0:
            set_cell(table.cell(r, c), text, size=12.5, bold=True, color=WHITE, bg=NAVY, align=PP_ALIGN.LEFT)
        else:
            new_flag = "\u2605 new" in text
            bg = TEAL_LIGHT if new_flag else (ROW_ALT if r % 2 == 0 else WHITE)
            bold = new_flag and c == 0
            set_cell(table.cell(r, c), text, size=11.5, bold=bold, color=NAVY, bg=bg)

add_footer(s2, "Medical VQA for Chest X-ray Analysis \u2014 Literature Review", 2)

# =========================================================================
# SLIDE 3 — Deep dive: the 3 new papers (key numbers / insights)
# =========================================================================
s3 = prs.slides.add_slide(BLANK)
add_background(s3, WHITE)
add_top_bar(s3, "Deep Dive: The 3 New Papers We Added", kicker="DEEP DIVE")

card_w, card_h = Inches(3.95), Inches(5.2)
gap = Inches(0.2)
left0 = Inches(0.5)
top0 = Inches(1.45)

cards = [
    ("CheXagent", ACCENT, "arXiv 2401.12208 \u2014 Stanford AIMI, 2024",
     [
        ("What:", "First foundation model trained end-to-end only on chest X-rays."),
        ("Data:", "CheXinstruct \u2014 8.5M instruction triplets from 32 public CXR datasets, 35 tasks."),
        ("Result:", "3.1B params beats GPT-4V, LLaVA-Med, RadFM on 8/8 CheXbench tasks; 36% report-drafting time saved (8-radiologist study)."),
        ("Weak spot:", "Fine-grained laterality/severity reasoning only ~77-79% acc. \u2014 vs. ~90%+ on coarse tasks. Exactly our example use-case."),
     ]),
    ("PMC-VQA / MedVInT", TEAL, "arXiv 2305.10415 \u2014 2023",
     [
        ("What:", "Standard generative MedVQA baseline most later papers (incl. RULE) compare against."),
        ("Data:", "227K QA pairs auto-generated by ChatGPT from 149K PMC figure captions; 2-stage filter to require image grounding."),
        ("Result:", "New SOTA on VQA-RAD/SLAKE (81.6% / 88.0%); but only ~35-40% on their own harder PMC-VQA-test."),
        ("Weak spot:", "Caption-derived QA risks text-shortcut answers; ~82%-accurate filter is imperfect; not CXR-exclusive."),
     ]),
    ("RULE", NAVY, "EMNLP 2024 \u2014 arXiv 2407.05131",
     [
        ("What:", "The only paper to actually build + fix RAG for medical VLMs."),
        ("Key finding:", "47-59% of all RAG-caused errors happen because the model over-trusts retrieved text even when its own answer was already right."),
        ("Fix:", "Calibrated top-k context selection + preference fine-tuning (KBPT) using those failure cases."),
        ("Result:", "+47.4% avg factual accuracy vs. best baseline on MIMIC-CXR / IU X-ray VQA."),
        ("Gap:", "Retrieves similar patient reports, not authoritative knowledge/guidelines \u2014 open door for us."),
     ]),
]

for i, (name, color, src, bullets) in enumerate(cards):
    left = left0 + i * (card_w + gap)
    card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top0, card_w, card_h)
    card.adjustments[0] = 0.035
    card.fill.solid(); card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = color
    card.line.width = Pt(1.5)
    card.shadow.inherit = False

    head = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top0, card_w, Inches(0.55))
    head.adjustments[0] = 0.12
    head.fill.solid(); head.fill.fore_color.rgb = color
    head.line.fill.background()
    head.shadow.inherit = False
    tf = head.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = name
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    src_tb = s3.shapes.add_textbox(left + Inches(0.12), top0 + Inches(0.58), card_w - Inches(0.24), Inches(0.3))
    p = src_tb.text_frame.paragraphs[0]
    p.text = src
    p.font.size = Pt(9.5)
    p.font.italic = True
    p.font.color.rgb = GREY

    body = s3.shapes.add_textbox(left + Inches(0.15), top0 + Inches(0.95), card_w - Inches(0.3), card_h - Inches(1.05))
    tf = body.text_frame
    tf.word_wrap = True
    for j, (lead, rest) in enumerate(bullets):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        p.line_spacing = 1.02
        r1 = p.add_run()
        r1.text = lead + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = color
        r2 = p.add_run()
        r2.text = rest
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = NAVY

add_footer(s3, "Medical VQA for Chest X-ray Analysis \u2014 Literature Review", 3)

# =========================================================================
# SLIDE 4 — Dataset landscape
# =========================================================================
s4 = prs.slides.add_slide(BLANK)
add_background(s4, WHITE)
add_top_bar(s4, "Dataset Landscape: What's Available & When to Use It", kicker="DATASETS")

rows4 = [
    ("Dataset", "Scope", "Size", "Best use for us"),
    ("VQA-RAD", "Radiology (head/chest/abdomen mix)", "315 img / 3.5k QA", "Quick baseline check (filter to chest only \u2014 note: known train/test leakage)"),
    ("SLAKE (EN)", "Radiology, multi-organ + knowledge graph", "642 img / ~7k QA", "Secondary benchmark, chest subset only"),
    ("PathVQA", "Pathology (not radiology)", "5k img / 32.8k QA", "Out of scope \u2014 cross-domain sanity check only"),
    ("PMC-VQA", "Mixed biomedical modalities", "149k img / 227k QA", "Pretraining-style resource; not CXR-exclusive"),
    ("CheXinstruct", "Chest X-ray only \u2014 rolls up 32 CXR datasets", "8.5M triplets / 35 tasks", "\u2605 Most relevant training/eval resource available"),
    ("MIMIC-CXR(-VQA)", "Chest X-ray only, largest public CXR+report set", "377k img", "Best large-scale CXR source (needs PhysioNet credential)"),
    ("Open-I / IU X-ray", "Chest X-ray only, de-identified", "7.4k img / 3.9k reports", "\u2605 Easiest access (no credentialing) \u2014 use to build our own eval set"),
    ("Rad-Restruct", "Chest X-ray, structured template QA", "\u2014", "Harder held-out CXR benchmark (used by CheXagent)"),
]
n_rows4 = len(rows4)
tbl_top4 = Inches(1.45)
tbl_h4 = Inches(5.45)
gframe4 = s4.shapes.add_table(n_rows4, 4, Inches(0.5), tbl_top4, Inches(12.3), tbl_h4)
table4 = gframe4.table
col_widths4 = [Inches(2.1), Inches(3.5), Inches(2.1), Inches(4.6)]
for i, w in enumerate(col_widths4):
    table4.columns[i].width = w
for r in range(n_rows4):
    table4.rows[r].height = Inches(5.45/n_rows4)
    for c in range(4):
        text = rows4[r][c]
        if r == 0:
            set_cell(table4.cell(r, c), text, size=12, bold=True, color=WHITE, bg=NAVY)
        else:
            starred = "\u2605" in text
            bg = TEAL_LIGHT if starred else (ROW_ALT if r % 2 == 0 else WHITE)
            set_cell(table4.cell(r, c), text, size=10.5, color=NAVY, bg=bg, bold=starred and c == 0)

add_footer(s4, "Medical VQA for Chest X-ray Analysis \u2014 Literature Review", 4)

# =========================================================================
# SLIDE 5 — Gaps & proposed direction
# =========================================================================
s5 = prs.slides.add_slide(BLANK)
add_background(s5, WHITE)
add_top_bar(s5, "Gaps We Found & Our Proposed Direction", kicker="SO WHAT CAN WE ADD?")

add_bullets(s5, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.9), [
    ("No knowledge-grounded RAG for CXR VQA.", "RULE retrieves similar patients' reports; nobody retrieves finding definitions/guidelines from authoritative references."),
    ("Naive RAG is risky, not free.", "RULE shows 47-59% of RAG errors come from over-trusting retrieved text \u2014 any RAG we add must budget for this."),
    ("Fine-grained laterality/severity reasoning is the field's weak point.", "Even CheXagent, the best CXR-only model, scores only ~77-79% here \u2014 directly matches our \"which side\" example question."),
    ("Automated metrics mislead.", "Med-Flamingo shows BERT-sim/exact-match disagree with real clinical usefulness \u2014 we need a relevance / unsupported-claims rubric, not just accuracy."),
    ("Caption-derived QA data risks text-shortcut answers.", "LLaVA-Med & PMC-VQA's auto-generated QA can be answered without truly looking at the image."),
    ("No uncertainty / abstention anywhere.", "None of the 6 papers let the model say \"I'm not sure\" \u2014 open direction matching our brief."),
], size=12, space_after=9)

# right panel: recommendation box
rec_left, rec_top, rec_w, rec_h = Inches(6.75), Inches(1.4), Inches(6.05), Inches(4.9)
rec_box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rec_left, rec_top, rec_w, rec_h)
rec_box.adjustments[0] = 0.04
rec_box.fill.solid(); rec_box.fill.fore_color.rgb = NAVY
rec_box.line.fill.background()
rec_box.shadow.inherit = False

rec_title = s5.shapes.add_textbox(rec_left + Inches(0.3), rec_top + Inches(0.2), rec_w - Inches(0.6), Inches(0.5))
p = rec_title.text_frame.paragraphs[0]
p.text = "Proposed Project Direction"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT

rec_body = s5.shapes.add_textbox(rec_left + Inches(0.3), rec_top + Inches(0.8), rec_w - Inches(0.6), rec_h - Inches(1.0))
tf = rec_body.text_frame
tf.word_wrap = True
items = [
    ("1. Baseline:", "Use CheXagent (open-source, 3.1B, CXR-specific) instead of a general medical VLM \u2014 cheaper to run, already specialized to our exact scope."),
    ("2. Improvement:", "Knowledge-grounded RAG \u2014 retrieve short authoritative passages on CXR findings (not case reports), with RULE-style calibrated top-k and over-reliance safeguards."),
    ("3. Stretch goal:", "Lightweight uncertainty/abstention signal + optional targeted fine-tune on laterality-labeled data to attack the side/severity weak spot."),
    ("4. Evaluation:", "Chest-filtered VQA-RAD/SLAKE + a small hand-built Open-I-based set, scored on accuracy AND a relevance / unsupported-claims rubric."),
]
for j, (lead, rest) in enumerate(items):
    p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
    p.space_after = Pt(12)
    p.line_spacing = 1.05
    r1 = p.add_run()
    r1.text = lead + " "
    r1.font.bold = True
    r1.font.size = Pt(12.5)
    r1.font.color.rgb = RGBColor(0x8F, 0xC9, 0xC6)
    r2 = p.add_run()
    r2.text = rest
    r2.font.size = Pt(12.5)
    r2.font.color.rgb = WHITE

add_footer(s5, "Medical VQA for Chest X-ray Analysis \u2014 Literature Review", 5)

prs.save("/workspace/slides/Medical_VQA_CXR_Literature_Review.pptx")
print("Saved presentation.")
