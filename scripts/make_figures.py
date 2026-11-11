"""Figures for the project report and slides (deliverables/).

  python3 scripts/make_figures.py

Writes deliverables/report/figures/*.pdf (vector, for LaTeX) and *.png (for the slides). Every number is read from the
repository's own result files (results/, docs numbers are not retyped) except where a comment says otherwise.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.text
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.evaluate import wilson_interval  # noqa: E402
from baseline.knowledge import constant_floor, item_scores  # noqa: E402

OUT = Path("deliverables/report/figures")
OUT.mkdir(parents=True, exist_ok=True)

NAVY, TEAL, ORANGE, GREY, LIGHT = "#12263A", "#0F8B8D", "#E4572E", "#6B7785", "#EEF3F6"
GREEN, AMBER, WHITE, MID = "#2E8B57", "#E9A23B", "#FFFFFF", "#C9D4DC"

for _f in font_manager.findSystemFonts():
    if "arlito" in _f:
        font_manager.fontManager.addfont(_f)
for fam in ("Carlito", "Lato", "DejaVu Sans"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, text="", fc=LIGHT, ec=NAVY, tc=NAVY, fs=13, bold=False, lw=1.6, r=0.12,
        ha="center", pad=0.15, alpha=1.0, va="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, alpha=alpha))
    if text:
        tx = x + w / 2 if ha == "center" else x + pad
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
                linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2, style="-|>", rad=0.0, ms=16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


SCALE = 1.2


def save(fig, name):
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontsize(t.get_fontsize() * SCALE)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------ 1
def fig_cxr_anatomy():
    fig, ax = canvas(12.4, 5.6)
    # film
    ax.add_patch(Rectangle((0.3, 0.3), 4.6, 5.0, fc="#0B1724", ec=NAVY, lw=2))
    cx = 2.6
    ax.add_patch(Ellipse((cx, 2.55), 3.6, 4.4, fc="#1C2D3F", ec="none"))              # chest wall
    ax.add_patch(Polygon([[1.0, 4.3], [2.35, 4.9], [2.35, 1.55], [0.95, 1.65], [0.65, 3.0]], fc="#0E1A28", ec="#53697D", lw=1.2))  # lung (image left = patient's right)
    ax.add_patch(Polygon([[4.2, 4.3], [2.85, 4.9], [2.85, 1.45], [4.25, 1.65], [4.55, 3.0]], fc="#0E1A28", ec="#53697D", lw=1.2))   # lung (image right)
    for k in range(6):                                                                  # ribs
        y = 4.55 - k * 0.52
        ax.add_patch(Arc((cx, y - 0.9), 3.55 - k * 0.03, 1.9, theta1=18, theta2=162, color="#3F566B", lw=1.3))
    ax.add_patch(Rectangle((2.52, 1.6), 0.16, 3.3, fc="#3C5A74", ec="none"))           # spine/trachea band
    ax.add_patch(Ellipse((2.95, 2.3), 1.7, 1.95, angle=-28, fc="#7C93A8", ec="#B8C7D3", lw=1.3, alpha=0.95))   # heart
    ax.add_patch(Arc((1.55, 1.55), 1.9, 1.0, theta1=0, theta2=180, color="#B8C7D3", lw=1.6))      # diaphragm domes
    ax.add_patch(Arc((3.65, 1.55), 1.9, 1.0, theta1=0, theta2=180, color="#B8C7D3", lw=1.6))
    ax.plot([1.1, 4.0], [4.55, 4.55], color="#6A8399", lw=2.2, solid_capstyle="round")  # clavicles
    ax.text(0.55, 4.85, "R", color=ORANGE, fontsize=22, fontweight="bold")
    ax.text(4.45, 4.85, "L", color=ORANGE, fontsize=22, fontweight="bold")
    ax.text(2.6, 0.12, "patient's right appears on the image's left", color=GREY, ha="center", fontsize=11.5, style="italic")
    labels = [("Trachea", 2.6, 4.45, 5.7, 5.0), ("Lung field (dark = air)", 3.95, 3.5, 5.7, 4.2), ("Heart", 3.0, 2.3, 5.7, 3.4),
              ("Diaphragm", 3.8, 1.78, 5.7, 2.6), ("Ribs", 4.35, 3.1, 5.7, 1.8)]
    for t, x1, y1, x2, y2 in labels:
        ax.plot([x1, x2 - 0.1], [y1, y2], color=GREY, lw=1, zorder=1)
        ax.text(x2, y2, t, va="center", fontsize=12.5, color=NAVY)
    # four findings
    cards = [("Cardiomegaly", "heart wider than half the chest (cardiothoracic ratio > 0.5)", ORANGE),
             ("Pleural effusion", "fluid blunts the sharp corner at the lung base", TEAL),
             ("Pneumothorax", "air outside the lung: dark area with no lung markings", NAVY),
             ("Atelectasis", "collapsed lung looks whiter; structures pull toward it", AMBER)]
    for i, (t, d, c) in enumerate(cards):
        y = 4.05 - (i // 1) * 1.28
        box(ax, 8.15, y, 4.2, 1.1, fc=WHITE, ec=MID, lw=1.2)
        ax.add_patch(Rectangle((8.15, y), 0.12, 1.1, fc=c, ec="none"))
        ax.text(8.45, y + 0.78, t, fontsize=14, fontweight="bold", color=NAVY, va="center")
        ax.text(8.45, y + 0.36, d, fontsize=11.3, color=GREY, va="center", wrap=True)
    ax.text(8.15, 5.35, "What a radiologist looks for", fontsize=14.5, color=NAVY, fontweight="bold")
    save(fig, "fig_cxr_anatomy")


# ------------------------------------------------------------------------------------------ 2
def fig_vqa_task():
    fig, ax = canvas(12.4, 5.2)
    box(ax, 0.3, 3.1, 2.2, 1.7, fc="#0B1724", ec=NAVY)
    ax.add_patch(Ellipse((1.4, 3.95), 1.5, 1.2, fc="#1C2D3F", ec="#53697D"))
    ax.add_patch(Ellipse((1.2, 3.9), 0.62, 0.8, fc="#7C93A8", ec="none"))
    ax.text(1.4, 2.82, "chest X-ray", ha="center", fontsize=12.5, color=NAVY)
    box(ax, 0.3, 1.9, 2.2, 0.62, "Is the heart enlarged?", fc=WHITE, ec=TEAL, fs=12.5, tc=NAVY)
    arrow(ax, 2.7, 3.2, 3.9, 3.2)
    box(ax, 3.95, 2.45, 2.7, 1.5, "Vision-language\nmodel\n(CheXagent-2-3b)", fc=NAVY, ec=NAVY, tc=WHITE, fs=14, bold=True)
    arrow(ax, 6.85, 3.2, 8.1, 3.2)
    box(ax, 8.15, 2.7, 3.9, 1.0, "free-text answer", fc=WHITE, ec=TEAL, fs=14)
    ax.text(10.1, 2.3, "scored against the dataset's gold answer", ha="center", fontsize=11.5, color=GREY)
    # question types
    ax.text(0.3, 1.25, "Four kinds of question, two kinds of difficulty", fontsize=14, fontweight="bold", color=NAVY)
    chips = [("Yes / no", "Is there a pleural effusion?", TEAL, "perception"),
             ("Which side", "Which side is the finding on?", TEAL, "perception"),
             ("How severe", "Is the heart enlarged?", TEAL, "perception"),
             ("Knowledge", "What causes this disease?", ORANGE, "medical knowledge")]
    for i, (t, q, c, k) in enumerate(chips):
        x = 0.3 + i * 3.05
        box(ax, x, 0.1, 2.9, 0.95, fc=WHITE, ec=c, lw=2)
        ax.text(x + 0.15, 0.78, t, fontsize=13, fontweight="bold", color=c, va="center")
        ax.text(x + 0.15, 0.4, q, fontsize=11.3, color=NAVY, va="center")
        ax.text(x + 2.78, 0.78, k, fontsize=10, color=GREY, va="center", ha="right", style="italic")
    save(fig, "fig_vqa_task")


# ------------------------------------------------------------------------------------------ 3
def fig_papers_matrix():
    fig, ax = canvas(12.4, 4.9)
    cols = ["Chest X-ray focus", "Writes free text", "Uses retrieval", "Clinician /\nblind rating", "Textbook\nsource"]
    rows = [
        ("CheXagent (2024)", [1, 1, 0, 1, 0], "our baseline model; weakest on left/right and severity (77-79%)"),
        ("RULE (2024)", [0, 1, 1, 0, 0], "naive retrieval hurts: 47-59% of its errors follow the text"),
        ("Med-Flamingo (2023)", [0, 1, 0, 1, 0], "automated scores disagree with clinicians: rate answers blind"),
        ("LLaVA-Med (2023)", [0, 1, 0, 0, 0], "cheap curriculum tuning; GPT-4-written training data"),
        ("PMC-VQA (2023)", [0, 1, 0, 0, 0], "caption-made questions can be answered from text alone"),
        ("PubMedCLIP (2023)", [0, 0, 0, 0, 0], "the image encoder matters more than the text encoder"),
    ]
    x0, colw, rowh, top = 2.5, 1.0, 0.56, 4.15
    for j, c in enumerate(cols):
        ax.text(x0 + j * colw + colw / 2, top + 0.35, c.replace(" ", "\n", 1) if (len(c) > 14 and "\n" not in c) else c, ha="center",
                va="center", fontsize=10.6, color=NAVY, fontweight="bold", linespacing=1.1)
    ax.text(x0 + 5 * colw + 0.2, top + 0.35, "what we took from it", fontsize=11, fontweight="bold", color=NAVY, va="center")
    for i, (name, marks, note) in enumerate(rows):
        y = top - (i + 1) * rowh
        key = i < 2
        if key:
            ax.add_patch(Rectangle((0.05, y - 0.02), 12.3, rowh - 0.06, fc="#E3F2F2", ec="none"))
        ax.text(0.15, y + rowh / 2 - 0.04, name, fontsize=12, fontweight="bold" if key else "normal", color=NAVY, va="center")
        for j, m in enumerate(marks):
            cx = x0 + j * colw + colw / 2
            ax.add_patch(Circle((cx, y + rowh / 2 - 0.04), 0.12, fc=TEAL if m else WHITE, ec=TEAL if m else MID, lw=1.4))
        ax.text(x0 + 5 * colw + 0.2, y + rowh / 2 - 0.04, note, fontsize=10.2, color=GREY, va="center")
    y = top - 7 * rowh - 0.15
    ax.add_patch(Rectangle((0.05, y - 0.02), 12.3, rowh - 0.02, fc="#FDEDE8", ec=ORANGE, lw=1.4))
    ax.text(0.15, y + rowh / 2 - 0.02, "This project", fontsize=12, fontweight="bold", color=ORANGE, va="center")
    for j, m in enumerate([1, 1, 1, 0.5, 1]):
        cx = x0 + j * colw + colw / 2
        ax.add_patch(Circle((cx, y + rowh / 2 - 0.02), 0.12, fc=ORANGE if m == 1 else WHITE, ec=ORANGE, lw=2))
    ax.text(x0 + 5 * colw + 0.2, y + rowh / 2 - 0.02, "none of the papers combines all of these (hollow = planned)", fontsize=10.2,
            color=ORANGE, va="center", fontweight="bold")
    save(fig, "fig_papers_matrix")


# ------------------------------------------------------------------------------------------ 4
def fig_gaps_to_plan():
    fig, ax = canvas(12.4, 5.0)
    rows = [
        ("CheXagent is weakest on left/right and severity", "Test perception on its own:\nmirror-image test, heart-size wordings", TEAL),
        ("Naive retrieval makes answers worse (RULE)", "Few passages, a no-harm check, and an arm that\ngives the disease name without any text", TEAL),
        ("Automatic scores disagree with clinicians", "Blind 0-1-2 rating, plus a constant-answer floor\nthat a real system must beat", TEAL),
        ("No reviewed paper retrieves textbook\nknowledge for chest X-rays", "BM25 search over MedlinePlus + medical textbooks,\nindependent of the test data", TEAL),
    ]
    ax.text(0.3, 4.7, "What the papers show", fontsize=13.5, fontweight="bold", color=GREY)
    ax.text(7.35, 4.7, "What we decided to do", fontsize=13.5, fontweight="bold", color=GREY)
    for i, (a, b, c) in enumerate(rows):
        y = 3.55 - i * 1.12
        box(ax, 0.3, y, 6.1, 0.92, a, fc=LIGHT, ec=LIGHT, fs=13.5, tc=NAVY)
        arrow(ax, 6.55, y + 0.46, 7.15, y + 0.46, color=c, lw=2.6)
        box(ax, 7.3, y, 5.0, 0.92, b, fc=WHITE, ec=c, fs=12, tc=NAVY, lw=2)
    save(fig, "fig_gaps_to_plan")


# ------------------------------------------------------------------------------------------ 5
def fig_roadmap():
    fig, ax = canvas(12.4, 4.3)
    steps = [
        ("Phase 1", "Zero-shot baseline\n224 questions", "run", "text-only*"),
        ("Phase 1b", "Knowledge + side\ntest sets (270)", "run", "text-only*"),
        ("Phase 1c", "Perception\ndiagnostics", "run", "text-only*"),
        ("Phase 1d", "Can it name the\ndisease?", "run", "exposed a bug"),
        ("Fix", "Image patch +\nIMAGE CHECK", "done", "done"),
        ("Phase 2", "Knowledge RAG\n(retrieval check done)", "ready", "model run ready"),
        ("Phase 3", "Targeted\nfine-tuning", "later", "optional"),
    ]
    colours = {"run": AMBER, "done": GREEN, "ready": TEAL, "later": MID}
    w, gap = 1.62, 0.18
    for i, (t, d, st, note) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        c = colours[st]
        ax.add_patch(Polygon([[x, 2.0], [x + w - 0.25, 2.0], [x + w, 2.75], [x + w - 0.25, 3.5], [x, 3.5], [x + 0.25, 2.75]],
                             fc=c, ec="none"))
        ax.text(x + w / 2 + 0.05, 2.75, t, ha="center", va="center", fontsize=13.5, fontweight="bold", color=WHITE if st != "later" else NAVY)
        ax.text(x + w / 2, 1.35, d, ha="center", va="center", fontsize=11.3, color=NAVY, linespacing=1.2)
        ax.text(x + w / 2, 0.72, note, ha="center", va="center", fontsize=10.6, color=GREY, style="italic")
    for c, t, x in ((AMBER, "run on Colab", 0.3), (GREEN, "done", 2.6), (TEAL, "built, waiting for model run", 3.8), (MID, "optional / later", 7.2)):
        ax.add_patch(Rectangle((x, 3.95), 0.22, 0.22, fc=c, ec="none"))
        ax.text(x + 0.32, 4.06, t, fontsize=11, va="center", color=NAVY)
    ax.text(0.1, 0.2, "* the model could not see the image in these runs; the results are a text-only control (see the turning point)",
            fontsize=10.6, color=ORANGE, va="center")
    save(fig, "fig_roadmap")


# ------------------------------------------------------------------------------------------ 6
def fig_eval_protocol():
    fig, ax = canvas(12.4, 4.9)
    steps = ["Write the rules\nbefore any run", "Freeze test sets,\nscorer, settings", "Run the model\n(Colab T4)",
             "Score: item recall vs\nconstant-answer floor", "Paired tests +\nblind rating", "Report, including\nnegative results"]
    w, gap = 1.85, 0.25
    for i, s in enumerate(steps):
        x = 0.15 + i * (w + gap)
        box(ax, x, 3.35, w, 1.1, s, fc=NAVY if i in (0, 5) else WHITE, ec=NAVY, tc=WHITE if i in (0, 5) else NAVY, fs=12, lw=1.8)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.02, 3.9, x + w + gap - 0.02, 3.9, ms=14)
    ax.text(0.15, 2.85, "Test sets used so far (all chest X-rays, SLAKE + VQA-RAD)", fontsize=13, fontweight="bold", color=NAVY)
    tiles = [("219", "baseline\nquestions"), ("130", "knowledge\nquestions"), ("140", "left / right\nquestions"),
             ("120", "mirror-image\npairs"), ("47", "heart-size\nquestions"), ("130", "images with a\ndisease label")]
    for i, (n, t) in enumerate(tiles):
        x = 0.15 + i * (w + gap)
        box(ax, x, 1.15, w, 1.35, fc=LIGHT, ec=LIGHT)
        ax.text(x + w / 2, 1.98, n, ha="center", va="center", fontsize=26, fontweight="bold", color=TEAL)
        ax.text(x + w / 2, 1.42, t, ha="center", va="center", fontsize=11, color=NAVY, linespacing=1.1)
    ax.text(0.15, 0.62, "Statistics: bootstrap resampled by image (several questions share an image), exact McNemar test, exact sign test.",
            fontsize=11.3, color=GREY)
    ax.text(0.15, 0.22, "Scorer: item recall (share of the gold answer's items the answer covers); frozen as version 2.", fontsize=11.3, color=GREY)
    save(fig, "fig_eval_protocol")


# ------------------------------------------------------------------------------------------ 7
def fig_text_only_results():
    d = pd.read_csv("results/run_2/baseline_results.csv")
    d = d[~d.id.isin(set(Path("results/run_2/excluded_ids.txt").read_text().split()))]
    k = pd.read_csv("results/run_3/knowledge_results.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [0.9, 1.7, 1.0]})
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.84, wspace=0.6)

    # (a) accuracy by question type
    ax = axs[0]
    groups = [("general", "General"), ("laterality", "Left/right"), ("severity", "Severity")]
    vals, los, his = [], [], []
    for key, _ in groups:
        g = d[d.category == key]
        c, n = int(g.exact_match.sum()), len(g)
        lo, hi = wilson_interval(c, n)
        vals.append(c / n)
        los.append(c / n - lo)
        his.append(hi - c / n)
    ax.bar([g[1] for g in groups], vals, color=[TEAL, ORANGE, AMBER], yerr=[los, his], capsize=4, ecolor=NAVY, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, his[i] + v + 0.03, f"{v:.0%}", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, .5, 1], ["0", "50%", "100%"])
    ax.set_title("Answers right, by question type\n(219 questions)", fontsize=12.5, color=NAVY, loc="left")

    # (b) knowledge vs constant floor
    ax = axs[1]
    subs = [("symptoms", "symptoms"), ("treatment", "treatment"), ("prevention", "prevention"), ("cause", "cause")]
    import numpy as np
    k = k.assign(r=[item_scores(str(p), str(g)).recall for p, g in zip(k.prediction, k.gold_answer)])
    kk = k[k.knowledge_subtype != "function"]
    floor = constant_floor([{"knowledge_subtype": r.knowledge_subtype, "gold_answer": r.gold_answer} for r in kk.itertuples()])
    x = np.arange(len(subs) + 1)
    model = [kk[kk.knowledge_subtype == s].r.mean() for s, _ in subs] + [kk.r.mean()]
    fl = [floor[s] for s, _ in subs] + [floor["all"]]
    ax.bar(x - 0.19, model, 0.36, color=TEAL, label="model (no image)")
    ax.bar(x + 0.19, fl, 0.36, color=MID, label="always-the-common-answer")
    ax.set_xticks(x, [s[1] for s in subs] + ["all"], fontsize=9.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .5, 1], ["0", "0.5", "1.0"])
    ax.legend(frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Knowledge answers: item recall\n(105 questions)", fontsize=12.5, color=NAVY, loc="left")

    # (c) left/right behaviour
    ax = axs[2]
    ident = 112 / 120   # docs/phase1c_findings.md: byte-identical answers on original and mirrored image
    ax.barh(["Same answer,\nmirrored image", "\"Right\" on explicit\nside questions"],
            [ident, 26 / 29], color=[ORANGE, ORANGE], height=0.5)
    for i, (v, t) in enumerate([(ident, "112 of 120"), (26 / 29, "26 of 29")]):
        ax.text(v + 0.02, i, t, va="center", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, .5, 1], ["0", "50%", "100%"])
    ax.set_title("Left/right behaviour", fontsize=12.5, color=NAVY, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(MID)
        a.tick_params(colors=NAVY, labelsize=10.5)
    save(fig, "fig_text_only_results")


# ------------------------------------------------------------------------------------------ 8
def fig_bug_flow():
    fig, ax = canvas(12.4, 5.5)
    ax.text(0.2, 5.2, "What went wrong", fontsize=14, fontweight="bold", color=ORANGE)
    chain = [("Prompt:\nimage path +\nquestion", LIGHT, NAVY), ("generate()\nin transformers\n4.46.3", LIGHT, NAVY),
             ("passes an empty\ncache, not None", LIGHT, NAVY), ("model code:\n\"if cache is None:\nencode image\"", "#FDEDE8", ORANGE),
             ("image skipped;\nanswer from\ntext only", ORANGE, WHITE)]
    w, gap = 2.15, 0.35
    for i, (t, fc, tc) in enumerate(chain):
        x = 0.2 + i * (w + gap)
        box(ax, x, 3.65, w, 1.25, t, fc=fc, ec=ORANGE if i >= 3 else NAVY, tc=tc, fs=11.8, bold=(i == 4))
        if i < len(chain) - 1:
            arrow(ax, x + w + 0.02, 4.27, x + w + gap - 0.02, 4.27, color=ORANGE if i >= 2 else NAVY, ms=13)
    ax.text(0.2, 3.2, "How we noticed", fontsize=14, fontweight="bold", color=NAVY)
    ev = [("130 of 130", "images got the same answer\n('Pneumonia' / 'No Finding')"),
          ("0.002-0.009", "spread of yes/no scores across\n130 different X-rays")]
    for i, (n, t) in enumerate(ev):
        x = 0.2 + i * 3.4
        box(ax, x, 1.95, 3.2, 1.0, fc=LIGHT, ec=LIGHT)
        ax.text(x + 0.2, 2.62, n, fontsize=19, fontweight="bold", color=ORANGE, va="center")
        ax.text(x + 0.2, 2.2, t, fontsize=10.2, color=NAVY, va="center", linespacing=1.1)
    ax.text(7.35, 3.2, "The fix", fontsize=14, fontweight="bold", color=GREEN)
    box(ax, 7.35, 1.95, 4.85, 1.0, "patch the condition so an empty cache still\nencodes the image", fc=WHITE, ec=GREEN, fs=11.5, lw=2)
    arrow(ax, 9.75, 1.9, 9.75, 1.45, color=GREEN, ms=13)
    box(ax, 7.35, 0.4, 4.85, 1.0, "IMAGE CHECK: same file, black / white / noise\npixels. Output must change, or the run stops.",
        fc=GREEN, ec=GREEN, tc=WHITE, fs=11.2, bold=True)
    ax.text(0.2, 1.45, "Earlier results are kept as a text-only control.", fontsize=11.5, color=GREY, style="italic")
    ax.text(0.2, 1.05, "Every notebook now carries the check.", fontsize=11.5, color=GREY, style="italic")
    save(fig, "fig_bug_flow")


# ------------------------------------------------------------------------------------------ 9
def fig_rag_pipeline():
    fig, ax = canvas(12.4, 5.3)
    box(ax, 0.15, 3.55, 2.55, 1.35, "Question\n\"What symptoms will the\ndisease on the left bring?\"", fc=WHITE, ec=NAVY, fs=11.2)
    ax.text(1.42, 3.3, "never names the disease", fontsize=10.5, color=ORANGE, ha="center", style="italic")
    arrow(ax, 2.8, 4.22, 3.25, 4.22)
    box(ax, 3.3, 3.45, 2.75, 1.55, "Step a: name the\ndisease from the\nimage (perception)", fc=TEAL, ec=TEAL, tc=WHITE, fs=12.5, bold=True)
    ax.text(4.67, 3.18, "model's guess  |  true label (oracle)", fontsize=10.2, color=GREY, ha="center", style="italic")
    arrow(ax, 6.15, 4.22, 6.6, 4.22)
    box(ax, 6.65, 3.45, 2.75, 1.55, "Step b: look up\ncause / symptoms /\ntreatment / prevention", fc=NAVY, ec=NAVY, tc=WHITE, fs=12.5, bold=True)
    ax.text(9.75, 2.15, "BM25 search,\n2 short passages", fontsize=10.2, color=GREY, ha="left", style="italic")
    arrow(ax, 9.5, 4.22, 9.95, 4.22)
    box(ax, 10.0, 3.55, 2.3, 1.35, "CheXagent answers\nwith the passages\nin its prompt", fc=WHITE, ec=NAVY, fs=11.5)
    # corpora
    box(ax, 6.65, 1.75, 1.25, 0.8, "MedlinePlus\n(NLM, public)", fc=LIGHT, ec=LIGHT, fs=10.2)
    box(ax, 8.1, 1.75, 1.3, 0.8, "18 medical\ntextbooks", fc=LIGHT, ec=LIGHT, fs=10.2)
    arrow(ax, 7.3, 2.6, 7.6, 3.4, color=GREY, lw=1.6, ms=12)
    arrow(ax, 8.75, 2.6, 8.45, 3.4, color=GREY, lw=1.6, ms=12)
    ax.text(6.65, 1.4, "independent of the test data", fontsize=10.2, color=GREY, style="italic")
    # two-step evidence
    box(ax, 0.15, 0.2, 5.9, 2.35, fc=LIGHT, ec=LIGHT)
    ax.text(0.35, 2.25, "Why two steps: SLAKE's answers are a lookup by disease", fontsize=12.5, fontweight="bold", color=NAVY)
    rows = [("Pneumonia", "cause: bacterial infection", "treatment: medical + supportive therapy"),
            ("Cardiomegaly", "cause: hypertension, dilated", "cardiomyopathy"),
            ("Mass", "prevention: keep healthy", "")]
    ax.text(0.35, 1.78, "same disease  ->  same gold answer, on every image:", fontsize=10.6, color=GREY)
    ax.text(0.35, 1.42, "Pneumonia   cause: bacterial infection", fontsize=11.3, color=NAVY)
    ax.text(0.35, 1.06, "Pneumonia   treatment: medical therapy, supportive therapy", fontsize=11.3, color=NAVY)
    ax.text(0.35, 0.7, "Cardiomegaly   cause: hypertension, dilated cardiomyopathy", fontsize=11.3, color=NAVY)
    ax.text(0.35, 0.34, "8 chest diseases cover all 130 labelled images", fontsize=10.6, color=GREY)
    save(fig, "fig_rag_pipeline")


# ------------------------------------------------------------------------------------------ 10
def fig_arms():
    fig, ax = canvas(12.4, 4.5)
    heads = ["Arm", "Disease named", "Retrieved text", "Role"]
    xs = [0.2, 1.6, 5.3, 8.4]
    for h, x in zip(heads, xs):
        ax.text(x, 4.2, h, fontsize=12.5, fontweight="bold", color=NAVY, va="center")
    arms = [("A1", "none", "none", "fair baseline (same answer style)", MID),
            ("A2", "none", "from the question alone", "oracle subset: shows why disease matters", AMBER),
            ("A3", "true disease (oracle)", "none", "upper bound: value of knowing the disease", AMBER),
            ("A4", "true disease (oracle)", "yes", "upper bound: value of retrieval", AMBER),
            ("A5", "model's own guess", "none", "real: guess alone", TEAL),
            ("A6", "model's own guess", "yes", "the system", TEAL)]
    for i, (a, d, t, role, c) in enumerate(arms):
        y = 3.6 - i * 0.55
        ax.add_patch(Rectangle((0.1, y - 0.22), 12.2, 0.46, fc=LIGHT if i % 2 == 0 else WHITE, ec="none"))
        ax.add_patch(Rectangle((0.1, y - 0.22), 0.1, 0.46, fc=c, ec="none"))
        ax.text(0.4, y, a, fontsize=12.5, fontweight="bold", color=NAVY, va="center")
        ax.text(xs[1], y, d, fontsize=11.5, color=NAVY, va="center")
        ax.text(xs[2], y, t, fontsize=11.5, color=NAVY, va="center")
        ax.text(xs[3], y, role, fontsize=11.2, color=GREY, va="center")
    ax.add_patch(Rectangle((0.2, 0.05), 0.22, 0.22, fc=AMBER, ec="none"))
    ax.text(0.52, 0.16, "oracle arms use SLAKE's own label: upper bounds, never reported as the result", fontsize=10.6, color=NAVY, va="center")
    ax.add_patch(Rectangle((7.2, 0.05), 0.22, 0.22, fc=TEAL, ec="none"))
    ax.text(7.52, 0.16, "real arms use only what the model outputs", fontsize=10.6, color=NAVY, va="center")
    save(fig, "fig_arms")


# ------------------------------------------------------------------------------------------ 11
def fig_retrieval_results():
    d = json.load(open("results/phase2/retrieval_check.json"))
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.88, wspace=0.3)
    ax = axs[0]
    import numpy as np
    labels = ["true disease", "wrong disease", "question only"]
    keys = ["true disease", "wrong disease", "question only"]
    held = [d["held-out"][k]["strict"] for k in keys]
    dev = [d["dev"][k]["strict"] for k in keys]
    x = np.arange(3)
    ax.bar(x - 0.19, dev, 0.36, color=MID, label="dev, 16 questions (used to tune)")
    ax.bar(x + 0.19, held, 0.36, color=[TEAL, AMBER, ORANGE], label="held-out, 34 questions")
    for i, v in enumerate(held):
        ax.text(i + 0.19, v + 0.015, f"{v:.0%}", ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.set_xticks(x, labels, fontsize=11.5)
    ax.set_ylim(0, 0.55)
    ax.set_yticks([0, .25, .5], ["0", "25%", "50%"])
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Share of the gold answer's items found in the retrieved text", fontsize=12, color=NAVY, loc="left")
    ax = axs[1]
    sub = ["symptoms", "cause", "treatment", "prevention"]
    allv = [d["all"]["by_subtype_true_disease"][s]["strict"] for s in sub]
    ns = [d["all"]["by_subtype_true_disease"][s]["n"] for s in sub]
    ax.barh(sub[::-1], allv[::-1], color=TEAL, height=0.55)
    for i, (v, n) in enumerate(zip(allv[::-1], ns[::-1])):
        ax.text(v + 0.015, i, f"{v:.0%}  (n={n})", va="center", fontsize=11.5, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 0.95)
    ax.set_xticks([0, .5], ["0", "50%"])
    ax.set_title("By question type, true disease, all 50 questions", fontsize=12, color=NAVY, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(MID)
        a.tick_params(colors=NAVY, labelsize=11)
    save(fig, "fig_retrieval_results")


if __name__ == "__main__":
    for f in (fig_cxr_anatomy, fig_vqa_task, fig_papers_matrix, fig_gaps_to_plan, fig_roadmap, fig_eval_protocol,
              fig_text_only_results, fig_bug_flow, fig_rag_pipeline, fig_arms, fig_retrieval_results):
        f()
