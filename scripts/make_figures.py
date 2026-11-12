"""Chart and photo figures for the project report and slides (minimalist style).

  python3 scripts/make_figures.py          # charts, tables, annotated X-ray photo
  MMDC=... python3 scripts/render_diagrams.py   # block diagrams (Mermaid sources in deliverables/diagrams/)

Writes deliverables/report/figures/*.pdf (vector, for LaTeX) and *.png (for the slides). Every number is read from the
repository's own result files; the few that are quoted from findings notes carry a comment.
The chest X-ray is a public-domain (CC0) photograph from Wikimedia Commons, kept in figures/src/.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.evaluate import wilson_interval  # noqa: E402
from baseline.knowledge import constant_floor, item_scores  # noqa: E402

import os

BW = os.environ.get("FIG_BW") == "1"          # FIG_BW=1 writes the black-and-white set used by the LaTeX report
SRC = Path("deliverables/report/figures/src")
OUT = Path("deliverables/report/figures_bw" if BW else "deliverables/report/figures")
OUT.mkdir(parents=True, exist_ok=True)

if BW:
    DARK, GREY, LINE, LIGHT = "#000000", "#555555", "#CFCFCF", "#F0F0F0"
    TEAL, ORANGE, AMBER = "#333333", "#7A7A7A", "#A8A8A8"
else:
    DARK, GREY, LINE, LIGHT = "#1F2933", "#6B7785", "#D5DBE1", "#F2F5F7"
    TEAL, ORANGE, AMBER = "#0F8B8D", "#E4572E", "#E9A23B"

for _f in font_manager.findSystemFonts():
    if "arlito" in _f:
        font_manager.fontManager.addfont(_f)
for fam in ("Carlito", "Lato", "DejaVu Sans"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({"text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def clean(ax, left=True):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if not left:
        ax.set_yticks([])


# ------------------------------------------------------------------------------------------ real X-ray
def fig_cxr_real():
    img = Image.open(SRC / "cxr_normal_pa.jpg").convert("L")
    w, h = img.size
    fig, ax = plt.subplots(figsize=(7.6, 5.7))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img, cmap="gray", extent=(0, w, h, 0))
    ax.set_xlim(0, w * 1.62)
    ax.set_ylim(h, 0)
    ax.axis("off")
    sx, sy = w / 787, h / 900          # coordinates below were read off a 787 x 900 preview
    marks = [("Trachea", 372, 200), ("Lung field\n(dark = air)", 640, 330), ("Rib", 715, 450),
             ("Heart", 505, 600), ("Diaphragm", 610, 745), ("Costophrenic angle\n(a sharp corner when healthy)", 702, 792)]
    ys = np.linspace(h * 0.09, h * 0.91, len(marks))
    lx = w * 1.08
    for (name, x, y), ly in zip(marks, ys):
        x, y = x * sx, y * sy
        ax.plot([x, lx - 30], [y, ly], color=TEAL, lw=1.6, zorder=3)
        ax.scatter([x], [y], s=34, color=TEAL, edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(lx, ly, name, va="center", ha="left", fontsize=14.5, color=DARK, linespacing=1.15)
    save(fig, "fig_cxr_real")


# ------------------------------------------------------------------------------------------ papers matrix
def fig_papers_matrix():
    cols = ["Chest X-ray\nfocus", "Free-text\nanswers", "Uses\nretrieval", "Clinician /\nblind rating", "Textbook\nsource"]
    rows = [
        ("CheXagent (2024)", [1, 1, 0, 1, 0], "our baseline model; weakest on left/right and severity (77-79%)", True),
        ("RULE (2024)", [0, 1, 1, 0, 0], "naive retrieval hurts: 47-59% of its errors follow the text", True),
        ("Med-Flamingo (2023)", [0, 1, 0, 1, 0], "automated scores disagree with clinicians: rate blind", True),
        ("LLaVA-Med (2023)", [0, 1, 0, 0, 0], "cheap tuning; GPT-4-written training data", False),
        ("PMC-VQA (2023)", [0, 1, 0, 0, 0], "caption-made questions can be answered from text alone", False),
        ("PubMedCLIP (2023)", [0, 0, 0, 0, 0], "the image encoder matters more than the text encoder", False),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 4.7))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    x0, cw, rh, top = 2.75, 1.08, 0.52, 4.0
    for j, c in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, top + 0.2, c, ha="center", va="bottom", fontsize=12.5, color=GREY, linespacing=1.1)
    ax.text(x0 + 5 * cw + 0.15, top + 0.2, "what we took from it", ha="left", va="bottom", fontsize=12.5, color=GREY)
    for i, (name, marks, note, key) in enumerate(rows):
        y = top - (i + 1) * rh + rh / 2
        ax.plot([0.1, 12.3], [y + rh / 2, y + rh / 2], color=LINE, lw=0.8)
        ax.text(0.1, y, name, va="center", fontsize=14, fontweight="bold" if key else "normal", color=DARK if key else GREY)
        for j, m in enumerate(marks):
            ax.scatter([x0 + j * cw + cw / 2], [y], s=150, color=TEAL if m else "white", edgecolor=TEAL if m else LINE, linewidth=1.4)
        ax.text(x0 + 5 * cw + 0.15, y, note, va="center", fontsize=12.2, color=DARK if key else GREY)
    y = top - 7 * rh + rh / 2 - 0.12
    ax.plot([0.1, 12.3], [y + rh / 2, y + rh / 2], color=DARK, lw=1.2)
    ax.text(0.1, y, "This project", va="center", fontsize=14, fontweight="bold", color=TEAL)
    for j, m in enumerate([1, 1, 1, 0.5, 1]):
        ax.scatter([x0 + j * cw + cw / 2], [y], s=150, color=TEAL if m == 1 else "white", edgecolor=TEAL, linewidth=1.8)
    ax.text(x0 + 5 * cw + 0.15, y, "none of the six papers combines these", va="center", fontsize=12.2,
            color=TEAL, fontweight="bold")
    save(fig, "fig_papers_matrix")


# ------------------------------------------------------------------------------------------ test sets
def fig_test_sets():
    tiles = [("219", "baseline\nquestions"), ("130", "knowledge\nquestions"), ("140", "left / right\nquestions"),
             ("120", "mirror-image\npairs"), ("47", "heart-size\nquestions"), ("130", "images with a\ndisease label")]
    fig, ax = plt.subplots(figsize=(12.4, 1.9))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    for i, (n, t) in enumerate(tiles):
        x = 0.1 + i * 2.05
        ax.text(x, 1.15, n, fontsize=40, fontweight="bold", color=TEAL, va="center")
        ax.text(x, 0.35, t, fontsize=14, color=GREY, va="center", linespacing=1.15)
    save(fig, "fig_test_sets")


# ------------------------------------------------------------------------------------------ text-only results
def fig_text_only_results():
    d = pd.read_csv("results/run_2/baseline_results.csv")
    d = d[~d.id.isin(set(Path("results/run_2/excluded_ids.txt").read_text().split()))]
    k = pd.read_csv("results/run_3/knowledge_results.csv")
    k = k.assign(r=[item_scores(str(p), str(g)).recall for p, g in zip(k.prediction, k.gold_answer)])
    kk = k[k.knowledge_subtype != "function"]
    floor = constant_floor([{"knowledge_subtype": r.knowledge_subtype, "gold_answer": r.gold_answer} for r in kk.itertuples()])

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.3), gridspec_kw={"width_ratios": [0.85, 1.55, 1.0]})
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.12, top=0.8, wspace=0.5)

    ax = axs[0]
    groups = [("general", "General"), ("laterality", "Left/right"), ("severity", "Severity")]
    vals, ups = [], []
    for key, _ in groups:
        g = d[d.category == key]
        c, n = int(g.exact_match.sum()), len(g)
        lo, hi = wilson_interval(c, n)
        vals.append(c / n)
        ups.append(hi)
    ax.bar([g[1] for g in groups], vals, color=[TEAL, TEAL, TEAL], width=0.55)
    for i, (v, u) in enumerate(zip(vals, ups)):
        ax.text(i, v - 0.04, f"{v:.0%}", ha="center", va="top", fontsize=17, fontweight="bold", color="white")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Answers right\n(219 questions)", fontsize=15, loc="left", color=DARK, pad=10)
    ax.tick_params(axis="x", labelsize=13)
    clean(ax, left=False)

    ax = axs[1]
    subs = ["symptoms", "treatment", "prevention", "cause"]
    x = np.arange(len(subs) + 1)
    model = [kk[kk.knowledge_subtype == s].r.mean() for s in subs] + [kk.r.mean()]
    fl = [floor[s] for s in subs] + [floor["all"]]
    ax.bar(x - 0.2, model, 0.38, color=TEAL, label="CheXagent, text only (image skipped)")
    ax.bar(x + 0.2, fl, 0.38, color=LINE, label="copy rule: most common answer")
    ax.set_xticks(x, subs + ["all"])
    ax.tick_params(axis="x", labelsize=11.5)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    for xi, m, f in zip(x, model, fl):
        ax.text(xi - 0.2, m + 0.02, f"{m:.2f}", ha="center", fontsize=11, color=DARK)
        ax.text(xi + 0.2, f + 0.02, f"{f:.2f}", ha="center", fontsize=11, color=GREY)
    ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Knowledge answers: item recall\n(105 questions)", fontsize=15, loc="left", color=DARK, pad=10)
    clean(ax, left=False)

    ax = axs[2]
    ident = 112 / 120   # docs/phase1c_findings.md: byte-identical answers on original and mirrored image
    ax.barh([1, 0], [ident, 26 / 29], color=[ORANGE, ORANGE], height=0.5)
    ax.set_yticks([1, 0], ["Same answer on\nmirrored image", "\"Right\" on explicit\nside questions"], fontsize=12)
    for yv, v, t in ((1, ident, "112 of 120"), (0, 26 / 29, "26 of 29")):
        ax.text(v - 0.03, yv, t, va="center", ha="right", fontsize=16, fontweight="bold", color="white")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Left/right behaviour", fontsize=15, loc="left", color=DARK, pad=10)
    clean(ax)
    save(fig, "fig_text_only_results")


def _diag_numbers(path):
    import csv
    from baseline.diagnostics import flip_report, variant_report, heart_report
    by = {}
    for r in csv.DictReader(open(path, newline="")):
        by.setdefault((r["test"], r["variant"]), {})[r["id"]] = r
    orig, flipped = by[("A_flip", "original")], by[("A_flip", "flipped")]
    flip = flip_report([{"gold_side": o["gold_side"], "explicit": o["explicit"] == "True",
                         "pred_original": o["prediction"], "pred_flipped": flipped[i]["prediction"]}
                        for i, o in orig.items()])["all"]
    side = variant_report([{"gold_side": r["gold_side"], "prediction": r["prediction"]}
                           for r in by[("B_side_prompt", "plain")].values()])
    heart = heart_report([{"gold_yes": r["gold"].strip().lower() == "yes", "prediction": r["prediction"]}
                          for r in by[("C_heart", "plain")].values()], "plain")
    return flip, side, heart


def fig_perception_results():
    skipped = _diag_numbers("results/run_4/diagnostics_results.csv")
    seen = _diag_numbers("results/run_6_with_image/diagnostics_results.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [0.8, 0.8, 1.3]})
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.14, top=0.8, wspace=0.35)
    lab = ["image\nskipped", "image\nvisible"]

    def two(ax, a, b, title, ref=None):
        ax.bar(lab, [a, b], color=[LINE, TEAL], width=0.6)
        for i, v in enumerate([a, b]):
            y = v + 0.02 if ref is None or abs(v - ref) > 0.06 else max(v, ref) + 0.03
            ax.text(i, y, f"{v:.0%}", ha="center", fontsize=20, fontweight="bold", color=DARK)
        if ref is not None:
            ax.axhline(ref, xmax=0.76, color=GREY, lw=1, ls=(0, (4, 3)))
            ax.text(1.36, ref, "chance", fontsize=11, color=GREY, va="center")
            ax.set_xlim(-0.5, 1.9)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=13)
        ax.set_title(title, fontsize=15, loc="left", pad=10)
        clean(ax, left=False)

    two(axs[0], skipped[0]["swap_rate"], seen[0]["swap_rate"], "Answer changes side\nwhen X-ray is mirrored")
    two(axs[1], skipped[1]["acc_overall"], seen[1]["acc_overall"], "Left/right correct\n(60 questions)", ref=0.5)

    ax = axs[2]
    groups = [("Enlarged hearts\nfound (22)", "sensitivity"), ("Normal hearts\ncalled normal (25)", "specificity")]
    x = np.arange(2)
    sv = [skipped[2][k] for _, k in groups]
    iv = [seen[2][k] for _, k in groups]
    ax.bar(x - 0.2, sv, 0.38, color=LINE, label="image skipped")
    ax.bar(x + 0.2, iv, 0.38, color=TEAL, label="image visible")
    for xi, a, b in zip(x, sv, iv):
        ax.text(xi - 0.2, a + 0.02, f"{a:.0%}", ha="center", fontsize=16, color=GREY)
        ax.text(xi + 0.2, b + 0.02, f"{b:.0%}", ha="center", fontsize=18, fontweight="bold")
    ax.set_xticks(x, [g[0] for g in groups])
    ax.tick_params(axis="x", labelsize=13)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=2)
    ax.set_title("Heart enlarged? (47 questions)", fontsize=15, loc="left", pad=10)
    clean(ax, left=False)
    save(fig, "fig_perception_results")


# ------------------------------------------------------------------------------------------ arms table
def fig_arms():
    rows = [("A1", "none", "none", "fair baseline, same answer style", GREY),
            ("A2", "none", "from the question alone", "oracle subset: why the disease matters", AMBER),
            ("A3", "true disease (oracle)", "none", "upper bound: value of knowing the disease", AMBER),
            ("A4", "true disease (oracle)", "yes", "upper bound: value of retrieval", AMBER),
            ("A5", "model's own guess", "none", "real: the guess alone", TEAL),
            ("A6", "model's own guess", "yes", "the system", TEAL)]
    fig, ax = plt.subplots(figsize=(12.4, 4.3))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    xs = [0.45, 1.6, 5.0, 7.9]
    for h, x in zip(["Arm", "Disease named", "Retrieved text", "Role"], xs):
        ax.text(x, 3.95, h, fontsize=13.5, color=GREY, va="center")
    for i, (a, d, t, role, c) in enumerate(rows):
        y = 3.4 - i * 0.55
        ax.plot([0.1, 12.3], [y + 0.28, y + 0.28], color=LINE, lw=0.8)
        ax.scatter([0.18], [y], s=70, color=c)
        ax.text(xs[0], y, a, fontsize=15, fontweight="bold", va="center")
        ax.text(xs[1], y, d, fontsize=14.5, va="center")
        ax.text(xs[2], y, t, fontsize=14.5, va="center")
        ax.text(xs[3], y, role, fontsize=13.5, color=GREY, va="center")
    ax.scatter([0.18], [0.12], s=70, color=AMBER)
    ax.text(0.4, 0.12, "oracle: uses SLAKE's own label, an upper bound, never the result", fontsize=12.5, va="center", color=GREY)
    ax.scatter([7.1], [0.12], s=70, color=TEAL)
    ax.text(7.32, 0.12, "real: uses only what the model outputs", fontsize=12.5, va="center", color=GREY)
    save(fig, "fig_arms")


# ------------------------------------------------------------------------------------------ lookup example
def fig_lookup_example():
    rows = [("Pneumonia", "cause", "Bacterial infection"),
            ("Pneumonia", "treatment", "Medical therapy, supportive therapy"),
            ("Cardiomegaly", "cause", "Hypertension, dilated cardiomyopathy")]
    fig, ax = plt.subplots(figsize=(8.6, 2.2))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    for x, h in zip([0.05, 2.2, 3.75], ["Disease on the image", "Question about", "Gold answer, on every such image"]):
        ax.text(x, 1.95, h, fontsize=12.5, color=GREY, va="center")
    for i, (a, b, c) in enumerate(rows):
        y = 1.38 - i * 0.55
        ax.plot([0.05, 8.5], [y + 0.28, y + 0.28], color=LINE, lw=0.8)
        ax.text(0.05, y, a, fontsize=15, fontweight="bold", va="center", color=TEAL)
        ax.text(2.2, y, b, fontsize=15, va="center")
        ax.text(3.75, y, c, fontsize=15, va="center")
    save(fig, "fig_lookup_example")


# ------------------------------------------------------------------------------------------ retrieval results
def fig_retrieval_results():
    d = json.load(open("results/phase2/retrieval_check.json"))
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.82, wspace=0.28)
    keys = ["true disease", "wrong disease", "question only"]
    held = [d["held-out"][k]["strict"] for k in keys]
    dev = [d["dev"][k]["strict"] for k in keys]
    ax = axs[0]
    x = np.arange(3)
    ax.bar(x - 0.2, dev, 0.38, color=LINE, label="dev, 16 questions (used to tune)")
    ax.bar(x + 0.2, held, 0.38, color=TEAL, label="held-out, 34 questions")
    for i, v in enumerate(held):
        ax.text(i + 0.2, v + 0.012, f"{v:.0%}", ha="center", fontsize=17, fontweight="bold")
    ax.set_xticks(x, keys)
    ax.tick_params(axis="x", labelsize=13.5)
    ax.set_ylim(0, 0.55)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.set_title("Gold items found in the retrieved text", fontsize=15, loc="left", pad=10)
    clean(ax, left=False)
    ax = axs[1]
    sub = ["symptoms", "cause", "treatment", "prevention"]
    allv = [d["all"]["by_subtype_true_disease"][s]["strict"] for s in sub]
    ns = [d["all"]["by_subtype_true_disease"][s]["n"] for s in sub]
    yy = np.arange(len(sub))[::-1]
    ax.barh(yy, allv, color=TEAL, height=0.55)
    ax.set_yticks(yy, sub)
    ax.tick_params(axis="y", labelsize=14)
    for yv, v, n in zip(yy, allv, ns):
        ax.text(v + 0.015, yv, f"{v:.0%}  (n={n})", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 0.95)
    ax.set_xticks([])
    ax.set_title("By question type, true disease (all 50)", fontsize=15, loc="left", pad=10)
    clean(ax)
    save(fig, "fig_retrieval_results")


if __name__ == "__main__":
    for f in (fig_cxr_real, fig_papers_matrix, fig_test_sets, fig_text_only_results, fig_perception_results, fig_arms, fig_lookup_example,
              fig_retrieval_results):
        f()
