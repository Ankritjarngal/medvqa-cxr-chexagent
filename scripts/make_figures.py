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


def fig_perception_slide():
    seen = _diag_numbers("results/run_6_with_image/diagnostics_results.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.4))
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.14, top=0.8, wspace=0.3)
    panels = [
        (seen[0]["swap_rate"], ("Model", "Ignores\nthe image"), 0.0, "Answer changes side\nwhen X-ray is mirrored"),
        (seen[1]["acc_overall"], ("Model", "Guessing"), 0.5, "Left/right correct\n(60 questions)"),
        (seen[2]["accuracy"], ("Model", "Always\nanswers no"), seen[2]["always_no_accuracy"], "Heart enlarged? Correct\n(47 questions)"),
    ]
    for ax, (v, labels, ref, title) in zip(axs, panels):
        ax.bar(labels, [v, ref], color=[TEAL, LINE], width=0.55)
        ax.text(0, v + 0.02, f"{v:.0%}", ha="center", fontsize=22, fontweight="bold")
        ax.text(1, ref + 0.02, f"{ref:.0%}", ha="center", fontsize=18, color=GREY)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=14)
        ax.set_title(title, fontsize=15, loc="left", pad=10)
        clean(ax, left=False)
    save(fig, "fig_perception_slide")


def fig_disease_results():
    import csv
    from collections import defaultdict
    from baseline.disease_labels import class_table, disease_report, parse_diseases
    rows = list(csv.DictReader(open("results/run_7_disease_with_image/disease_results.csv", newline="")))
    by, scores = defaultdict(list), {}
    for r in rows:
        labels = [x.strip() for x in r["gold"].split(",")]
        if r["method"] == "perclass":
            s = json.loads(r["raw"])
            scores[r["id"]] = (s, labels)
            pred = sorted(s, key=lambda c: -s[c])[:1]
        else:
            pred = parse_diseases(r["raw"])
        by[r["method"]].append({"id": r["id"], "labels": labels, "pred": pred, "split": r["split"]})
    rep = {m: disease_report(by[m], n_shuffles=200) for m in ("open", "forced", "perclass")}
    majority = rep["open"]["majority_acc"]
    table = class_table({i: s for i, (s, _) in scores.items()}, {i: l for i, (_, l) in scores.items()})

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.14, top=0.82, wspace=0.3)
    ax = axs[0]
    labels = ["Open\nquestion", "Pick from\nthe list", "Yes/no for\neach disease", "Always the most\ncommon label"]
    vals = [rep["open"]["acc_top"], rep["forced"]["acc_top"], rep["perclass"]["acc_top"], majority]
    ax.bar(labels, vals, color=[TEAL, TEAL, TEAL, LINE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=20 if i < 3 else 17, fontweight="bold" if i < 3 else "normal",
                color=DARK if i < 3 else GREY)
    ax.set_ylim(0, 0.72)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=12.5)
    ax.set_title("Disease named correctly (130 images)", fontsize=15, loc="left", pad=10)
    clean(ax, left=False)

    ax = axs[1]
    present = {x for _, (_, l) in scores.items() for x in l}
    items = sorted(((k, v) for k, v in table.items() if k in present), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color=TEAL, height=0.6)
    for yv, (k, v) in enumerate(items):
        ax.text(v - 0.012, yv, f"{v:.2f}", va="center", ha="right", fontsize=13, fontweight="bold", color="white")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color=GREY, lw=1)
    ax.set_xticks([0.5, 1.0], ["0.5 = no skill", "1.0 = perfect"], fontsize=11, color=GREY)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title("Does the yes/no score rank the true disease higher?\n(AUC per disease)", fontsize=14, loc="left", pad=10)
    clean(ax)
    save(fig, "fig_disease_results")


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
    for f in (fig_cxr_real, fig_papers_matrix, fig_test_sets, fig_perception_slide, fig_disease_results, fig_arms, fig_lookup_example,
              fig_retrieval_results):
        f()
