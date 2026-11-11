"""Renders deliverables/diagrams/*.mmd (Mermaid) to PNG (slides) and PDF (LaTeX) in deliverables/report/figures/.

  npm install @mermaid-js/mermaid-cli      # once; needs Chrome/Chromium
  MMDC=node_modules/.bin/mmdc python3 scripts/render_diagrams.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

SRC = Path("deliverables/diagrams")
BW = os.environ.get("FIG_BW") == "1"           # FIG_BW=1 renders the black-and-white set used by the LaTeX report
OUT = Path("deliverables/report/figures_bw" if BW else "deliverables/report/figures")
MMDC = os.environ.get("MMDC", "mmdc")
GREYS = {"0F8B8D": "444444", "1F2933": "000000", "E4572E": "7A7A7A", "2E8B57": "707070", "E9A23B": "B5B5B5",
         "F2F5F7": "F0F0F0", "E5E9ED": "DDDDDD", "6B7785": "555555", "9AA5B1": "888888", "C9D1D9": "BBBBBB"}


def grey(text: str) -> str:
    for old, new in GREYS.items():
        text = text.replace(old, new).replace(old.lower(), new)
    return text


def render(src: Path):
    cfg = SRC / "mermaid.config.json"
    if BW:
        tmp = Path(tempfile.mkdtemp())
        (tmp / src.name).write_text(grey(src.read_text()))
        (tmp / "cfg.json").write_text(grey(cfg.read_text()))
        src, cfg = tmp / src.name, tmp / "cfg.json"
    base = ["-p", str(SRC / "puppeteer.json"), "-c", str(cfg), "-i", str(src)]
    subprocess.run([MMDC, *base, "-o", str(OUT / f"{src.stem}.png"), "-s", "3", "-b", "white"], check=True, capture_output=True)
    subprocess.run([MMDC, *base, "-o", str(OUT / f"{src.stem}.pdf")], check=True, capture_output=True)
    print("rendered", src.stem)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("*.mmd")):
        render(f)
