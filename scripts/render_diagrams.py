"""Renders deliverables/diagrams/*.mmd (Mermaid) to PNG (slides) and PDF (LaTeX) in deliverables/report/figures/.

  npm install @mermaid-js/mermaid-cli      # once; needs Chrome/Chromium
  MMDC=node_modules/.bin/mmdc python3 scripts/render_diagrams.py
"""
import os
import subprocess
from pathlib import Path

SRC = Path("deliverables/diagrams")
OUT = Path("deliverables/report/figures")
MMDC = os.environ.get("MMDC", "mmdc")


def render(src: Path):
    base = ["-p", str(SRC / "puppeteer.json"), "-c", str(SRC / "mermaid.config.json"), "-i", str(src)]
    subprocess.run([MMDC, *base, "-o", str(OUT / f"{src.stem}.png"), "-s", "3", "-b", "white"], check=True, capture_output=True)
    subprocess.run([MMDC, *base, "-o", str(OUT / f"{src.stem}.pdf")], check=True, capture_output=True)
    print("rendered", src.stem)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("*.mmd")):
        render(f)
