import io
import contextlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "run_4" / "diagnostics_results.csv"


@pytest.mark.skipif(not CSV.exists(), reason="run 4 results not present")
def test_analysis_reproduces_headline_numbers():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_diagnostics

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze_diagnostics.main(str(CSV))
    out = buf.getvalue()
    assert "swapped: 2  stayed the same: 71" in out
    assert "9/22 (41%)" in out
