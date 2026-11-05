from baseline.model import patch_version_asserts

ASSERT_LINE = (
    'assert transformers.__version__ == "4.40.0", '
    '"Please install a specific HF transformers version: pip install transformers==4.40.0"'
)


def test_patches_assert_lines_in_all_py_files(tmp_path):
    (tmp_path / "modeling_chexagent.py").write_text(f"import transformers\n{ASSERT_LINE}\nx = 1\n")
    (tmp_path / "modeling_visual.py").write_text(f"{ASSERT_LINE}\ny = 2\n")
    (tmp_path / "config.json").write_text("{}")

    assert patch_version_asserts(str(tmp_path)) == 2

    for name in ("modeling_chexagent.py", "modeling_visual.py"):
        text = (tmp_path / name).read_text()
        assert "4.40.0" not in text
        assert "pass" in text
    assert "x = 1" in (tmp_path / "modeling_chexagent.py").read_text()


def test_idempotent_and_leaves_other_asserts_alone(tmp_path):
    (tmp_path / "a.py").write_text(f"{ASSERT_LINE}\nassert 1 == 1\n")
    assert patch_version_asserts(str(tmp_path)) == 1
    assert patch_version_asserts(str(tmp_path)) == 0
    assert "assert 1 == 1" in (tmp_path / "a.py").read_text()
