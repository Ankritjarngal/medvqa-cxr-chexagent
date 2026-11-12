import pytest

from baseline.model import _IMAGE_BRANCH_ORIGINAL, _IMAGE_BRANCH_PATCHED, patch_image_cache_check, patch_version_asserts

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


def test_image_branch_patch_is_applied_once_and_fails_loudly(tmp_path):
    model_py = tmp_path / "modeling_chexagent.py"
    model_py.write_text(f"def forward():\n        {_IMAGE_BRANCH_ORIGINAL}\n            pass\n")
    assert patch_image_cache_check(str(tmp_path)) == 1
    text = model_py.read_text()
    assert _IMAGE_BRANCH_PATCHED in text and _IMAGE_BRANCH_ORIGINAL not in text
    assert patch_image_cache_check(str(tmp_path)) == 1          # idempotent
    assert model_py.read_text() == text
    model_py.write_text("def forward():\n    pass\n")
    with pytest.raises(RuntimeError):
        patch_image_cache_check(str(tmp_path))


def test_patched_condition_is_true_for_none_and_empty_cache_but_not_a_filled_cache():
    class Cache:
        def __init__(self, n): self.n = n
        def get_seq_length(self): return self.n

    def cond(past, has_image_token=True):
        expr = _IMAGE_BRANCH_PATCHED.removeprefix("if ").removesuffix(":").replace(
            "torch.any(input_ids == self.tokenizer.img_start_id)", "has_image_token")
        return eval(expr, {"past_key_values": past, "has_image_token": has_image_token})

    assert cond(None) and cond(Cache(0))
    assert not cond(Cache(12))          # later decoding steps: no image encoding
    assert not cond(None, has_image_token=False)


def test_align_visual_dtype_fixes_mixed_precision_tower():
    torch = pytest.importorskip("torch")
    from baseline.model import align_visual_dtype

    class Tower(torch.nn.Module):
        """Mimics CheXagent's visual tower: a float32 parameter registered on the tower itself comes first in
        parameters(), while the sub-module is float16, and encode() casts pixels to the first parameter's dtype."""
        def __init__(self):
            super().__init__()
            self.pos_embed = torch.nn.Parameter(torch.zeros(4), requires_grad=False)          # float32, first
            self.model = torch.nn.Conv2d(3, 2, 1).half()
            self.attn_pool = torch.nn.Linear(2, 2).half()

        def encode(self, x):
            x = x.to(dtype=next(self.parameters()).dtype)
            return self.attn_pool(self.model(x).flatten(2).transpose(1, 2))

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Tower()

    m = Wrapper()
    with pytest.raises(RuntimeError):
        m.visual.encode(torch.zeros(1, 3, 2, 2))                  # reproduces the Colab failure on CPU
    align_visual_dtype(m, torch.float16)
    assert m.visual.encode(torch.zeros(1, 3, 2, 2)).dtype == torch.float16


def test_align_visual_dtype_fails_loudly_without_a_tower():
    torch = pytest.importorskip("torch")
    from baseline.model import align_visual_dtype
    with pytest.raises(RuntimeError):
        align_visual_dtype(torch.nn.Linear(1, 1), torch.float16)
