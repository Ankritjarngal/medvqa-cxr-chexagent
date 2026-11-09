import json

import pytest
import requests

from baseline import data


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def iter_content(self, chunk_size=1):
        yield self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


def test_download_retries_429_honours_wait_and_succeeds(tmp_path, monkeypatch):
    responses = [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(503), FakeResponse(200, b"hello")]
    waits = []
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(data.time, "sleep", waits.append)
    out = data._download_file("http://x/f", tmp_path / "f.bin")
    assert out.read_bytes() == b"hello"
    assert waits[0] >= 7 and len(waits) == 2
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_existing_and_fails_fast_on_404(tmp_path, monkeypatch):
    (tmp_path / "have.bin").write_bytes(b"x")
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: pytest.fail("should not download"))
    assert data._download_file("http://x", tmp_path / "have.bin").read_bytes() == b"x"
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(404))
    monkeypatch.setattr(data.time, "sleep", lambda s: pytest.fail("404 must not be retried"))
    with pytest.raises(requests.HTTPError):
        data._download_file("http://x", tmp_path / "missing.bin")


def test_retry_wait_grows_and_is_capped():
    assert data._retry_wait(None, 0, 2.0) == 2.0
    assert data._retry_wait(None, 10, 2.0) == 60.0
    assert data._retry_wait(FakeResponse(429, headers={"Retry-After": "500"}), 0, 2.0) == 120.0


def test_slake_records_from_json_file(tmp_path, monkeypatch):
    rows = [
        {"qid": 1, "img_name": "a/source.jpg", "question": "Is the heart enlarged?", "answer": "No",
         "answer_type": "CLOSED", "q_lang": "en", "location": "Chest_heart", "modality": "X-Ray"},
        {"qid": 2, "img_name": "b/source.jpg", "question": "Which organ?", "answer": "Brain",
         "answer_type": "OPEN", "q_lang": "en", "location": "Head", "modality": "CT"},
        {"qid": 3, "img_name": "a/source.jpg", "question": "Heart?", "answer": "yes",
         "answer_type": "CLOSED", "q_lang": "zh", "location": "Chest_heart", "modality": "X-Ray"},
    ]
    monkeypatch.setattr(data, "_download_file", lambda url, dest, **k: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                                        dest.write_text(json.dumps(rows)), dest)[2])
    monkeypatch.setattr(data, "RAW_CACHE_DIR", tmp_path)
    recs = data.fetch_slake_chest_records("test")
    assert [r["uid"] for r in recs] == ["SLAKE_1"]
    assert recs[0]["category"] == "severity" and recs[0]["answer_format"] == "yes/no"


def test_vqa_rad_parquet_rows_share_images_by_content(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    img_a, img_b = b"\xff\xd8 image A", b"\xff\xd8 image B"
    table = pa.table({
        "image": [{"bytes": img_a, "path": "a.jpg"}, {"bytes": img_a, "path": "a2.jpg"}, {"bytes": img_b, "path": "b.jpg"}],
        "question": ["is the heart enlarged?", "are the lungs clear?", "describe the lesion"],
        "answer": ["yes", "no", "ring enhancing"],
    })
    pq.write_table(table, tmp_path / "vqa_rad_test.parquet")
    monkeypatch.setattr(data, "RAW_CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_download_file", lambda url, dest, **k: dest)
    recs = data.fetch_vqa_rad_chest_records(("test",))
    assert [r["uid"] for r in recs] == ["VQA-RAD_test_0", "VQA-RAD_test_1"]  # 'describe the lesion' is not chest
    assert recs[0]["image_id"] == recs[1]["image_id"]
    paths = data.download_vqa_rad_images(recs, tmp_path / "imgs")
    assert paths[recs[0]["image_key"]] == paths[recs[1]["image_key"]]
    assert open(paths[recs[0]["image_key"]], "rb").read() == img_a
