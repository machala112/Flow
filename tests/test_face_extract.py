"""Tests for identity.face_extract — no model downloads needed (fake app)."""
import numpy as np
import pytest

from characterlock.identity import face_extract
from characterlock.pipeline import project as project_mod


class _FakeFace:
    def __init__(self, emb, box):
        self.normed_embedding = emb
        self.bbox = np.array(box, dtype=np.float32)


class _FakeApp:
    """Returns two faces per image: a small one and a big one with a
    fixed embedding, so we can check largest-face selection + averaging."""

    def __init__(self, emb):
        self.emb = emb
        self.calls = 0

    def get(self, img):
        self.calls += 1
        return [_FakeFace(self.emb, (0, 0, 10, 10)),      # small
                _FakeFace(self.emb * 2, (0, 0, 100, 100))]  # big (picked)


def _write_png(path):
    # 1x1 PNG; content irrelevant — cv2.imread is bypassed via _app? No:
    # extract_identity calls cv2.imread itself. We monkeypatch cv2 instead.
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")


def test_extract_averages_and_normalizes(tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(4):
        _write_png(str(refs / f"r{i}.png"))

    emb = np.random.RandomState(0).rand(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    fake = _FakeApp(emb)

    import cv2
    monkeypatch.setattr(cv2, "imread", lambda p: np.zeros((8, 8, 3), np.uint8))

    ident = face_extract.extract_identity(str(refs), _app=fake)
    assert fake.calls == 4
    # biggest face picked -> embedding doubled then normalized == emb
    np.testing.assert_allclose(ident["embedding"], emb, atol=1e-6)
    assert abs(np.linalg.norm(ident["embedding"]) - 1.0) < 1e-6
    assert ident["n_images"] == 4 and ident["n_faces"] == 4


def test_extract_rejects_faceless_refs(tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    refs.mkdir()
    _write_png(str(refs / "r0.png"))

    class _NoFace:
        def get(self, img):
            return []

    import cv2
    monkeypatch.setattr(cv2, "imread", lambda p: np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="no faces detected"):
        face_extract.extract_identity(str(refs), _app=_NoFace())


def test_save_load_roundtrip(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    emb = np.random.RandomState(1).rand(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    ident = {"embedding": emb, "n_images": 7, "n_faces": 6,
             "model": "insightface:buffalo_l"}
    path = face_extract.save_identity(str(proj), ident)
    assert path.endswith("identity.npz")
    back = face_extract.load_identity(str(proj))
    np.testing.assert_array_equal(back["embedding"], emb)
    assert (back["n_images"], back["n_faces"]) == (7, 6)
    assert back["model"] == "insightface:buffalo_l"
    # project helper agrees on the path
    assert path == project_mod.identity_path(str(proj))
