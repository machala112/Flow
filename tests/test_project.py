"""Tests for pipeline.project."""
import json
import os

import pytest

from characterlock.pipeline import project as project_mod


@pytest.fixture
def refs_dir(tmp_path):
    d = tmp_path / "refs"
    d.mkdir()
    # minimal valid PNGs (1x1) — project init only checks extensions
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
           b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0c"
           b"IDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00"
           b"IEND\xaeB`\x82")
    for i in range(3):
        (d / f"face{i}.png").write_bytes(png)
    return str(d)


@pytest.fixture
def voice_file(tmp_path):
    p = tmp_path / "voice.wav"
    p.write_bytes(b"RIFF....WAVE")  # init only checks existence
    return str(p)


def test_init_and_load(tmp_path, refs_dir, voice_file):
    proj = project_mod.init_project(str(tmp_path / "myshow"), refs_dir,
                                    voice_file)
    assert proj["n_ref_images"] == 3
    cfg = json.load(open(os.path.join(proj["path"], "project.json")))
    assert cfg["refs_dir"] == refs_dir
    loaded = project_mod.load_project(proj["path"])
    assert loaded["voice_path"] == voice_file
    assert project_mod.identity_path(proj["path"]).endswith("identity.npz")
    assert project_mod.voice_embedding_path(
        proj["path"]).endswith("voice_embedding.npy")


def test_init_rejects_bad_inputs(tmp_path, refs_dir, voice_file):
    with pytest.raises(ValueError):
        project_mod.init_project(str(tmp_path / "p"), "/nonexistent",
                                 voice_file)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        project_mod.init_project(str(tmp_path / "p"), str(empty), voice_file)
    with pytest.raises(ValueError):
        project_mod.init_project(str(tmp_path / "p"), refs_dir,
                                 "/nonexistent.wav")
    with pytest.raises(ValueError):
        project_mod.load_project(str(tmp_path / "never-initialised"))
