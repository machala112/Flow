"""Project folder management.

A project is a folder holding everything CharacterLock learns about one
character:

    <project>/
        project.json          # refs dir, voice path, created_at, version
        identity.npz          # canonical face embedding (+ metadata)
        voice_embedding.npy   # speaker embedding
"""
import json
import os
import time

from characterlock import __version__

PROJECT_FILE = "project.json"
IDENTITY_FILE = "identity.npz"
VOICE_FILE = "voice_embedding.npy"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _abspath(p):
    return os.path.abspath(os.path.expanduser(p))


def init_project(project_path, refs_dir, voice_path):
    """Create/validate a project folder; write project.json. Returns dict."""
    project_path, refs_dir, voice_path = map(
        _abspath, (project_path, refs_dir, voice_path))

    if not os.path.isdir(refs_dir):
        raise ValueError(f"refs dir not found: {refs_dir}")
    images = [f for f in sorted(os.listdir(refs_dir))
              if f.lower().endswith(IMAGE_EXTS)]
    if len(images) < 1:
        raise ValueError(f"no images in refs dir: {refs_dir}")
    if not os.path.isfile(voice_path):
        raise ValueError(f"voice file not found: {voice_path}")

    os.makedirs(project_path, exist_ok=True)
    meta = {
        "version": __version__,
        "refs_dir": refs_dir,
        "n_ref_images": len(images),
        "voice_path": voice_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(project_path, PROJECT_FILE), "w") as f:
        json.dump(meta, f, indent=2)
    return {"path": project_path, **meta}


def load_project(project_path):
    """Load project.json; raise if the folder is not a project."""
    project_path = _abspath(project_path)
    cfg_path = os.path.join(project_path, PROJECT_FILE)
    if not os.path.isfile(cfg_path):
        raise ValueError(f"not a CharacterLock project: {project_path} "
                         f"(run `characterlock init` first)")
    with open(cfg_path) as f:
        meta = json.load(f)
    return {"path": project_path, **meta}


def identity_path(project_path):
    return os.path.join(_abspath(project_path), IDENTITY_FILE)


def voice_embedding_path(project_path):
    return os.path.join(_abspath(project_path), VOICE_FILE)
