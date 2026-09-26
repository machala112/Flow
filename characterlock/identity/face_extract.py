"""Identity extraction: reference images -> canonical face embedding.

Uses InsightFace (buffalo_l pack): detects the largest face in each
reference image, takes its L2-normalized 512-d embedding, averages across
all images, and re-normalizes. The result is stored as
<project>/identity.npz.
"""
import os

import numpy as np

from characterlock.identity.models import BUFFALO_PACK
from characterlock.pipeline.project import IMAGE_EXTS, identity_path

EMBEDDING_DIM = 512


def get_face_app(det_size=(640, 640), ctx_id=-1):
    """Build a CPU (ctx_id=-1) or GPU (ctx_id>=0) FaceAnalysis app."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=BUFFALO_PACK)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def largest_face(faces):
    """Pick the largest face by bounding-box area."""
    return max(faces,
               key=lambda f: (f.bbox[2] - f.bbox[0]) *
                             (f.bbox[3] - f.bbox[1]))


def extract_identity(refs_dir, progress=None, _app=None):
    """Extract the canonical embedding from a folder of reference images.

    Returns {"embedding": (512,) float32, "n_images": int,
             "n_faces": int, "model": str}. Images with no detectable face
    are skipped; raises ValueError if none yields a face.
    """
    import cv2

    app = _app if _app is not None else get_face_app()
    images = [f for f in sorted(os.listdir(refs_dir))
              if f.lower().endswith(IMAGE_EXTS)]
    if not images:
        raise ValueError(f"no images found in {refs_dir}")

    embs = []
    for i, fname in enumerate(images):
        img = cv2.imread(os.path.join(refs_dir, fname))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        embs.append(largest_face(faces).normed_embedding.astype(np.float64))
        if progress:
            progress(i + 1, len(images))

    if not embs:
        raise ValueError(
            f"no faces detected in any of {len(images)} reference images "
            f"under {refs_dir}")

    mean = np.mean(embs, axis=0)
    mean /= np.linalg.norm(mean)
    return {
        "embedding": mean.astype(np.float32),
        "n_images": len(images),
        "n_faces": len(embs),
        "model": f"insightface:{BUFFALO_PACK}",
    }


def save_identity(project_path, identity):
    """Persist the identity dict to <project>/identity.npz. Returns path."""
    path = identity_path(project_path)
    np.savez(path,
             embedding=identity["embedding"],
             n_images=np.int64(identity["n_images"]),
             n_faces=np.int64(identity["n_faces"]),
             model=np.str_(identity["model"]))
    return path


def load_identity(project_path):
    """Load <project>/identity.npz back into a dict."""
    data = np.load(identity_path(project_path), allow_pickle=False)
    emb = data["embedding"]
    if emb.shape != (EMBEDDING_DIM,):
        raise ValueError(f"bad embedding shape {emb.shape} in {project_path}")
    return {
        "embedding": emb.astype(np.float32),
        "n_images": int(data["n_images"]),
        "n_faces": int(data["n_faces"]),
        "model": str(data["model"]),
    }
