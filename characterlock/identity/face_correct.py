"""Per-clip face correction: identity swap + restoration.

For each frame of a clip:
  1. Detect faces (InsightFace buffalo_l).
  2. Swap every detected face toward the canonical identity embedding
     (inswapper_128), pasted back into the original frame.
  3. Restore/enhance with GFPGAN to remove artifacts and blend into the
     frame's lighting/skin tone.
Frames are re-encoded with ffmpeg; the original audio track is preserved
(the voice pass replaces it later).
"""
import os
import shutil
import tempfile

import cv2
import numpy as np

from characterlock.identity import face_extract
from characterlock.identity import models as face_models
from characterlock.identity import restore as restore_mod
from characterlock.pipeline import video_io


def source_face_for(embedding):
    """Build the Face-like object inswapper needs as the identity source.

    inswapper.get() only reads ``source_face.normed_embedding``; in
    InsightFace 2.0 that is a read-only property derived from
    ``embedding``, so we set ``embedding`` (already L2-normalized by
    extract_identity) and let the property compute.
    """
    from insightface.app.common import Face
    face = Face()
    face["embedding"] = np.asarray(embedding, dtype=np.float32)
    return face


def correct_frame(img_bgr, app, swapper, source_face, restorer=None,
                  restore_weight=0.5):
    """Swap + restore faces in one BGR frame. Returns the corrected frame."""
    faces = app.get(img_bgr)
    if not faces:
        return img_bgr
    # largest first so small/overlapping faces don't clobber big ones
    faces = sorted(faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                 (f.bbox[3] - f.bbox[1]),
                   reverse=True)
    for target in faces:
        img_bgr = swapper.get(img_bgr, target, source_face, paste_back=True)
    if restorer is not None:
        img_bgr = restore_mod.restore_frame(img_bgr, restorer,
                                            weight=restore_weight)
    return img_bgr


def correct_clip_faces(video_path, identity, out_path, restore=True,
                       restore_weight=0.5, progress=None,
                       _app=None, _swapper=None, _restorer=None):
    """Face-correct every frame of a clip. Returns out_path."""
    info = video_io.probe(video_path)
    fps = info["fps"] or 24.0
    if fps <= 0:
        fps = 24.0

    app = _app if _app is not None else face_extract.get_face_app()
    swapper = _swapper if _swapper is not None else face_models.get_swapper()
    source_face = source_face_for(identity["embedding"])
    restorer = None
    if restore:
        restorer = (_restorer if _restorer is not None
                    else restore_mod.get_restorer(progress=progress))

    with tempfile.TemporaryDirectory(prefix="characterlock-face") as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        frames = video_io.extract_frames(video_path, in_dir)
        n = len(frames)
        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            img = correct_frame(img, app, swapper, source_face, restorer,
                                restore_weight)
            cv2.imwrite(os.path.join(out_dir, f"frame_{i+1:06d}.png"), img)
            if progress:
                progress(i + 1, n, "face")

        silent = os.path.join(tmp, "silent.mp4")
        video_io.assemble_video(out_dir, silent, fps)
        if info["has_audio"]:
            wav = os.path.join(tmp, "orig.wav")
            video_io.extract_audio(video_path, wav, sample_rate=48000,
                                   channels=2)
            video_io.mux_audio(silent, wav, out_path)
        else:
            shutil.move(silent, out_path)
    return out_path
