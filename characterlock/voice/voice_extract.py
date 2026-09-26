"""Voice embedding extraction with OpenVoice V2.

Extracts a 256-d speaker embedding from the reference voice clip
(10-30 s). The embedding is stored as <project>/voice_embedding.npy.
"""
import os
import tempfile

import numpy as np

from characterlock.pipeline.project import voice_embedding_path
from characterlock.voice.models import get_converter

EMBEDDING_DIM = 256


def _to_numpy(se_tensor):
    import torch
    if isinstance(se_tensor, torch.Tensor):
        se_tensor = se_tensor.detach().cpu()
    arr = np.asarray(se_tensor, dtype=np.float32).reshape(-1)
    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(f"unexpected speaker embedding shape {arr.shape}")
    return arr


def extract_voice(voice_path, progress=None, _converter=None):
    """Extract the target speaker embedding from a reference audio file.

    Returns a (256,) float32 numpy array. Calls the converter's
    ``extract_se`` directly on the file: this avoids
    ``se_extractor.get_se``, whose non-VAD branch hardcodes CUDA and whose
    VAD branch needs whisper models. The reference clip is expected to be
    clean speech (10-30 s), so no VAD pre-segmentation is required.
    """
    if not os.path.isfile(voice_path):
        raise ValueError(f"voice file not found: {voice_path}")
    conv = _converter if _converter is not None else get_converter(
        progress=progress)
    se = conv.extract_se([voice_path])
    return _to_numpy(se)


def save_voice_embedding(project_path, embedding):
    """Persist the embedding to <project>/voice_embedding.npy."""
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if emb.shape != (EMBEDDING_DIM,):
        raise ValueError(f"bad voice embedding shape {emb.shape}")
    path = voice_embedding_path(project_path)
    np.save(path, emb)
    return path


def load_voice_embedding(project_path):
    """Load <project>/voice_embedding.npy back into a (256,) array."""
    emb = np.load(voice_embedding_path(project_path)).astype(np.float32)
    if emb.shape != (EMBEDDING_DIM,):
        raise ValueError(f"bad voice embedding shape {emb.shape}")
    return emb
