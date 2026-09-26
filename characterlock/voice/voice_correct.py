"""Per-clip voice correction with OpenVoice V2.

For each clip: extract its audio, estimate the *source* speaker embedding
from the clip itself, convert the clip's speech toward the *target*
(reference) voice embedding — preserving content and timing — then mux the
converted audio back onto the video.
"""
import os
import tempfile

import numpy as np

from characterlock.pipeline import video_io
from characterlock.voice import voice_extract
from characterlock.voice.models import get_converter

# OpenVoice V2 always outputs 22050 Hz mono; we resample back to the
# clip's native audio rate at mux time via ffmpeg.
OUTPUT_SR = 22050


def _to_se_tensor(embedding, device):
    """(256,) embedding -> [1, 256, 1] tensor the converter expects."""
    import torch
    arr = np.asarray(embedding, dtype=np.float32).reshape(1, -1, 1)
    if arr.shape[1] != 256:
        raise ValueError(f"bad voice embedding shape {arr.shape}")
    return torch.from_numpy(arr).to(device)


def correct_clip_voice(video_path, voice_embedding, out_path,
                       tau=0.3, progress=None, _converter=None):
    """Convert one clip's speech to the reference voice.

    voice_embedding: (256,) array from voice_extract.load_voice_embedding.
    Returns out_path. Clips without an audio track are copied unchanged.
    """
    info = video_io.probe(video_path)
    if not info["has_audio"]:
        # nothing to convert — copy the video stream as-is
        import shutil
        shutil.copyfile(video_path, out_path)
        return out_path

    conv = _converter if _converter is not None else get_converter(
        progress=progress)
    device = conv.device
    tgt_se = _to_se_tensor(voice_embedding, device)

    with tempfile.TemporaryDirectory(prefix="characterlock-vc") as tmp:
        src_wav = os.path.join(tmp, "src.wav")
        video_io.extract_audio(video_path, src_wav, sample_rate=44100,
                               channels=1)
        src_se = conv.extract_se([src_wav]).to(device)

        converted = os.path.join(tmp, "converted.wav")
        conv.convert(audio_src_path=src_wav, src_se=src_se, tgt_se=tgt_se,
                     output_path=converted, tau=tau)
        if progress:
            progress("muxing converted audio")
        # -ar native rate: resample 22050 Hz output back to a standard rate
        tmp_r = os.path.join(tmp, "converted48k.wav")
        video_io._run(["ffmpeg", "-y", "-v", "error", "-i", converted,
                       "-ar", "48000", "-ac", "1", tmp_r])
        video_io.mux_audio(video_path, tmp_r, out_path)
    return out_path
