"""ffmpeg/ffprobe wrappers: all video<->frames<->audio plumbing.

Every function shells out to the system ffmpeg/ffprobe binaries, so this
module has no heavy Python dependencies. All paths are str.
"""
import json
import os
import shutil
import subprocess

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


class FFmpegError(RuntimeError):
    pass


def _run(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise FFmpegError(f"binary not found: {cmd[0]} — install ffmpeg") from e
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{proc.stderr.strip()[-2000:]}")
    return proc


def check_ffmpeg():
    """Return (ffmpeg_ok, ffprobe_ok)."""
    return shutil.which("ffmpeg") is not None, shutil.which("ffprobe") is not None


def probe(path):
    """Return dict with width, height, fps, duration, nb_frames, has_audio."""
    proc = _run(["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", path])
    info = json.loads(proc.stdout)
    v = next((s for s in info.get("streams", [])
              if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", [])
              if s.get("codec_type") == "audio"), None)
    fps_s = v.get("avg_frame_rate", "0/1")
    num, den = (int(x) for x in fps_s.split("/")) if "/" in fps_s else (0, 1)
    return {
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": (num / den) if den else 0.0,
        "duration": float(info.get("format", {}).get("duration", 0.0)),
        "nb_frames": int(v.get("nb_frames", 0) or 0),
        "has_audio": a is not None,
    }


def list_clips(input_dir):
    """Sorted list of video files in a folder (non-recursive)."""
    files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]
    return [f for f in files
            if os.path.isfile(f) and f.lower().endswith(VIDEO_EXTS)]


def extract_frames(video_path, out_dir, fps=None):
    """Dump frames as PNGs into out_dir; return sorted list of frame paths."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%06d.png")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video_path]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [pattern]
    _run(cmd)
    frames = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    return [os.path.join(out_dir, f) for f in frames]


def extract_audio(video_path, out_wav, sample_rate=16000, channels=1):
    """Extract audio track to wav (default 16 kHz mono for voice models)."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", video_path,
          "-vn", "-ac", str(channels), "-ar", str(sample_rate),
          "-c:a", "pcm_s16le", out_wav])
    return out_wav


def assemble_video(frames_dir, out_path, fps):
    """Re-encode PNG frames (frame_%06d.png) into an mp4 (no audio)."""
    pattern = os.path.join(frames_dir, "frame_%06d.png")
    _run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
          "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-crf", "18", out_path])
    return out_path


def mux_audio(video_path, audio_path, out_path):
    """Mux an audio file onto a video (replaces any existing audio)."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-i", audio_path,
          "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-shortest", out_path])
    return out_path


def concat_videos(inputs, out_path):
    """Concatenate clips (same codec/dims) into one mp4 via concat demuxer."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in inputs:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name
    try:
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", list_path, "-c", "copy", out_path])
    finally:
        os.unlink(list_path)
    return out_path
