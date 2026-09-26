"""Tests for pipeline.video_io — uses ffmpeg-generated synthetic clips only."""
import os

import pytest

from characterlock.pipeline import video_io

ffmpeg_ok, ffprobe_ok = video_io.check_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(not (ffmpeg_ok and ffprobe_ok),
                                  reason="ffmpeg/ffprobe not on PATH")


def _make_clip(path, duration=2, size="320x240", fps=10):
    video_io._run(["ffmpeg", "-y", "-v", "error",
                   "-f", "lavfi", "-i",
                   f"testsrc=duration={duration}:size={size}:rate={fps}",
                   "-f", "lavfi", "-i",
                   f"sine=frequency=440:duration={duration}:sample_rate=44100",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-shortest", path])


@needs_ffmpeg
def test_probe_and_list_clips(tmp_path):
    clip = str(tmp_path / "scene1.mp4")
    _make_clip(clip)
    info = video_io.probe(clip)
    assert info["width"] == 320 and info["height"] == 240
    assert info["has_audio"] is True
    assert info["duration"] == pytest.approx(2.0, abs=0.2)
    (tmp_path / "notes.txt").write_text("not a video")
    assert video_io.list_clips(str(tmp_path)) == [clip]


@needs_ffmpeg
def test_frame_audio_roundtrip(tmp_path):
    clip = str(tmp_path / "scene1.mp4")
    _make_clip(clip, duration=1)
    frames_dir = str(tmp_path / "frames")
    frames = video_io.extract_frames(clip, frames_dir)
    assert len(frames) == 10  # 1s @ 10fps

    silent = str(tmp_path / "silent.mp4")
    video_io.assemble_video(frames_dir, silent, fps=10)
    assert video_io.probe(silent)["has_audio"] is False

    wav = str(tmp_path / "audio.wav")
    video_io.extract_audio(clip, wav)
    assert os.path.getsize(wav) > 1000

    muxed = str(tmp_path / "muxed.mp4")
    video_io.mux_audio(silent, wav, muxed)
    assert video_io.probe(muxed)["has_audio"] is True


@needs_ffmpeg
def test_concat(tmp_path):
    clips = []
    for i in range(2):
        p = str(tmp_path / f"c{i}.mp4")
        _make_clip(p, duration=1)
        clips.append(p)
    out = str(tmp_path / "stitched.mp4")
    video_io.concat_videos(clips, out)
    assert video_io.probe(out)["duration"] == pytest.approx(2.0, abs=0.3)
