"""CharacterLock command-line interface.

Two commands:
  characterlock init    --refs <dir> --voice <wav> --project <dir>
  characterlock process --project <dir> --input <dir> --output <dir> [--stitch]
"""
import argparse
import sys

from characterlock import __version__


def cmd_init(args):
    """Register a character: reference faces + reference voice -> project."""
    from characterlock.pipeline import project as project_mod
    from characterlock.identity import face_extract
    from characterlock.voice import voice_extract

    proj = project_mod.init_project(args.project, args.refs, args.voice)
    print(f"project initialised at {proj['path']}")

    emb = face_extract.extract_identity(args.refs)
    face_extract.save_identity(proj["path"], emb)
    print(f"identity saved ({emb['embedding'].shape[0]}-d embedding "
          f"from {emb['n_images']} images)")

    vemb = voice_extract.extract_voice(args.voice)
    voice_extract.save_voice_embedding(proj["path"], vemb)
    print(f"voice embedding saved ({vemb.shape[0]}-d)")

    print("done. now run: characterlock process --project "
          f"{args.project} --input <clips> --output <dir>")
    return 0


def cmd_process(args):
    """Correct a folder of clips against a registered project."""
    import os
    from characterlock.pipeline import project as project_mod
    from characterlock.pipeline import video_io
    from characterlock.identity import face_extract, face_correct
    from characterlock.voice import voice_extract, voice_correct

    proj = project_mod.load_project(args.project)
    identity = face_extract.load_identity(proj["path"])
    voice_emb = voice_extract.load_voice_embedding(proj["path"])

    clips = video_io.list_clips(args.input)
    if not clips:
        print(f"no video clips found in {args.input}", file=sys.stderr)
        return 1
    os.makedirs(args.output, exist_ok=True)
    print(f"{len(clips)} clip(s) -> {args.output} "
          f"{'(stitched)' if args.stitch else '(separate files)'}")

    def progress(*args):
        # face module: progress(i, n, stage); voice module: progress(message)
        if len(args) == 3:
            i, n, stage = args
            print(f"  [{stage}] {i}/{n}", end="\r", flush=True)
        else:
            print(f"  {args[0]}", flush=True)

    corrected = []
    for clip in clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        out_path = os.path.join(args.output, f"{name}_fixed.mp4")
        print(f"processing {os.path.basename(clip)}...")

        # 1. face correction (swap + GFPGAN restore), keeps original audio
        face_out = face_correct.correct_clip_faces(
            clip, identity, out_path + ".face.mp4",
            restore=not args.no_restore, progress=progress)
        print()

        # 2. voice correction (replaces audio with target voice)
        info = video_io.probe(face_out)
        if info["has_audio"]:
            voice_correct.correct_clip_voice(
                face_out, voice_emb, out_path, progress=progress)
            print()
            os.unlink(face_out)
        else:
            os.rename(face_out, out_path)
            print("  (no audio track; voice pass skipped)")
        corrected.append(out_path)
        print(f"  -> {out_path}")

    if args.stitch and len(corrected) > 1:
        stitched = os.path.join(args.output, "stitched.mp4")
        video_io.concat_videos(corrected, stitched)
        print(f"stitched {len(corrected)} clips -> {stitched}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="characterlock",
        description="Enforce face/voice consistency across AI-generated clips.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="register a character from refs + voice")
    pi.add_argument("--refs", required=True,
                    help="folder of reference face images (5-20)")
    pi.add_argument("--voice", required=True,
                    help="reference voice sample (10-30s wav)")
    pi.add_argument("--project", required=True,
                    help="project folder to create")
    pi.set_defaults(func=cmd_init)

    pp = sub.add_parser("process", help="correct clips against a project")
    pp.add_argument("--project", required=True, help="project folder")
    pp.add_argument("--input", required=True, help="folder of raw clips")
    pp.add_argument("--output", required=True, help="folder for corrected clips")
    pp.add_argument("--stitch", action="store_true",
                    help="concatenate corrected clips into one video")
    pp.add_argument("--no-restore", action="store_true",
                    help="skip GFPGAN face restoration (faster, more artifacts)")
    pp.set_defaults(func=cmd_process)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
