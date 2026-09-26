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
    from characterlock.pipeline import project as project_mod
    from characterlock.pipeline import video_io

    proj = project_mod.load_project(args.project)
    clips = video_io.list_clips(args.input)
    if not clips:
        print(f"no video clips found in {args.input}", file=sys.stderr)
        return 1
    print(f"{len(clips)} clip(s) -> {args.output} "
          f"{'(stitched)' if args.stitch else '(separate files)'}")
    # The per-clip face/voice pipeline is wired up in later commits.
    print("pipeline not yet implemented (skeleton).", file=sys.stderr)
    return 2


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
