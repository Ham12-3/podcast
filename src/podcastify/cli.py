"""Command line entry point.

Examples:
  uv run podcastify "path/to/file.pdf" --length medium
  uv run podcastify https://example.com/blog-post --length short
  uv run podcastify "https://www.youtube.com/watch?v=..." --length long
  uv run podcastify notes.txt --script-only            # write + save the script, no audio
  uv run podcastify paper.pdf --extract-only           # just check the extracted text
  uv run podcastify "https://youtu.be/..." --force-asr # transcribe locally even if captions exist
  uv run podcastify --from-script output/my-topic_medium.txt   # audio from an edited script
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import LENGTHS, load_settings
from .errors import PodcastifyError
from .pipeline import Pipeline
from .timing import fmt_seconds


class ConsoleProgress:
    """Print progress only when the message changes or another 10% is done."""

    def __init__(self) -> None:
        self.last_msg = ""
        self.last_bucket = -1

    def __call__(self, fraction: float, message: str) -> None:
        bucket = int(fraction * 10)
        stem = message.split(" (")[0]  # ignore changing ETA text when deciding to print
        if stem.startswith("Speaking line"):
            return  # the pipeline already logs TTS progress every 10%
        if stem != self.last_msg or bucket != self.last_bucket:
            print(f"  [{int(fraction * 100):3d}%] {message}", flush=True)
            self.last_msg, self.last_bucket = stem, bucket


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="podcastify",
        description="Turn a PDF, blog post, YouTube video or text file into a two-host podcast MP3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1] if __doc__ else None,
    )
    p.add_argument("source", nargs="?", help="PDF/.txt/.md path, blog URL or YouTube URL")
    p.add_argument("-l", "--length", choices=LENGTHS, help="episode length (default from config.toml)")
    p.add_argument("-o", "--out", type=Path, help="output folder (default from config.toml)")
    p.add_argument("-c", "--config", help="path to a config.toml")
    p.add_argument("--model", help="Ollama model, e.g. qwen3.5:4b")
    p.add_argument("--voice-a", help="voice for Host A (TTS backend specific)")
    p.add_argument("--voice-b", help="voice for Host B (TTS backend specific)")
    p.add_argument("--tts", help="TTS backend name (default from config.toml)")
    p.add_argument("--new-script", action="store_true", help="rewrite the script even if one is cached")
    p.add_argument("--no-cache", action="store_true", help="ignore all cached text, summaries and scripts")
    p.add_argument("--script-only", action="store_true", help="stop after writing the script")
    p.add_argument("--extract-only", action="store_true", help="only extract the text and save it as <title>_source.txt")
    p.add_argument("--force-asr", action="store_true", help="YouTube: ignore captions and transcribe the audio locally")
    p.add_argument("--from-script", metavar="FILE", help="make audio from a saved .json or .txt script")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not be UTF-8; never crash on an unusual character.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    if not args.source and not args.from_script:
        build_parser().print_help()
        return 2

    try:
        settings = load_settings(args.config)
    except (OSError, ValueError) as exc:
        print(f"Error in config: {exc}", file=sys.stderr)
        return 2

    if args.model:
        settings.llm.model = args.model
    if args.force_asr:
        settings.extract.force_asr = True
    if args.tts:
        settings.tts.backend = args.tts
    backend_opts = settings.tts.backends.setdefault(settings.tts.backend, {})
    if args.voice_a:
        backend_opts["voice_a"] = args.voice_a
    if args.voice_b:
        backend_opts["voice_b"] = args.voice_b

    pipeline = Pipeline(settings, log=lambda m: print(m, flush=True), use_cache=not args.no_cache)
    progress = ConsoleProgress()
    started = time.perf_counter()
    try:
        if args.extract_only:
            extracted, path = pipeline.extract_only(args.source, progress=progress, out_dir=args.out)
            print(f"\nDone in {fmt_seconds(time.perf_counter() - started)}.")
            print(f"  Text:   {path}  ({len(extracted.text.split())} words)")
            return 0
        if args.from_script:
            result = pipeline.run_from_script(args.from_script, progress=progress, out_dir=args.out)
        else:
            result = pipeline.run(
                args.source,
                args.length,
                progress=progress,
                new_script=args.new_script or args.no_cache,
                script_only=args.script_only,
                out_dir=args.out,
            )
    except PodcastifyError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130

    print(f"\nDone in {fmt_seconds(time.perf_counter() - started)}.")
    if result.mp3_path:
        print(f"  MP3:    {result.mp3_path}")
    print(f"  Script: {result.script_txt}")
    if result.script_json != result.script_txt:
        print(f"  JSON:   {result.script_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
