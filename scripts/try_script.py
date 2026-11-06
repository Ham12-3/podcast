"""Milestone 2 check: turn a text file into a two-host JSON script with Ollama.

Run:  uv run python scripts/try_script.py
      uv run python scripts/try_script.py samples/noise_cancelling.txt --length medium
      uv run python scripts/try_script.py --chunk-chars 800   # force the chunk+summarise path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from podcastify.cache import Cache
from podcastify.config import PROJECT_ROOT, load_settings
from podcastify.errors import PodcastifyError
from podcastify.script import estimated_minutes, save_json, save_text, word_count
from podcastify.script_writer import ScriptWriter
from podcastify.system import keep_awake
from podcastify.timing import StageTimer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", default=str(PROJECT_ROOT / "samples" / "noise_cancelling.txt"))
    parser.add_argument("--length", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--model", help="override [llm] model")
    parser.add_argument("--chunk-chars", type=int, help="override [llm] chunk_chars")
    parser.add_argument("--no-cache", action="store_true", help="don't reuse cached summaries")
    args = parser.parse_args()

    settings = load_settings()
    if args.model:
        settings.llm.model = args.model
    if args.chunk_chars:
        settings.llm.chunk_chars = args.chunk_chars

    text = Path(args.file).read_text(encoding="utf-8")
    print(f"Input: {args.file} ({len(text)} chars, {len(text.split())} words)")

    timer = StageTimer()
    cache = Cache(settings.general.cache_dir, enabled=not args.no_cache)
    writer = ScriptWriter(settings.llm, settings.script, cache=cache)
    last = [-1]

    def progress(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        if pct // 10 != last[0] // 10:
            print(f"  {pct:3d}%  {msg}")
            last[0] = pct

    try:
        with keep_awake(settings.general.keep_awake), timer.stage("script"):
            result = writer.write(text, args.length, progress)
    except PodcastifyError as exc:
        print(f"\nERROR: {exc}")
        return 1

    lines = result.lines
    out_dir = settings.general.output_dir
    stem = f"{Path(args.file).stem}_{args.length}"
    save_json(lines, out_dir / f"{stem}.json")
    save_text(lines, out_dir / f"{stem}.txt", settings.script.host_a_name, settings.script.host_b_name)

    turns_ok = all(a.speaker != b.speaker for a, b in zip(lines, lines[1:]))
    target = settings.script.target_words(args.length)
    print("\n--- check ---")
    print(f"lines: {len(lines)}  (A: {sum(l.speaker == 'A' for l in lines)}, B: {sum(l.speaker == 'B' for l in lines)})")
    print(f"words: {word_count(lines)} of target {target}  -> ~{estimated_minutes(lines, settings.script.words_per_minute):.1f} min")
    print(f"hosts alternate: {turns_ok} | first speaker: {lines[0].speaker} | summarised: {result.summarised}")
    print(f"LLM: {result.stats.summary()}")
    print(f"saved: {out_dir / (stem + '.json')}\n       {out_dir / (stem + '.txt')}")
    print(timer.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
