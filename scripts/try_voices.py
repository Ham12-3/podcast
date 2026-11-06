"""Milestone 1 check: say two lines in two voices and save a WAV.

Run:  uv run python scripts/try_voices.py
      uv run python scripts/try_voices.py --voice-a af_bella --voice-b bm_george
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf

from podcastify.config import load_settings
from podcastify.timing import StageTimer
from podcastify.tts import ScriptLine, create_backend

LINES = [
    ScriptLine("A", "Hi everyone, and welcome to Podcastify. Today we're testing two voices."),
    ScriptLine("B", "That's right. I'm the second host, and I sound different. Pretty neat, huh?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-a")
    parser.add_argument("--voice-b")
    parser.add_argument("--quality", help="Kokoro model quality, e.g. fp32 or q8")
    parser.add_argument("--out", default=None, help="WAV path (default: output/voices_test.wav)")
    args = parser.parse_args()

    settings = load_settings()
    opts = settings.tts.backends.setdefault(settings.tts.backend, {})
    if args.voice_a:
        opts["voice_a"] = args.voice_a
    if args.voice_b:
        opts["voice_b"] = args.voice_b
    if args.quality:
        opts["model_quality"] = args.quality

    out = args.out or str(settings.general.output_dir / "voices_test.wav")
    settings.general.output_dir.mkdir(parents=True, exist_ok=True)

    timer = StageTimer()
    with timer.stage("load model"):
        tts = create_backend(settings.tts)
    print("Backend:", tts.describe())

    with tts:
        with timer.stage("tts"):
            t0 = time.perf_counter()
            clips = tts.synthesize_script(
                LINES, on_progress=lambda i, n: print(f"  line {i}/{n} done")
            )
            synth_secs = time.perf_counter() - t0

    rate = clips[0].sample_rate
    gap = np.zeros(int(rate * settings.audio.pause_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for clip in clips:
        assert clip.sample_rate == rate, "sample rates differ between voices"
        parts += [clip.samples, gap]
    audio = np.concatenate(parts[:-1])
    sf.write(out, audio, rate)

    speech_secs = sum(c.duration for c in clips)
    print(f"\nSaved {out}")
    print(f"  sample rate {rate} Hz, length {len(audio) / rate:.1f}s")
    for line, clip in zip(LINES, clips):
        peak = float(np.max(np.abs(clip.samples))) if clip.samples.size else 0.0
        print(f"  {line.speaker}: {clip.duration:.1f}s, peak {peak:.2f}")
    print(
        f"  speed: {speech_secs:.1f}s of speech in {synth_secs:.1f}s "
        f"({speech_secs / synth_secs:.1f}x real time)"
    )
    print(timer.summary())


if __name__ == "__main__":
    main()
