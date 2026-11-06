"""Stitch spoken lines into one podcast track and export it as MP3 (pydub + ffmpeg).

Per line: trim the silence the TTS adds at the edges, then match loudness so both
hosts sound equally loud. Between lines: a short pause with a little random variation
(plus a touch more when the speaker changes) so it doesn't sound robotic. Finally the
whole track is peak-normalised and exported with ID3 tags.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AudioSettings
from .errors import AudioError
from .tts.base import AudioClip

FFMPEG_HELP = """ffmpeg was not found. It is needed to write MP3 files.

Install it on Windows with ONE of these, then close and reopen your terminal:
  winget install --id Gyan.FFmpeg -e        (built into Windows 10/11)
  choco install ffmpeg                       (if you use Chocolatey)
  scoop install ffmpeg                       (if you use Scoop)

Check it works with:  ffmpeg -version"""

TARGET_RMS_DBFS = -20.0   # per-line loudness target
PEAK_HEADROOM_DB = 1.0    # final peak level = -1 dBFS


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError(FFMPEG_HELP)
    return path


def _db_to_amp(db: float) -> float:
    return float(10 ** (db / 20))


def trim_silence(
    samples: np.ndarray, rate: int, threshold_db: float = -45.0, keep_start_ms: int = 30, keep_end_ms: int = 60
) -> np.ndarray:
    """Cut quiet padding from both ends, keeping a few ms so words aren't clipped."""
    if samples.size == 0:
        return samples
    loud = np.flatnonzero(np.abs(samples) > _db_to_amp(threshold_db))
    if loud.size == 0:
        return samples[:0]
    start = max(0, loud[0] - int(rate * keep_start_ms / 1000))
    end = min(samples.size, loud[-1] + int(rate * keep_end_ms / 1000))
    return samples[start:end]


def match_loudness(samples: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """Scale so the speech (ignoring near-silent parts) hits the target RMS level."""
    if samples.size == 0:
        return samples
    voiced = samples[np.abs(samples) > _db_to_amp(-40.0)]
    if voiced.size < 100:
        return samples
    rms = float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))
    if rms <= 0:
        return samples
    gain = _db_to_amp(target_dbfs) / rms
    out = samples * gain
    # Keep peaks below full scale before the final normalisation pass.
    peak = float(np.max(np.abs(out)))
    return out / peak * 0.98 if peak > 0.98 else out


def _to_segment(samples: np.ndarray, rate: int):
    from pydub import AudioSegment

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    return AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=rate, channels=1)


def stitch(
    clips: Sequence[AudioClip],
    speakers: Sequence[str],
    settings: AudioSettings,
    seed: int = 0,
):
    """Join clips with natural pauses. Returns a pydub AudioSegment.

    `speakers[i]` is the speaker of `clips[i]`. A backend that renders the whole
    dialogue as one clip can pass a single clip; then only lead-in/tail are added.
    """
    from pydub import AudioSegment
    from pydub.effects import normalize

    clips = [c for c in clips if c.samples.size]
    if not clips:
        raise AudioError("There is no audio to stitch (every line came back empty).")
    rate = clips[0].sample_rate
    for c in clips:
        if c.sample_rate != rate:
            raise AudioError(f"Clips have different sample rates ({c.sample_rate} vs {rate}).")

    rng = random.Random(seed)
    silence = lambda ms: AudioSegment.silent(duration=max(0, int(ms)), frame_rate=rate)  # noqa: E731

    parts = [silence(settings.lead_in_ms)]
    single_take = len(clips) == 1
    for i, clip in enumerate(clips):
        samples = np.asarray(clip.samples, dtype=np.float32)
        if not single_take:
            samples = match_loudness(trim_silence(samples, rate))
        parts.append(_to_segment(samples, rate))
        if i < len(clips) - 1:
            pause = settings.pause_ms + rng.uniform(-1, 1) * settings.pause_jitter_ms
            if i + 1 < len(speakers) and speakers[i + 1] != speakers[i]:
                pause += settings.speaker_change_extra_ms
            parts.append(silence(max(120, pause)))
    parts.append(silence(settings.tail_ms))

    # One join instead of repeated `+=` (which re-copies the growing track every time).
    track = AudioSegment(
        data=b"".join(p.raw_data for p in parts), sample_width=2, frame_rate=rate, channels=1
    )
    return normalize(track, headroom=PEAK_HEADROOM_DB)


def export_mp3(track, path: Path, bitrate: str = "128k", tags: dict[str, str] | None = None) -> Path:
    find_ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = track.export(
            str(path),
            format="mp3",
            bitrate=bitrate,
            tags={k: v for k, v in (tags or {}).items() if v},
            id3v2_version="3",  # v2.3 so Windows Explorer shows the title/artist
            parameters=["-ac", "1"],
        )
        handle.close()
    except Exception as exc:  # pydub raises CouldntEncodeError with ffmpeg's stderr
        raise AudioError(f"ffmpeg could not write the MP3: {exc}") from exc
    return path
