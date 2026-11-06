"""The TTS interface every speech backend implements.

The rest of the app only talks to `TTSBackend`. To add a new engine (for example a GPU
VibeVoice backend), create `tts/<name>_backend.py` with a subclass, register it in
`tts/__init__.py`, add a `[tts.<name>]` section to config.toml and set `[tts] backend`.
Nothing else in the app needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

__all__ = ["SPEAKERS", "AudioClip", "ProgressFn", "ScriptLine", "TTSBackend", "TTSError"]

import numpy as np

from ..errors import PodcastifyError
from ..script import SPEAKERS, ScriptLine

# Called after each line: (lines_done, lines_total)
ProgressFn = Callable[[int, int], None]


class TTSError(PodcastifyError):
    """A speech backend failed in a way the user can act on."""


@dataclass
class AudioClip:
    """Mono audio as float32 samples in the range -1..1."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0


class TTSBackend(ABC):
    """Base class for speech backends.

    Subclasses get their own config section as `options` (a plain dict) and must map
    the speakers "A" and "B" to voices.
    """

    name: str = "base"

    def __init__(self, options: dict[str, Any]):
        self.options = options

    # ---- required ---------------------------------------------------------------
    @abstractmethod
    def synthesize(self, text: str, speaker: str) -> AudioClip:
        """Render one line of speech for speaker "A" or "B"."""

    # ---- optional overrides -----------------------------------------------------
    def synthesize_script(
        self, lines: Sequence[ScriptLine], on_progress: ProgressFn | None = None
    ) -> list[AudioClip]:
        """Render a whole script, one clip per line.

        Backends that can render a full multi-speaker dialogue in one pass (such as
        VibeVoice) may override this and still return one clip per line, or a single
        clip; `audio.stitch` accepts either.
        """
        clips: list[AudioClip] = []
        total = len(lines)
        for i, line in enumerate(lines, start=1):
            clips.append(self.synthesize(line.text, line.speaker))
            if on_progress:
                on_progress(i, total)
        return clips

    def describe(self) -> str:
        return self.name

    def close(self) -> None:
        """Release models / sessions."""

    def __enter__(self) -> "TTSBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
