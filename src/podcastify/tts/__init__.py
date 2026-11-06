"""Speech backends. Pick one by name with `create_backend(settings)`.

To add a backend: write `tts/<name>_backend.py`, then add one line to `_REGISTRY`.
Imports are lazy so a GPU-only backend never loads on a CPU machine unless selected.
"""

from __future__ import annotations

import importlib

from ..config import TTSSettings
from .base import SPEAKERS, AudioClip, ScriptLine, TTSBackend, TTSError

# name -> "module:ClassName" (relative to this package)
_REGISTRY: dict[str, str] = {
    "kokoro": ".kokoro_backend:KokoroBackend",
    # "vibevoice": ".vibevoice_backend:VibeVoiceBackend",
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_backend(settings: TTSSettings, name: str | None = None) -> TTSBackend:
    name = name or settings.backend
    if name not in _REGISTRY:
        raise TTSError(
            f"Unknown TTS backend {name!r}. Available: {', '.join(available_backends())}"
        )
    module_name, class_name = _REGISTRY[name].split(":")
    module = importlib.import_module(module_name, package=__name__)
    cls = getattr(module, class_name)
    return cls(settings.options(name))


__all__ = [
    "SPEAKERS",
    "AudioClip",
    "ScriptLine",
    "TTSBackend",
    "TTSError",
    "available_backends",
    "create_backend",
]
