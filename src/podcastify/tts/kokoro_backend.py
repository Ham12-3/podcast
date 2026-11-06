"""Kokoro-82M text to speech via pykokoro (ONNX Runtime, runs well on CPU).

Config section: [tts.kokoro]
    voice_a, voice_b   Kokoro voice IDs, e.g. "af_heart", "am_michael"
    language           pronunciation language, e.g. "en-us"
    speed              1.0 = normal
    model_variant      "v1.0" has the English voices
    model_quality      "fp32" (best) or "q8" (smaller/faster)
    provider           "cpu" now, "cuda" with an NVIDIA GPU + pykokoro[gpu]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..system import is_gpu_provider, prepare_gpu, silence_native_output
from .base import SPEAKERS, AudioClip, TTSBackend, TTSError


class KokoroBackend(TTSBackend):
    name = "kokoro"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        try:
            import pykokoro
        except ImportError as exc:  # pragma: no cover - install problem
            raise TTSError(
                'pykokoro is not installed. Run: uv add "pykokoro[cpu]"'
            ) from exc

        self._pk = pykokoro
        provider = str(options.get("provider", "cpu"))
        if is_gpu_provider(provider) and not prepare_gpu(provider):
            provider = "cpu"
        self.provider = provider
        self.language: str = options.get("language", "en-us")
        self.voices: dict[str, str] = {
            "A": options.get("voice_a", "af_heart"),
            "B": options.get("voice_b", "am_michael"),
        }
        config = pykokoro.SynthesisConfig(
            voice=self.voices["A"],
            generation=pykokoro.GenerationConfig(
                lang=self.language, speed=float(options.get("speed", 1.0))
            ),
            model_variant=options.get("model_variant", "v1.0"),
            model_quality=options.get("model_quality", "fp32"),
            provider=self.provider,
            # Lines from the LLM are usually short, but this keeps a rare long
            # line from failing: pykokoro splits it at sentence boundaries.
            long_text_split="sentence",
            long_text_use_spacy=False,
            asset_progress=pykokoro.ConsoleAssetProgress(),
        )
        try:
            self._synth = pykokoro.KokoroSynthesizer(config)
        except pykokoro.KokoroError as exc:
            raise TTSError(f"Could not start Kokoro: {exc}") from exc

    def describe(self) -> str:
        o = self.options
        return (
            f"kokoro {o.get('model_variant', 'v1.0')}/{o.get('model_quality', 'fp32')} "
            f"on {self.provider} | A={self.voices['A']} B={self.voices['B']}"
        )

    def synthesize(self, text: str, speaker: str) -> AudioClip:
        if speaker not in SPEAKERS:
            raise TTSError(f"Unknown speaker {speaker!r}; expected one of {SPEAKERS}")
        text = text.strip()
        if not text:
            return AudioClip(np.zeros(0, dtype=np.float32), 24000)
        try:
            with silence_native_output():  # hide eSpeak's phoneme printouts
                rendered = self._synth.synthesize_text(
                    text, language=self.language, voice=self.voices[speaker]
                )
        except self._pk.InvalidVoiceError as exc:
            raise TTSError(
                f"Kokoro voice {self.voices[speaker]!r} is not available. "
                "Check voice_a / voice_b in config.toml."
            ) from exc
        except self._pk.KokoroError as exc:
            raise TTSError(f"Kokoro failed on line: {text[:80]!r}: {exc}") from exc
        audio = np.asarray(rendered.audio, dtype=np.float32).reshape(-1)
        return AudioClip(audio, int(rendered.sample_rate))

    def close(self) -> None:
        synth = getattr(self, "_synth", None)
        if synth is not None:
            synth.close()
            self._synth = None
