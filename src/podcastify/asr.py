"""Local speech-to-text, used for YouTube videos that have no captions.

Uses onnx-asr to run NVIDIA Parakeet TDT 0.6B v3 with onnxruntime, the same engine
Kokoro already uses. Why not NVIDIA's own NeMo toolkit: it needs PyTorch and several GB
of packages and is hard to install on Windows. onnx-asr is pip-only and light on CPU.

Parakeet can only take about 20-30 seconds of audio at once, so the Silero VAD (voice
activity detection) model first splits long audio at natural pauses.

Models are downloaded once from Hugging Face into its normal cache folder
(%USERPROFILE%\\.cache\\huggingface\\hub). Interrupted downloads resume next time.

Change the model in config.toml [asr], e.g. "nemo-parakeet-tdt-0.6b-v2" (English only)
or "onnx-community/whisper-small". With an NVIDIA GPU, set provider = "cuda".
"""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path
from typing import Callable

from .config import ASRSettings
from .errors import ExtractionError
from .system import prepare_gpu
from .timing import fmt_seconds

_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
}
PARAGRAPH_WORDS = 120   # start a new paragraph after about this many words...
PARAGRAPH_GAP_S = 1.5   # ...or after a pause this long


def _providers(name: str, log: Callable[[str], None]) -> list[str]:
    import onnxruntime as ort

    wanted = _PROVIDERS.get(name.lower(), name)
    if wanted == "CPUExecutionProvider":
        return [wanted]
    prepare_gpu(name, log)
    available = ort.get_available_providers()
    if wanted not in available:
        log(f"  Warning: ASR provider {name!r} is not available here ({', '.join(available)}); using CPU")
        return ["CPUExecutionProvider"]
    return [wanted, "CPUExecutionProvider"]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _load(settings: ASRSettings, log: Callable[[str], None]):
    # Quiet two harmless Hugging Face warnings (Windows symlinks, no login token).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    import onnx_asr
    from onnx_asr.utils import ModelLoadingError

    quant = settings.quantization or None
    log(
        f"  Loading speech-to-text model {settings.model} ({quant or 'full precision'}). "
        "The first time, it is downloaded (about 0.7 GB for int8)."
    )
    started = time.perf_counter()
    try:
        model = onnx_asr.load_model(settings.model, quantization=quant, providers=_providers(settings.provider, log))
        vad = onnx_asr.load_vad("silero")
    except ModelLoadingError as exc:
        raise ExtractionError(
            f"Could not load the speech-to-text model {settings.model!r} ({type(exc).__name__}: {exc}). "
            "Check [asr] model and quantization in config.toml."
        ) from exc
    except Exception as exc:  # network problems surface as many different exception types
        raise ExtractionError(
            f"Could not download the speech-to-text model {settings.model!r}: {type(exc).__name__}: {exc}. "
            "Check your internet connection and try again (the download resumes)."
        ) from exc
    log(f"  Speech-to-text model ready in {fmt_seconds(time.perf_counter() - started)}")
    return model.with_vad(
        vad,
        max_speech_duration_s=settings.vad_max_speech_s,
        min_silence_duration_ms=settings.vad_min_silence_ms,
    )


def transcribe(
    wav_path: Path,
    settings: ASRSettings,
    *,
    log: Callable[[str], None] = print,
    progress: Callable[[float, str], None] | None = None,
) -> str:
    """Transcribe a WAV file (16-bit PCM; any sample rate) to plain text with paragraphs."""
    duration = wav_duration(wav_path)
    recognizer = _load(settings, log)

    paragraphs: list[str] = []
    current: list[str] = []
    last_end = 0.0
    started = time.perf_counter()
    for seg in recognizer.recognize(str(wav_path)):
        text = seg.text.strip()
        if text:
            if current and (seg.start - last_end > PARAGRAPH_GAP_S or len(" ".join(current).split()) > PARAGRAPH_WORDS):
                paragraphs.append(" ".join(current))
                current = []
            current.append(text)
            last_end = seg.end
        if progress and duration > 0:
            done = min(1.0, seg.end / duration)
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (1 - done) if done > 0 else 0
            progress(done, f"Transcribing {fmt_seconds(seg.end)} of {fmt_seconds(duration)} (about {fmt_seconds(eta)} left)")
    if current:
        paragraphs.append(" ".join(current))

    elapsed = time.perf_counter() - started
    log(
        f"  Transcribed {fmt_seconds(duration)} of audio in {fmt_seconds(elapsed)} "
        f"({duration / max(elapsed, 1e-6):.1f}x real time)"
    )
    return "\n\n".join(paragraphs)
