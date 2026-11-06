"""Load settings from config.toml into simple typed objects.

Every value has a default here, so a missing key in config.toml never crashes the app.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LENGTHS = ("short", "medium", "long")


@dataclass
class GeneralSettings:
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    keep_awake: bool = True


@dataclass
class LLMSettings:
    host: str = "http://localhost:11434"
    model: str = "qwen3.5:4b"
    temperature: float = 0.7
    num_ctx: int = 8192
    think: bool = False
    chunk_chars: int = 6000
    words_per_part: int = 400
    request_timeout: int = 900
    keep_alive: str = "15m"
    unload_after_script: bool = True


@dataclass
class ScriptSettings:
    default_length: str = "medium"
    show_name: str = "Podcastify"
    host_a_name: str = "Ava"
    host_b_name: str = "Leo"
    words_per_minute: int = 150
    minutes: dict[str, int] = field(
        default_factory=lambda: {"short": 3, "medium": 8, "long": 15}
    )

    def target_words(self, length: str) -> int:
        if length not in self.minutes:
            raise ValueError(f"Unknown length {length!r}. Use one of: {', '.join(LENGTHS)}")
        return self.minutes[length] * self.words_per_minute


@dataclass
class TTSSettings:
    backend: str = "kokoro"
    # Raw per-backend sections, e.g. backends["kokoro"] = {"voice_a": ..., ...}.
    # Each backend reads and validates its own section, so adding a new backend
    # never requires changing this file.
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)

    def options(self, name: str | None = None) -> dict[str, Any]:
        return dict(self.backends.get(name or self.backend, {}))


@dataclass
class ExtractSettings:
    pdf_layout: bool = True
    skip_references: bool = True
    keep_tables: bool = False
    web_timeout: int = 30
    youtube_languages: list[str] = field(default_factory=lambda: ["en"])
    force_asr: bool = False
    max_video_minutes: int = 180


@dataclass
class ASRSettings:
    model: str = "nemo-parakeet-tdt-0.6b-v3"
    quantization: str = "int8"
    provider: str = "cpu"
    # VAD: split long audio at pauses so each piece fits the model (max ~20-30 s).
    vad_max_speech_s: float = 20.0
    vad_min_silence_ms: int = 300


@dataclass
class AudioSettings:
    pause_ms: int = 350
    pause_jitter_ms: int = 120
    speaker_change_extra_ms: int = 100
    lead_in_ms: int = 400
    tail_ms: int = 800
    mp3_bitrate: str = "128k"


@dataclass
class Settings:
    general: GeneralSettings = field(default_factory=GeneralSettings)
    extract: ExtractSettings = field(default_factory=ExtractSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    script: ScriptSettings = field(default_factory=ScriptSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    source_path: Path | None = None


def _pick(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys."""
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def find_config(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if env := os.environ.get("PODCASTIFY_CONFIG"):
        candidates.append(Path(env))
    candidates += [Path.cwd() / "config.toml", PROJECT_ROOT / "config.toml"]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    if explicit:
        raise FileNotFoundError(f"Config file not found: {explicit}")
    return None


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    config_path = find_config(path)
    raw: dict[str, Any] = {}
    if config_path:
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)

    base_dir = config_path.parent if config_path else PROJECT_ROOT

    general = _pick(GeneralSettings, raw.get("general", {}))
    general.output_dir = (base_dir / general.output_dir).resolve()
    general.cache_dir = (base_dir / general.cache_dir).resolve()

    tts_raw = dict(raw.get("tts", {}))
    backend = tts_raw.pop("backend", "kokoro")
    tts = TTSSettings(
        backend=backend,
        backends={k: v for k, v in tts_raw.items() if isinstance(v, dict)},
    )

    script_raw = dict(raw.get("script", {}))
    script = _pick(ScriptSettings, script_raw)
    if script.default_length not in LENGTHS:
        raise ValueError(f"[script] default_length must be one of {LENGTHS}")

    return Settings(
        general=general,
        extract=_pick(ExtractSettings, raw.get("extract", {})),
        asr=_pick(ASRSettings, raw.get("asr", {})),
        llm=_pick(LLMSettings, raw.get("llm", {})),
        script=script,
        tts=tts,
        audio=_pick(AudioSettings, raw.get("audio", {})),
        source_path=config_path,
    )
