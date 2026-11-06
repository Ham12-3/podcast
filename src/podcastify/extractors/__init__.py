"""Turn an input (file path or URL) into plain text.

Every extractor returns an `Extracted` with plain text plus a title. `extract()` picks
the right one from the input. Extractor modules are imported lazily so a missing
optional dependency only matters when that input type is used.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from ..config import ASRSettings, ExtractSettings
from ..errors import ExtractionError

MIN_WORDS = 30  # less than this is not enough to talk about

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".rst"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


@dataclass
class Extracted:
    text: str
    title: str
    source_type: str  # "text" | "pdf" | "web" | "youtube"
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Extracted":
        return cls(d["text"], d["title"], d["source_type"], d["source"])


def _no_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class ExtractOptions:
    """Everything an extractor may need besides the source itself."""

    settings: ExtractSettings = field(default_factory=ExtractSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    work_dir: Path | None = None  # scratch space, e.g. for downloaded audio
    log: Callable[[str], None] = print
    progress: Callable[[float, str], None] = _no_progress  # (0..1, message)


# source_type -> "module:function" (relative to this package)
_EXTRACTORS = {
    "text": ".textfile:extract_text_file",
    "pdf": ".pdf:extract_pdf",
    "web": ".web:extract_web",
    "youtube": ".youtube:extract_youtube",
}


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source.strip(), re.I))


def detect_type(source: str) -> str:
    source = source.strip().strip('"')
    if is_url(source):
        host = (urlparse(source).hostname or "").lower()
        return "youtube" if host in YOUTUBE_HOSTS else "web"
    if source.lower().startswith(("www.", "youtube.com", "youtu.be")):
        raise ExtractionError(f"That looks like a web address. Add https:// in front: https://{source}")
    path = Path(source)
    if not path.exists():
        raise ExtractionError(f"File not found: {source}")
    if path.is_dir():
        raise ExtractionError(f"That is a folder, not a file: {source}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in TEXT_SUFFIXES:
        return "text"
    raise ExtractionError(
        f"Unsupported file type {suffix or '(none)'!r}. Use a PDF, a .txt/.md file, a blog URL or a YouTube URL."
    )


def extract(source: str, opts: ExtractOptions | None = None) -> Extracted:
    """Pick the right extractor for `source` and return its plain text."""
    source = source.strip().strip('"')
    opts = opts or ExtractOptions()
    kind = detect_type(source)
    module_name, func_name = _EXTRACTORS[kind].split(":")
    try:
        module = importlib.import_module(module_name, package=__name__)
    except ModuleNotFoundError as exc:
        raise ExtractionError(
            f"Support for {kind} inputs needs the '{exc.name}' package. Run: uv sync"
        ) from exc
    result: Extracted = getattr(module, func_name)(source, opts)
    words = len(result.text.split())
    if words < MIN_WORDS:
        raise ExtractionError(
            f"Only {words} words of text were found in {source}. That's not enough for a podcast."
        )
    return result


__all__ = ["ExtractOptions", "Extracted", "ExtractionError", "detect_type", "extract", "is_url"]
