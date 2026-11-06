"""Plain text / Markdown files."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import ExtractionError
from ..textutil import normalize_text
from . import Extracted, ExtractOptions


def _read(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(encoding)
            if encoding == "utf-16" and "\x00" in text:
                continue
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # code blocks
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # links -> link text
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)    # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)         # bullets
    return re.sub(r"[*_`]{1,3}", "", text)                       # emphasis


def extract_text_file(source: str, opts: ExtractOptions | None = None) -> Extracted:
    path = Path(source)
    text = _read(path)
    if path.suffix.lower() in {".md", ".markdown"}:
        text = _strip_markdown(text)
    text = normalize_text(text)
    if not text:
        raise ExtractionError(f"The file is empty: {source}")

    first_line = text.split("\n", 1)[0].strip()
    title = first_line if 0 < len(first_line) <= 100 else path.stem.replace("_", " ")
    return Extracted(text=text, title=title, source_type="text", source=str(path.resolve()))
