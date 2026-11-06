"""Small file cache so expensive steps (extraction, LLM calls) are not repeated.

Layout: <cache_dir>/<kind>/<key>.<ext>, where key is a hash of everything that
affects the result (input text, model name, prompt version, settings...).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


class Cache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def path(self, kind: str, key: str, ext: str) -> Path:
        return self.root / kind / f"{key}.{ext}"

    def get_text(self, kind: str, key: str) -> str | None:
        if not self.enabled:
            return None
        p = self.path(kind, key, "txt")
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def put_text(self, kind: str, key: str, text: str) -> Path:
        p = self.path(kind, key, "txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def get_json(self, kind: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self.path(kind, key, "json")
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put_json(self, kind: str, key: str, data: Any) -> Path:
        p = self.path(kind, key, "json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p
