"""The podcast script data type plus helpers to save/load it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SPEAKERS = ("A", "B")


@dataclass
class ScriptLine:
    speaker: str  # "A" or "B"
    text: str


def to_json_list(lines: Iterable[ScriptLine]) -> list[dict[str, str]]:
    return [asdict(line) for line in lines]


def from_json_list(data: list[dict[str, str]]) -> list[ScriptLine]:
    return [ScriptLine(str(d["speaker"]), str(d["text"])) for d in data]


def save_json(lines: list[ScriptLine], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_list(lines), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_json(path: Path) -> list[ScriptLine]:
    return from_json_list(json.loads(path.read_text(encoding="utf-8")))


def load_text(path: Path, names: dict[str, str]) -> list[ScriptLine]:
    """Read a readable script ("Ava: Hello...") back, e.g. after editing it by hand."""
    lookup = {"a": "A", "b": "B", "host a": "A", "host b": "B"}
    lookup.update({name.lower(): spk for spk, name in names.items()})
    lines: list[ScriptLine] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw:
            continue
        label, text = raw.split(":", 1)
        speaker = lookup.get(label.strip().lower())
        if speaker and text.strip():
            lines.append(ScriptLine(speaker, text.strip()))
    return lines


def word_count(lines: Iterable[ScriptLine]) -> int:
    return sum(len(line.text.split()) for line in lines)


def estimated_minutes(lines: Iterable[ScriptLine], words_per_minute: int = 150) -> float:
    return word_count(lines) / float(words_per_minute)


def to_readable_text(
    lines: list[ScriptLine], host_a: str = "Host A", host_b: str = "Host B", title: str = ""
) -> str:
    names = {"A": host_a, "B": host_b}
    out = []
    if title:
        out += [title, "=" * len(title), ""]
    for line in lines:
        out.append(f"{names.get(line.speaker, line.speaker)}: {line.text}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def save_text(lines: list[ScriptLine], path: Path, host_a: str, host_b: str, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_readable_text(lines, host_a, host_b, title), encoding="utf-8")
