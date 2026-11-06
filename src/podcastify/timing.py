"""Tiny stage timer so you can see where time goes on CPU."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


@dataclass
class StageTimer:
    log: Callable[[str], None] = print
    # Optional listener: on_event(stage, "start" | "end" | "failed", seconds)
    on_event: Callable[[str, str, float], None] | None = None
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.log(f"[{name}] started")
        if self.on_event:
            self.on_event(name, "start", 0.0)
        start = time.perf_counter()
        ok = False
        try:
            yield
            ok = True
        finally:
            elapsed = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            self.log(f"[{name}] {'finished' if ok else 'stopped'} after {fmt_seconds(elapsed)}")
            if self.on_event:
                self.on_event(name, "end" if ok else "failed", elapsed)

    def summary(self) -> str:
        if not self.stages:
            return "No stages timed."
        width = max(len(n) for n in self.stages)
        total = sum(self.stages.values())
        lines = ["Timings:"]
        for name, secs in self.stages.items():
            share = (secs / total * 100) if total else 0
            lines.append(f"  {name:<{width}}  {fmt_seconds(secs):>10}  ({share:4.1f}%)")
        lines.append(f"  {'total':<{width}}  {fmt_seconds(total):>10}")
        return "\n".join(lines)
