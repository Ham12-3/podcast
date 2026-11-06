"""The full Podcastify pipeline, shared by the CLI and the Gradio UI.

    source -> [extraction] -> text -> [script] -> lines -> [tts] -> clips -> [stitching] -> MP3

Extracted text and finished scripts are cached under cache_dir, so running the same
input again (for example with different voices or pauses) skips straight to audio.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import script as script_io
from .audio import export_mp3, find_ffmpeg, stitch
from .cache import Cache, make_key
from .config import LENGTHS, Settings
from .errors import PodcastifyError
from .extractors import Extracted, ExtractOptions, detect_type, extract, is_url
from .script import ScriptLine
from .script_writer import PROMPT_VERSION, ScriptWriter
from .system import keep_awake
from .textutil import slugify
from .timing import StageTimer, fmt_seconds
from .tts import create_backend

# Overall progress: (fraction 0..1, message)
ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]
# Per-stage events for UIs: (stage, event, value, message) where event is
# "start", "progress" (value = 0..1 within the stage), "end" or "failed" (value = seconds).
StageFn = Callable[[str, str, float, str], None]

# Rough share of total time per stage on CPU, used for the overall progress bar.
_STAGE_SPAN = {
    "extraction": (0.00, 0.05),
    "script": (0.05, 0.60),
    "tts": (0.60, 0.97),
    "stitching": (0.97, 1.00),
}


@dataclass
class PipelineResult:
    title: str
    lines: list[ScriptLine]
    script_json: Path
    script_txt: Path
    mp3_path: Path | None = None
    audio_seconds: float = 0.0
    script_from_cache: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    timing_summary: str = ""


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        log: LogFn = print,
        use_cache: bool = True,
        on_stage: StageFn | None = None,
    ):
        self.settings = settings
        self.log = log
        self.on_stage = on_stage
        self.cache = Cache(settings.general.cache_dir, enabled=use_cache)
        self.names = {"A": settings.script.host_a_name, "B": settings.script.host_b_name}
        self._spans = _STAGE_SPAN

    # ------------------------------------------------------------------ helpers
    def _timer(self) -> StageTimer:
        on_event = None
        if self.on_stage:
            on_event = lambda name, event, secs: self.on_stage(name, event, secs, "")  # noqa: E731
        return StageTimer(log=self.log, on_event=on_event)

    def _stage_progress(self, progress: ProgressFn, stage: str) -> ProgressFn:
        lo, hi = self._spans[stage]

        def report(f: float, msg: str) -> None:
            f = max(0.0, min(1.0, f))
            if self.on_stage:
                self.on_stage(stage, "progress", f, msg)
            progress(lo + (hi - lo) * f, msg)

        return report

    def _source_key(self, source: str) -> str:
        # Extraction settings are part of the key: changing pdf_layout or force_asr re-extracts.
        ex, asr = self.settings.extract, self.settings.asr
        settings_part = (sorted(asdict(ex).items()), asr.model, asr.quantization)
        source = source.strip().strip('"')
        if is_url(source):
            if detect_type(source) == "youtube":
                from .extractors.youtube import parse_video_id

                return make_key("youtube", parse_video_id(source), settings_part)
            return make_key("url", source, settings_part)
        path = Path(source).resolve()
        stat = path.stat()
        return make_key("file", str(path).lower(), stat.st_size, stat.st_mtime_ns, settings_part)

    def _cached_extraction(self, source: str) -> Extracted | None:
        detect_type(source)  # validates the input (file exists, type supported...)
        data = self.cache.get_json("text", self._source_key(source))
        return Extracted.from_dict(data) if data else None

    def _script_key(self, text: str, length: str) -> str:
        llm, sc = self.settings.llm, self.settings.script
        return make_key(
            "script", PROMPT_VERSION, llm.model, llm.temperature, llm.think, llm.chunk_chars,
            llm.words_per_part, length, sc.target_words(length), sc.show_name,
            sc.host_a_name, sc.host_b_name, hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def _output_paths(self, title: str, suffix: str, out_dir: Path | None) -> dict[str, Path]:
        folder = Path(out_dir) if out_dir else self.settings.general.output_dir
        stem = f"{slugify(title)}_{suffix}"
        return {ext: folder / f"{stem}.{ext}" for ext in ("mp3", "txt", "json")}

    # --------------------------------------------------------------------- run
    def run(
        self,
        source: str,
        length: str | None = None,
        *,
        progress: ProgressFn | None = None,
        new_script: bool = False,
        script_only: bool = False,
        out_dir: Path | None = None,
    ) -> PipelineResult:
        """Full run: input -> MP3 (or just the script with script_only=True)."""
        progress = progress or (lambda f, m: None)
        length = length or self.settings.script.default_length
        if length not in LENGTHS:
            raise PodcastifyError(f"Length must be one of: {', '.join(LENGTHS)}")

        timer = self._timer()
        self._spans = _STAGE_SPAN
        writer = ScriptWriter(self.settings.llm, self.settings.script, cache=self.cache, log=self.log)

        with keep_awake(self.settings.general.keep_awake):
            if not script_only:
                find_ffmpeg()  # fail now, not after 10 minutes of LLM work

            # ---- extraction (cached) ----
            extracted = self._cached_extraction(source)
            script_cached = bool(
                extracted
                and not new_script
                and self.cache.get_json("scripts", self._script_key(extracted.text, length))
            )
            if not script_cached:
                writer.check_ready()  # Ollama problems surface before slow extraction

            with timer.stage("extraction"):
                extracted = self._extract(source, self._stage_progress(progress, "extraction"), extracted)

            # ---- script (cached) ----
            with timer.stage("script"):
                prog = self._stage_progress(progress, "script")
                key = self._script_key(extracted.text, length)
                cached = None if new_script else self.cache.get_json("scripts", key)
                if cached:
                    lines = script_io.from_json_list(cached["lines"])
                    self.log(f"  Using cached script ({len(lines)} lines) - pass --new-script to rewrite it")
                    prog(1.0, "Script loaded from cache")
                else:
                    result = writer.write(extracted.text, length, prog)
                    lines = result.lines
                    self.log(f"  {result.stats.summary()}")
                    if not script_only and self.settings.llm.unload_after_script:
                        writer.unload()
                    self.cache.put_json(
                        "scripts", key,
                        {"title": extracted.title, "length": length, "model": self.settings.llm.model,
                         "source": extracted.source, "lines": script_io.to_json_list(lines)},
                    )

            paths = self._output_paths(extracted.title, length, out_dir)
            self._save_script(lines, paths, extracted.title)
            self.log(
                f"  Script: {len(lines)} lines, {script_io.word_count(lines)} words "
                f"(~{script_io.estimated_minutes(lines, self.settings.script.words_per_minute):.1f} min)"
            )

            result = PipelineResult(
                title=extracted.title, lines=lines, script_json=paths["json"],
                script_txt=paths["txt"], script_from_cache=bool(cached),
            )
            if not script_only:
                self._render_audio(lines, extracted.title, extracted.source, paths["mp3"], timer, progress, result)

        progress(1.0, "Done")
        result.timings = dict(timer.stages)
        result.timing_summary = timer.summary()
        self.log(result.timing_summary)
        return result

    def extract_only(
        self,
        source: str,
        *,
        progress: ProgressFn | None = None,
        out_dir: Path | None = None,
    ) -> tuple[Extracted, Path]:
        """Just extract the text and save it as <title>_source.txt (handy for checking inputs)."""
        progress = progress or (lambda f, m: None)
        timer = self._timer()
        self._spans = {"extraction": (0.0, 1.0)}
        with keep_awake(self.settings.general.keep_awake):
            cached = self._cached_extraction(source)
            with timer.stage("extraction"):
                extracted = self._extract(source, self._stage_progress(progress, "extraction"), cached)
        folder = Path(out_dir) if out_dir else self.settings.general.output_dir
        path = folder / f"{slugify(extracted.title)}_source.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{extracted.title}\n\n{extracted.text}\n", encoding="utf-8")
        return extracted, path

    def run_from_script(
        self,
        script_path: str | Path,
        *,
        progress: ProgressFn | None = None,
        out_dir: Path | None = None,
        title: str | None = None,
    ) -> PipelineResult:
        """Skip extraction and the LLM: render audio from a saved (maybe hand-edited) script."""
        progress = progress or (lambda f, m: None)
        path = Path(script_path)
        if not path.is_file():
            raise PodcastifyError(f"Script file not found: {path}")
        if path.suffix.lower() == ".json":
            lines = script_io.load_json(path)
        else:
            lines = script_io.load_text(path, self.names)
        if not lines:
            raise PodcastifyError(f"No dialogue lines found in {path}")

        title = title or path.stem
        timer = self._timer()
        self._spans = {"tts": (0.0, 0.95), "stitching": (0.95, 1.0)}
        mp3 = (Path(out_dir) if out_dir else path.parent) / f"{path.stem}.mp3"
        result = PipelineResult(title=title, lines=lines, script_json=path, script_txt=path)
        with keep_awake(self.settings.general.keep_awake):
            find_ffmpeg()
            self._render_audio(lines, title, str(path), mp3, timer, progress, result)
        progress(1.0, "Done")
        result.timings = dict(timer.stages)
        result.timing_summary = timer.summary()
        self.log(result.timing_summary)
        return result

    # ----------------------------------------------------------------- stages
    def _extract(self, source: str, prog: ProgressFn, cached: Extracted | None) -> Extracted:
        if cached:
            self.log(f"  Using cached text for {source}")
            extracted = cached
        else:
            prog(0.0, "Extracting text")
            extracted = extract(source, ExtractOptions(
                settings=self.settings.extract,
                asr=self.settings.asr,
                work_dir=self.settings.general.cache_dir / "downloads",
                log=self.log,
                progress=prog,
            ))
            self.cache.put_json("text", self._source_key(source), extracted.to_dict())
        words = len(extracted.text.split())
        self.log(f"  Title: {extracted.title!r} | {extracted.source_type} | {words} words")
        prog(1.0, "Text ready")
        return extracted

    def _save_script(self, lines: list[ScriptLine], paths: dict[str, Path], title: str) -> None:
        script_io.save_json(lines, paths["json"])
        script_io.save_text(lines, paths["txt"], self.names["A"], self.names["B"], title=title)

    def _render_audio(
        self,
        lines: list[ScriptLine],
        title: str,
        source: str,
        mp3_path: Path,
        timer: StageTimer,
        progress: ProgressFn,
        result: PipelineResult,
    ) -> None:
        with timer.stage("tts"):
            prog = self._stage_progress(progress, "tts")
            prog(0.0, "Loading the voice model")
            with create_backend(self.settings.tts) as tts:
                self.log(f"  TTS: {tts.describe()}")
                started = time.perf_counter()

                def on_line(done: int, total: int) -> None:
                    elapsed = time.perf_counter() - started
                    eta = elapsed / done * (total - done)
                    msg = f"Speaking line {done}/{total} (about {fmt_seconds(eta)} left)"
                    prog(done / total, msg)
                    if done == total or done % max(1, total // 10) == 0:
                        self.log(f"    {msg}")

                clips = tts.synthesize_script(lines, on_progress=on_line)

        with timer.stage("stitching"):
            prog = self._stage_progress(progress, "stitching")
            prog(0.0, "Stitching audio")
            seed = int(hashlib.sha256("".join(l.text for l in lines).encode()).hexdigest()[:8], 16)
            track = stitch(clips, [l.speaker for l in lines], self.settings.audio, seed=seed)
            show = self.settings.script.show_name
            export_mp3(
                track,
                mp3_path,
                bitrate=self.settings.audio.mp3_bitrate,
                tags={
                    "title": title,
                    "artist": show,
                    "album": show,
                    "date": str(_dt.date.today().year),
                    "comment": f"Made locally with Podcastify from {source}",
                },
            )
            result.mp3_path = mp3_path
            result.audio_seconds = len(track) / 1000.0
            self.log(f"  Saved {mp3_path} ({fmt_seconds(result.audio_seconds)} of audio)")
            prog(1.0, "MP3 ready")
