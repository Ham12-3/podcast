"""Podcastify web UI (Gradio), running only on this computer.

Start it with:
    uv run podcastify-ui            (opens http://127.0.0.1:7860 in your browser)
    uv run podcastify-ui --port 8000 --no-browser

The pipeline runs in a background thread while the page shows live progress for each
stage. Stop asks the pipeline to finish at the next progress update (usually within a
few seconds; while the LLM reads a long prompt it can take up to a minute).
"""

from __future__ import annotations

import argparse
import copy
import html
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from .config import LENGTHS, PROJECT_ROOT, Settings, load_settings
from .errors import PodcastifyError
from .pipeline import Pipeline, PipelineResult
from .timing import fmt_seconds

STAGES = [
    ("extraction", "Reading the input"),
    ("script", "Writing the script"),
    ("tts", "Recording the voices"),
    ("stitching", "Making the MP3"),
]
ICONS = {"waiting": "○", "running": "◐", "done": "✓", "failed": "✕"}
LOG_LINES = 400  # keep the log box light


class Cancelled(PodcastifyError):
    pass


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


# ------------------------------------------------------------------ job state
@dataclass
class StageState:
    status: str = "waiting"  # waiting | running | done | failed
    fraction: float = 0.0
    message: str = ""
    seconds: float = 0.0
    started: float = 0.0


@dataclass
class Job:
    """One generation, run in a worker thread. The page polls this object."""

    settings: Settings
    source: str
    length: str
    new_script: bool
    uploaded: bool = False
    stages: dict[str, StageState] = field(default_factory=lambda: {n: StageState() for n, _ in STAGES})
    log_lines: list[str] = field(default_factory=list)
    overall: float = 0.0
    result: PipelineResult | None = None
    error: str | None = None
    cancelled: bool = False
    started: float = field(default_factory=time.perf_counter)
    finished: float = 0.0
    version: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        threading.Thread(target=self._run, name="podcastify-job", daemon=True).start()

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.lock:
            self.log_lines.append(message)
            del self.log_lines[:-LOG_LINES]
            self.version += 1

    def on_stage(self, stage: str, event: str, value: float, message: str) -> None:
        if event in ("start", "progress") and self.cancel.is_set():
            raise Cancelled("Stopped")
        with self.lock:
            state = self.stages.get(stage)
            if state is None:
                return
            if event == "start":
                state.status, state.fraction, state.started = "running", 0.0, time.perf_counter()
            elif event == "progress":
                state.fraction = value
                if message:
                    state.message = message
            elif event == "end":
                state.status, state.fraction, state.seconds = "done", 1.0, value
            elif event == "failed":
                state.status, state.seconds = "failed", value
            self.version += 1

    def progress(self, fraction: float, message: str) -> None:
        if self.cancel.is_set() and fraction < 0.999:
            raise Cancelled("Stopped")
        self.overall = fraction

    def _run(self) -> None:
        try:
            pipeline = Pipeline(self.settings, log=self.log, on_stage=self.on_stage)
            self.result = pipeline.run(
                self.source, self.length, progress=self.progress, new_script=self.new_script
            )
        except Cancelled:
            self.cancelled = True
            self.error = "Stopped. Anything already finished (text, summaries, script) is cached, so a rerun continues from there."
            self.log("Stopped by the user.")
        except PodcastifyError as exc:
            message = str(exc)
            if self.uploaded:  # show "paper.pdf", not Gradio's long temporary upload path
                message = message.replace(self.source, Path(self.source).name)
            self.error = message
            self.log(f"Error: {message}")
        except Exception as exc:  # show unexpected bugs instead of hanging the page
            self.error = f"Unexpected error: {type(exc).__name__}: {exc}"
            self.log(traceback.format_exc())
        finally:
            self.finished = time.perf_counter()
            self.done.set()


_current: Job | None = None  # the running job (this is a single-user local app)
_current_lock = threading.Lock()


# ------------------------------------------------------------------ rendering
CSS = """
.pc-status { border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);
             padding: 14px 16px; background: var(--background-fill-secondary); }
.pc-stage { display: grid; grid-template-columns: 22px minmax(120px, 190px) 1fr 52px;
            align-items: center; gap: 4px 10px; padding: 6px 0; }
.pc-icon { font-size: 15px; text-align: center; color: var(--body-text-color-subdued); }
.pc-label { font-weight: 600; }
.pc-time { text-align: right; font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued); }
.pc-bar { height: 8px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; }
.pc-bar > div { height: 100%; background: var(--color-accent); transition: width .4s; }
.pc-msg { grid-column: 2 / 5; font-size: 13px; color: var(--body-text-color-subdued); min-height: 0; }
.pc-msg:empty { display: none; }
.pc-done .pc-icon { color: #16a34a; }
.pc-running .pc-icon { color: var(--color-accent); }
.pc-failed .pc-icon, .pc-failed .pc-label { color: #dc2626; }
.pc-waiting .pc-label { color: var(--body-text-color-subdued); font-weight: 500; }
.pc-overall { margin-top: 8px; padding-top: 10px; border-top: 1px solid var(--border-color-primary);
              display: flex; justify-content: space-between; font-size: 14px; }
.pc-banner { margin-top: 10px; padding: 10px 12px; border-radius: var(--radius-md); font-size: 14px; }
.pc-banner.ok { background: rgba(22,163,74,.12); }
.pc-banner.err { background: rgba(220,38,38,.12); }
.pc-banner.warn { background: rgba(234,179,8,.15); }
.pc-checks { font-size: 14px; }
"""


def render_status(job: Job | None) -> str:
    rows = []
    now = time.perf_counter()
    for name, label in STAGES:
        s = job.stages[name] if job else StageState()
        secs = (now - s.started) if s.status == "running" else s.seconds
        msg = s.message if s.status == "running" else ("stopped here" if s.status == "failed" else "")
        rows.append(
            f'<div class="pc-stage pc-{s.status}">'
            f'<span class="pc-icon">{ICONS[s.status]}</span>'
            f'<span class="pc-label">{label}</span>'
            f'<div class="pc-bar"><div style="width:{s.fraction * 100:.0f}%"></div></div>'
            f'<span class="pc-time">{_clock(secs) if s.status != "waiting" else ""}</span>'
            f'<div class="pc-msg">{html.escape(msg)}</div></div>'
        )
    if job is None:
        footer = '<div class="pc-overall"><span>Ready</span><span></span></div>'
        banner = ""
    else:
        end = job.finished if job.done.is_set() else now
        state = "Finished" if job.result else ("Stopped" if job.cancelled else ("Failed" if job.error else "Working"))
        footer = (
            f'<div class="pc-overall"><span>{state} · {job.overall * 100:.0f}% overall</span>'
            f"<span>total {_clock(end - job.started)}</span></div>"
        )
        banner = ""
        if job.result and job.result.mp3_path:
            banner = (
                f'<div class="pc-banner ok">Your episode is ready: {fmt_seconds(job.result.audio_seconds)} of audio. '
                f"Saved to <code>{html.escape(str(job.result.mp3_path))}</code></div>"
            )
        elif job.error:
            kind = "warn" if job.cancelled else "err"
            banner = f'<div class="pc-banner {kind}">{html.escape(job.error)}</div>'
    return f'<div class="pc-status">{"".join(rows)}{footer}{banner}</div>'


def render_log(job: Job) -> str:
    with job.lock:
        return "\n".join(job.log_lines)


# ------------------------------------------------------------------ handlers
def make_handlers(base: Settings):
    def check_system() -> str:
        from .audio import find_ffmpeg
        from .script_writer import ScriptWriter

        lines = []
        try:
            ScriptWriter(base.llm, base.script, log=lambda m: None).check_ready()
            lines.append(f"✅ Ollama is running and **{base.llm.model}** is ready")
        except PodcastifyError as exc:
            lines.append(f"⚠️ {exc}")
        try:
            find_ffmpeg()
            lines.append("✅ ffmpeg found")
        except PodcastifyError:
            lines.append("⚠️ ffmpeg is missing. In a terminal run `winget install --id Gyan.FFmpeg -e`, then restart this app.")
        return " &nbsp;·&nbsp; ".join(lines)

    # The Generate/Stop buttons are outputs of `generate` itself: a `.then()` step would
    # not run after an error, leaving the Generate button disabled.
    def busy():
        return gr.Button(interactive=False), gr.Button(interactive=True)

    def idle():
        return gr.Button(interactive=True), gr.Button(interactive=False)

    skip = lambda n: tuple(gr.skip() for _ in range(n))  # noqa: E731

    def generate(source_text: str, upload: str | None, length: str, new_script: bool, force_asr: bool):
        global _current
        source = (upload or "").strip() or (source_text or "").strip().strip('"')
        if not source:
            gr.Warning("Paste a blog or YouTube link (or a file path), or upload a PDF / text file.", title="Nothing to do")
            yield *skip(6), *idle()
            return
        settings = copy.deepcopy(base)
        settings.extract.force_asr = bool(force_asr)
        job = Job(
            settings=settings, source=source, length=length or base.script.default_length,
            new_script=bool(new_script), uploaded=bool(upload),
        )
        with _current_lock:
            _current = job
        if upload and source_text.strip():
            job.log("Both a link and a file were given; using the uploaded file.")
        job.start()

        hidden = gr.DownloadButton(visible=False)
        yield render_status(job), "", None, hidden, hidden, "", *busy()
        last_version, last_tick = -1, 0.0
        try:
            while not job.done.wait(0.5):
                if job.version != last_version or time.time() - last_tick >= 1.0:
                    last_version, last_tick = job.version, time.time()
                    yield render_status(job), render_log(job), *skip(6)
        finally:
            if not job.done.is_set():  # page closed or event cancelled: stop the worker too
                job.cancel.set()

        result = job.result
        if result and result.mp3_path:
            script_text = Path(result.script_txt).read_text(encoding="utf-8")
            yield (
                render_status(job), render_log(job), str(result.mp3_path),
                gr.DownloadButton("Download MP3", value=str(result.mp3_path), visible=True),
                gr.DownloadButton("Download script (.txt)", value=str(result.script_txt), visible=True),
                script_text, *idle(),
            )
        else:
            yield render_status(job), render_log(job), None, hidden, hidden, "", *idle()

    def stop() -> None:
        with _current_lock:
            job = _current
        if job and not job.done.is_set():
            job.cancel.set()
            job.log("Stopping at the next progress update...")
            gr.Info("Stopping… this takes a few seconds (up to a minute while the LLM reads a long prompt).")

    return check_system, generate, stop


# ----------------------------------------------------------------------- page
def build_app(settings: Settings) -> gr.Blocks:
    check_system, generate, stop = make_handlers(settings)
    sc = settings.script
    voices = settings.tts.options()
    length_choices = [(f"{n.title()} (about {sc.minutes[n]} min)", n) for n in LENGTHS]

    with gr.Blocks(title="Podcastify") as demo:
        gr.Markdown(
            "# 🎙️ Podcastify\n"
            "Turn a PDF, blog post or YouTube video into a two-host podcast. "
            "Everything runs on this computer with free, open-source models."
        )
        checks = gr.Markdown("Checking Ollama and ffmpeg…", elem_classes="pc-checks")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                source_box = gr.Textbox(
                    label="Blog URL, YouTube URL or file path",
                    placeholder="https://…  or  C:\\Users\\you\\Documents\\paper.pdf",
                    lines=1,
                )
                upload = gr.File(
                    label="…or upload a PDF / text file",
                    file_types=[".pdf", ".txt", ".md"],
                    type="filepath",
                    height=120,
                )
                with gr.Row():
                    length = gr.Dropdown(length_choices, value=sc.default_length, label="Episode length", scale=2)
                with gr.Accordion("Options", open=False):
                    new_script = gr.Checkbox(label="Write a new script even if one is cached for this input")
                    force_asr = gr.Checkbox(label="YouTube: ignore captions and transcribe the audio on this computer")
                    gr.Markdown(
                        f"Hosts: **{sc.host_a_name}** explains (voice `{voices.get('voice_a', '?')}`), "
                        f"**{sc.host_b_name}** asks (voice `{voices.get('voice_b', '?')}`). "
                        f"Model: `{settings.llm.model}`. Change these in `config.toml`."
                    )
                with gr.Row():
                    go = gr.Button("Generate podcast", variant="primary", scale=3)
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False, scale=1)
                gr.Examples(
                    examples=[
                        [str(PROJECT_ROOT / "samples" / "noise_cancelling.txt")],
                        ["https://paulgraham.com/hwh.html"],
                        ["https://www.youtube.com/watch?v=aircAruvnKk"],
                    ],
                    inputs=[source_box],
                    label="Try one of these",
                )
                gr.Markdown(
                    "On a laptop CPU a short episode takes about 6–10 minutes and a medium one about "
                    "30. Keep the laptop plugged in: on battery it can be twice as slow.",
                    elem_classes="pc-checks",
                )

            with gr.Column(scale=6):
                status = gr.HTML(render_status(None))
                audio = gr.Audio(label="Episode", type="filepath", interactive=False, buttons=["download"])
                with gr.Row():
                    dl_mp3 = gr.DownloadButton("Download MP3", visible=False, variant="primary")
                    dl_txt = gr.DownloadButton("Download script (.txt)", visible=False)
                with gr.Accordion("Script", open=False):
                    script_box = gr.Textbox(show_label=False, lines=14, max_lines=40, interactive=False, buttons=["copy"])
                with gr.Accordion("Details (log)", open=False):
                    log_box = gr.Textbox(show_label=False, lines=12, max_lines=12, interactive=False, autoscroll=True)

        demo.load(check_system, outputs=checks, show_progress="hidden")
        go.click(
            generate,
            inputs=[source_box, upload, length, new_script, force_asr],
            outputs=[status, log_box, audio, dl_mp3, dl_txt, script_box, go, stop_btn],
            show_progress="hidden",
            concurrency_limit=1,
        )
        stop_btn.click(stop, queue=False)
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="podcastify-ui", description="Start the Podcastify web UI.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this computer (default)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("-c", "--config", help="path to a config.toml")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    settings.general.output_dir.mkdir(parents=True, exist_ok=True)
    demo = build_app(settings)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        allowed_paths=[str(settings.general.output_dir)],
        max_file_size="300mb",
        theme=gr.themes.Soft(),
        css=CSS,
        footer_links=["settings"],
    )


if __name__ == "__main__":
    main()
