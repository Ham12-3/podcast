"""YouTube videos.

1. Captions first (youtube-transcript-api): fast and accurate. Manual captions are
   preferred over auto-generated ones. If there are none in your language but another
   language can be translated by YouTube, that translation is used.
2. No captions? The audio is downloaded with yt-dlp, converted to 16 kHz mono WAV with
   ffmpeg, and transcribed locally (see podcastify/asr.py).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

from ..errors import ExtractionError
from ..textutil import normalize_text, word_count
from ..timing import fmt_seconds
from . import Extracted, ExtractOptions

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
PARAGRAPH_WORDS = 120
PARAGRAPH_GAP_S = 2.0


# --------------------------------------------------------------- video id
def parse_video_id(url: str) -> str:
    """Accepts watch?v=, youtu.be/, /shorts/, /embed/ and /live/ links."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    query = parse_qs(parsed.query)
    video_id = ""
    if host == "youtu.be":
        video_id = parts[0] if parts else ""
    elif "v" in query:
        video_id = query["v"][0]
    elif len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v", "e"):
        video_id = parts[1]
    elif "list" in query:
        raise ExtractionError(
            "That is a playlist link. Use the link of a single video (it contains watch?v=...)."
        )
    if not _VIDEO_ID.match(video_id):
        raise ExtractionError(
            f"Couldn't find a video in {url}. Use a link like https://www.youtube.com/watch?v=VIDEO_ID "
            "or https://youtu.be/VIDEO_ID"
        )
    return video_id


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _oembed_title(video_id: str, timeout: int) -> str | None:
    """Video title from YouTube's public oEmbed endpoint (no API key needed)."""
    try:
        resp = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": _watch_url(video_id), "format": "json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return (resp.json().get("title") or "").strip() or None
    except (httpx.HTTPError, ValueError):
        pass
    return None


# --------------------------------------------------------------- captions
def _paragraphs(snippets) -> str:
    """Join caption snippets into paragraphs (split at long pauses or every ~120 words)."""
    paragraphs, current, last_end = [], [], 0.0
    for snip in snippets:
        text = re.sub(r"\[[^\]]*\]|\([A-Za-z ]+\)|>>", " ", snip.text)  # [Music], (applause), >>
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if current and (snip.start - last_end > PARAGRAPH_GAP_S or len(" ".join(current).split()) > PARAGRAPH_WORDS):
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        last_end = snip.start + snip.duration
    if current:
        paragraphs.append(" ".join(current))
    return normalize_text("\n\n".join(paragraphs))


def _pick_transcript(transcripts, languages: list[str]):
    """Return (transcript, description) using the preferred languages when possible."""
    for finder in (transcripts.find_manually_created_transcript, transcripts.find_generated_transcript):
        try:
            t = finder(languages)
        except Exception:  # NoTranscriptFound
            continue
        auto = " (auto-generated)" if t.is_generated and "auto" not in t.language.lower() else ""
        return t, f"{t.language}{auto} captions"
    available = list(transcripts)
    if not available:
        return None, ""
    original = sorted(available, key=lambda t: t.is_generated)[0]  # manual first
    codes = {lang.language_code for lang in original.translation_languages}
    if original.is_translatable and languages[0] in codes:
        return original.translate(languages[0]), f"{original.language} captions, translated to '{languages[0]}' by YouTube"
    return original, f"{original.language} captions (no '{languages[0]}' version available)"


def _captions(video_id: str, languages: list[str], log) -> str | None:
    """Caption text, or None when the video has no usable captions."""
    import requests
    from youtube_transcript_api import (
        AgeRestricted,
        CouldNotRetrieveTranscript,
        InvalidVideoId,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApi,
    )

    try:
        transcripts = YouTubeTranscriptApi().list(video_id)
        transcript, description = _pick_transcript(transcripts, languages)
        if transcript is None:
            log("  This video has no captions")
            return None
        fetched = transcript.fetch()
    except TranscriptsDisabled:
        log("  This video has no captions (they are turned off)")
        return None
    except VideoUnavailable as exc:
        raise ExtractionError(
            f"This video is unavailable: {_watch_url(video_id)}. It may be private, deleted, "
            "or blocked in your country."
        ) from exc
    except InvalidVideoId as exc:
        raise ExtractionError(f"That is not a valid YouTube video: {_watch_url(video_id)}") from exc
    except AgeRestricted as exc:
        raise ExtractionError(
            "This video is age-restricted. YouTube only shows it to signed-in users, so Podcastify can't read it."
        ) from exc
    except RequestBlocked:
        log("  YouTube blocked the caption request from this network; trying the audio instead")
        return None
    except CouldNotRetrieveTranscript as exc:
        log(f"  Couldn't read the captions ({type(exc).__name__}); trying the audio instead")
        return None
    except requests.exceptions.RequestException as exc:
        raise ExtractionError(f"Could not reach YouTube ({type(exc).__name__}). Check your internet connection.") from exc

    text = _paragraphs(fetched)
    log(f"  Using {description}: {word_count(text)} words")
    return text


# ------------------------------------------------------------ audio + ASR
class _YtdlpLogger:
    """Keep yt-dlp quiet; remember errors so we can explain them."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        self.errors.append(msg)


def _explain_download_error(message: str, url: str) -> str:
    low = message.lower()
    if "not a bot" in low or "sign in to confirm" in low:
        return (
            "YouTube asked to confirm this isn't a bot, so the audio couldn't be downloaded. "
            "Try again later, or from another network."
        )
    if "private video" in low:
        return f"This video is private: {url}"
    if "age" in low and "sign in" in low:
        return "This video is age-restricted, so Podcastify can't download it."
    if "unavailable" in low or "removed" in low:
        return f"This video is unavailable: {url}. It may be deleted or blocked in your country."
    if "unable to download" in low or "timed out" in low or "getaddrinfo" in low:
        return f"Could not download the audio from YouTube. Check your internet connection. ({message})"
    return f"Could not download the audio from YouTube: {message}"


def _download_audio(video_id: str, folder: Path, opts: ExtractOptions) -> tuple[Path, str | None]:
    import yt_dlp
    from yt_dlp.utils import DownloadError

    url = _watch_url(video_id)
    logger = _YtdlpLogger()

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                frac = d.get("downloaded_bytes", 0) / total
                opts.progress(0.02 + 0.2 * frac, f"Downloading audio ({frac:.0%} of {total / 2**20:.0f} MB)")

    deno = Path(sys.executable).parent / ("deno.exe" if sys.platform == "win32" else "deno")
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(folder / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": logger,
        "progress_hooks": [hook],
        "socket_timeout": 30,
        "js_runtimes": {"deno": {"path": str(deno)}} if deno.exists() else None,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            minutes = (info.get("duration") or 0) / 60
            limit = opts.settings.max_video_minutes
            if minutes > limit:
                raise ExtractionError(
                    f"The video is {minutes:.0f} minutes long and has no captions. Transcribing it on CPU "
                    f"would take a long time, so videos over {limit} minutes are skipped "
                    "(change max_video_minutes in config.toml)."
                )
            if info.get("is_live"):
                raise ExtractionError("That is a live stream. Wait until it has finished and try again.")
            opts.log(f"  Downloading audio ({fmt_seconds(info.get('duration') or 0)} long)")
            info = ydl.process_ie_result(info, download=True)
    except DownloadError as exc:
        raise ExtractionError(_explain_download_error(logger.errors[-1] if logger.errors else str(exc), url)) from exc

    downloads = info.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))
    if not path.is_file():
        raise ExtractionError(f"yt-dlp finished but the audio file is missing: {path}")
    return path, info.get("title")


def _to_wav(src: Path, dst: Path) -> None:
    from ..audio import find_ffmpeg

    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not dst.is_file():
        raise ExtractionError(f"ffmpeg could not convert the downloaded audio: {result.stderr.strip()[-400:]}")


def _transcribe_video(video_id: str, opts: ExtractOptions) -> tuple[str, str | None]:
    from ..asr import transcribe

    folder = opts.work_dir or Path.cwd() / "cache" / "downloads"
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / f"{video_id}.16k.wav"
    title = None
    if wav.is_file():
        opts.log(f"  Using audio downloaded earlier: {wav.name}")
    else:
        audio, title = _download_audio(video_id, folder, opts)
        opts.progress(0.22, "Converting audio")
        _to_wav(audio, wav)
        audio.unlink(missing_ok=True)  # keep only the small 16 kHz WAV

    def asr_progress(frac: float, msg: str) -> None:
        opts.progress(0.25 + 0.75 * frac, msg)

    text = transcribe(wav, opts.asr, log=opts.log, progress=asr_progress)
    return normalize_text(text), title


# ------------------------------------------------------------------- main
def extract_youtube(source: str, opts: ExtractOptions | None = None) -> Extracted:
    opts = opts or ExtractOptions()
    settings = opts.settings
    video_id = parse_video_id(source)
    opts.progress(0.0, "Looking up the video")
    title = _oembed_title(video_id, settings.web_timeout)

    text = None
    if settings.force_asr:
        opts.log("  force_asr is on: skipping captions")
    else:
        text = _captions(video_id, list(settings.youtube_languages) or ["en"], opts.log)
    if not text:
        opts.log("  Transcribing the audio locally instead (this takes a while on CPU)")
        text, ytdlp_title = _transcribe_video(video_id, opts)
        title = title or ytdlp_title
    if not text.strip():
        raise ExtractionError(f"No speech was found in the video: {_watch_url(video_id)}")

    return Extracted(
        text=text,
        title=title or f"YouTube video {video_id}",
        source_type="youtube",
        source=_watch_url(video_id),
    )
