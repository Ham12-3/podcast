# Podcastify

Turn a **PDF**, a **blog post** or a **YouTube video** into a two-host podcast MP3, on your
own Windows laptop. No paid APIs and no accounts: every model is free, open source and runs
locally on the CPU (a GPU makes it faster, see [Switching to an NVIDIA GPU](#switching-to-an-nvidia-gpu)).

- **Ava** (Host A) explains the topic in plain English.
- **Leo** (Host B) asks questions, reacts and gives examples.
- You get an MP3 plus the script as a readable `.txt` (and `.json`), so you can check or edit it.

Use it from the command line or from a small web page that runs on your computer.

---

## Contents

- [How it works](#how-it-works)
- [Windows setup](#windows-setup)
- [Make a podcast: command line](#make-a-podcast-command-line)
- [Make a podcast: web UI](#make-a-podcast-web-ui)
- [How long it takes](#how-long-it-takes)
- [Settings](#settings-configtoml)
- [Where files go](#where-files-go)
- [Troubleshooting](#troubleshooting)
- [Switching to an NVIDIA GPU](#switching-to-an-nvidia-gpu)
- [Adding a new voice engine (e.g. VibeVoice)](#adding-a-new-voice-engine-eg-vibevoice)
- [Project layout](#project-layout)

---

## How it works

```
 input ──► 1. extract text ──► 2. write the script ──► 3. record the voices ──► 4. make the MP3
           PDF / blog / YouTube    local LLM (Ollama)      Kokoro TTS             pydub + ffmpeg
```

| Step | What happens | Tool / model |
|---|---|---|
| Extract | **PDF:** reads multi-column pages in order, joins hyphenated words, drops headers, footers, page numbers, figure labels, tables, `[12]` citations and the References section. **Blog:** keeps the article, drops menus, ads and "related posts". **YouTube:** uses the captions; if there are none, downloads the audio and transcribes it locally. | PyMuPDF (+ layout model), trafilatura, youtube-transcript-api, yt-dlp, NVIDIA Parakeet TDT 0.6B v3 via onnx-asr |
| Script | Long texts are split into chunks and summarised first. Then the LLM plans the episode and writes the dialogue as JSON (checked, and repaired or retried if needed). | Ollama with `qwen3.5:4b` |
| Voices | Each line is spoken with the host's voice. | Kokoro 82M (pykokoro, ONNX) |
| MP3 | Lines are trimmed, matched in loudness and joined with short, slightly varied pauses. | pydub + ffmpeg |

Extracted text, summaries and finished scripts are **cached**, so running the same input
again (for example with other voices) skips straight to the voices.

---

## Windows setup

**You need:** Windows 10 or 11, 16 GB RAM recommended (8 GB works for short episodes),
about 5 GB of free disk space, and an internet connection for the setup and for URLs.

Open **PowerShell** for the steps below. After each `winget install`, open a **new**
PowerShell window so the new program is found.

### 1. Install uv (Python project manager)

```bash
winget install --id=astral-sh.uv -e
```

(Or the official installer: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.)
You don't need to install Python yourself: uv downloads Python 3.11 for this project.

### 2. Install Ollama and download the language model

```bash
winget install --id Ollama.Ollama -e
```

(Or download it from <https://ollama.com/download>.) Ollama runs in the background; its
llama icon sits in the system tray. Then download the model (3.3 GB, once):

```bash
ollama pull qwen3.5:4b
```

### 3. Install ffmpeg (needed to make MP3s and to read YouTube audio)

```bash
winget install --id Gyan.FFmpeg -e
```

Check it in a new window:

```bash
ffmpeg -version
```

### 4. Install Podcastify

In PowerShell, go to the project folder (the one that contains `pyproject.toml`) and run:

```bash
uv sync
```

This creates a `.venv` folder with everything Podcastify needs (about 760 MB). It also
includes the eSpeak pronunciation engine and the JavaScript runtime that YouTube downloads
need, so there is nothing else to install.

### 5. Try it

```bash
uv run podcastify samples/noise_cancelling.txt --length short
```

The first run downloads the Kokoro voice model (about 340 MB). After about 6 minutes on a
laptop CPU (about 10 on battery) you'll find the episode in the `output` folder.

> **Keep the laptop plugged in.** On battery, Windows slows the CPU down and the voices
> take about twice as long. Podcastify stops Windows from going to sleep while it works,
> but closing the lid still puts the laptop to sleep.

---

## Make a podcast: command line

```bash
uv run podcastify "C:\Users\you\Documents\paper.pdf" --length medium
```

```bash
uv run podcastify https://paulgraham.com/hwh.html --length short
```

```bash
uv run podcastify "https://www.youtube.com/watch?v=aircAruvnKk" --length long
```

Inputs: a PDF, a `.txt` / `.md` file, a blog or article URL (a link straight to a PDF works
too), or a YouTube link (`watch?v=`, `youtu.be/`, `/shorts/`, `/embed/` and `/live/` links).
Put paths with spaces in quotes.

**Lengths** (change them in `config.toml`): `short` about 3 minutes, `medium` about 8,
`long` about 15.

**Output:** `output/<title>_<length>.mp3`, plus `.txt` (readable script) and `.json`.

| Option | What it does |
|---|---|
| `-l`, `--length short\|medium\|long` | Episode length (default: `medium`, from `config.toml`) |
| `-o`, `--out FOLDER` | Save the files somewhere else |
| `--script-only` | Write and save the script, but don't record it |
| `--from-script FILE` | Record audio from a saved `.txt` or `.json` script, e.g. one you edited |
| `--extract-only` | Only extract the text and save it as `output/<title>_source.txt` |
| `--new-script` | Write a new script even if one is cached |
| `--no-cache` | Ignore everything cached (text, summaries, scripts) |
| `--force-asr` | YouTube: ignore captions and transcribe the audio on this computer |
| `--model NAME` | Use another Ollama model for this run, e.g. `qwen3.5:9b` |
| `--voice-a NAME`, `--voice-b NAME` | Use other voices for this run, e.g. `af_bella`, `bm_george` |
| `--tts NAME` | Use another voice engine (see [Adding a new voice engine](#adding-a-new-voice-engine-eg-vibevoice)) |
| `-c`, `--config FILE` | Use another settings file |

### Check and edit the script before recording

The LLM can get a detail wrong, especially with the small CPU model. To review first:

```bash
uv run podcastify paper.pdf --length medium --script-only
```

Open the `.txt` in `output`, fix anything you like (keep the `Ava:` / `Leo:` at the start
of each line), then record it:

```bash
uv run podcastify --from-script output/deep-residual-learning-for-image-recognition_medium.txt
```

At the end of every run Podcastify prints how long each stage took:

```
Timings:
  extraction       13.5s  ( 0.7%)
  script       26m 30.1s  (81.2%)
  tts           5m 53.2s  (18.0%)
  stitching         2.6s  ( 0.1%)
  total        32m 39.5s
```

---

## Make a podcast: web UI

```bash
uv run podcastify-ui
```

Your browser opens <http://127.0.0.1:7860>. Paste a link or a file path, or drop a PDF /
text file onto the page, pick the length and click **Generate podcast**.

- Each stage (reading the input, writing the script, recording the voices, making the MP3)
  has its own progress bar, time and live details, e.g. "Speaking line 39/46 (about 24s left)".
- When it's done you can play the episode, download the MP3 and the script, and read the script.
- **Stop** cancels the run within a few seconds (up to a minute while the LLM is still
  reading a long prompt). Anything already finished stays cached, so a rerun continues from there.
- The top of the page tells you if Ollama isn't running, the model isn't downloaded or ffmpeg is missing.
- **Options:** write a new script even if one is cached; transcribe a YouTube video locally even if it has captions.

Only one episode is made at a time. Options: `--port 8000`, `--no-browser`. The page is
only reachable from this computer. `--host 0.0.0.0` makes it reachable from other devices
on your network, so only do that on a network you trust.

---

## How long it takes

Measured on an Intel Core i5-1135G7 laptop with 16 GB RAM and no GPU, plugged in:

| Input | Length | Script | Voices | Total |
|---|---|---|---|---|
| 420-word text file | short (2m 55s of audio) | 3m 43s | 2m 08s | about 6 min |
| 12-page research paper (PDF) | medium (5m 10s of audio) | 26m 30s (about 17 min of it summarising) | 5m 53s | 32m 40s |
| Same paper, script already cached | medium | from cache | 5m 53s | about 6 min |

- The LLM writes about 5–6 tokens (roughly 4 words) per second on this CPU, so long inputs and long episodes take longest.
- Voices: about 1.4× faster than real time when plugged in; about 0.7× on battery.
- YouTube captions take a few seconds. Without captions, a 19-minute video took
  5 minutes to transcribe (3.6× real time), and the text differed from the human-written
  captions by only 2.4%.
- With an NVIDIA GPU everything gets several times faster.

---

## Settings (`config.toml`)

All settings live in `config.toml` in the project folder. Missing keys fall back to
defaults. The ones you are most likely to change:

| Section | Setting | Default | Notes |
|---|---|---|---|
| `[llm]` | `model` | `qwen3.5:4b` | Any model you've pulled with `ollama pull` |
| `[llm]` | `chunk_chars` | `6000` | Inputs longer than this are summarised in chunks first |
| `[llm]` | `unload_after_script` | `true` | Frees the LLM's memory (3.4 GB) before the voices start |
| `[script]` | `host_a_name`, `host_b_name` | `Ava`, `Leo` | The hosts' names in the dialogue |
| `[script]` | `show_name` | `Podcastify` | Said in the opening |
| `[script.minutes]` | `short`, `medium`, `long` | `3`, `8`, `15` | Target minutes per length |
| `[tts.kokoro]` | `voice_a`, `voice_b` | `af_heart`, `am_michael` | See voices below |
| `[tts.kokoro]` | `language` | `en-us` | Use `en-gb` with British voices |
| `[tts.kokoro]` | `speed` | `1.0` | `1.1` = a bit faster |
| `[tts.kokoro]` | `model_quality` | `fp32` | `q8` is smaller but was not faster on the test laptop |
| `[audio]` | `pause_ms`, `pause_jitter_ms` | `350`, `120` | Pause between lines and how much it varies |
| `[audio]` | `mp3_bitrate` | `128k` | |
| `[extract]` | `pdf_layout` | `true` | `false` = much faster but rougher PDF reading |
| `[extract]` | `skip_references` | `true` | Drop a paper's References section |
| `[extract]` | `youtube_languages` | `["en"]` | Caption languages to try, in order |
| `[extract]` | `max_video_minutes` | `180` | Longest video it will transcribe without captions |
| `[asr]` | `model` | `nemo-parakeet-tdt-0.6b-v3` | Speech-to-text for videos without captions |
| `[general]` | `keep_awake` | `true` | Stop Windows from sleeping during a run |

**Voices** (Kokoro v1.0, English): American female `af_heart`, `af_bella`, `af_nicole`,
`af_sarah`, `af_sky`; American male `am_michael`, `am_adam`, `am_eric`, `am_liam`;
British female `bf_emma`, `bf_isabella`, `bf_alice`; British male `bm_george`,
`bm_lewis`, `bm_daniel`. Pick different voices for the two hosts. To hear two voices
quickly:

```bash
uv run python scripts/try_voices.py --voice-a af_bella --voice-b bm_george
```

This writes `output/voices_test.wav`.

---

## Where files go

| What | Where |
|---|---|
| Episodes, scripts, `--extract-only` text | `output\` in the project folder |
| Cached text, summaries, scripts, YouTube audio | `cache\` in the project folder (safe to delete) |
| Python packages | `.venv\` in the project folder |
| Ollama models | `%USERPROFILE%\.ollama\models` (remove one with `ollama rm <name>`) |
| Kokoro voice model | `%LOCALAPPDATA%\onnxvoice` |
| Speech-to-text model (only downloaded when a video has no captions, ~670 MB) | `%USERPROFILE%\.cache\huggingface\hub` |

---

## Troubleshooting

| Message or problem | What to do |
|---|---|
| *Can't reach Ollama* | Start the Ollama app from the Start menu (or run `ollama serve`), then try again. |
| *The model 'qwen3.5:4b' is not downloaded* | Run `ollama pull qwen3.5:4b`. |
| *ffmpeg was not found* | `winget install --id Gyan.FFmpeg -e`, then open a new PowerShell window. |
| Very slow | Plug the laptop in, close heavy apps, and try `--length short` first. |
| *No text found in the PDF* | It's a scanned PDF (pictures of pages). Run it through OCR first (e.g. "save as searchable PDF" in your scanner app). |
| *The PDF is password-protected* | Open it in a PDF viewer and save or print a copy without a password. |
| *The site refused the download (HTTP 403)* or no article text found | The site blocks apps or needs JavaScript or a login. Open the page in your browser, press Ctrl+P, choose "Save as PDF" and use that file. |
| YouTube: *confirm this isn't a bot* | YouTube is limiting your network. Try again later. If captions are blocked, Podcastify tries the audio automatically. |
| YouTube: *age-restricted* / *private* | These videos can't be read without signing in, which Podcastify doesn't do. |
| A medium or long episode comes out shorter than the target | Known limitation of the small 4B model: it writes shorter lines than planned (one 8-minute target gave about 5 minutes). A bigger model on a GPU follows the length better. |
| The hosts get a detail wrong | Use `--script-only`, fix the `.txt`, then `--from-script`. |
| SSL / certificate errors | Podcastify uses the Windows certificate store, which fixes most of these. On a work network a proxy may still block downloads. |

---

## Switching to an NVIDIA GPU

Three parts use the CPU today and can use an NVIDIA GPU: the LLM (Ollama), the voices
(Kokoro) and speech-to-text (Parakeet). Install the latest NVIDIA driver first, then run
`nvidia-smi` and note the **CUDA Version** it shows.

### 1. The LLM: just pick a bigger model

Ollama uses an NVIDIA GPU automatically. A GPU lets you use a bigger, better model, which
writes more natural scripts, follows the target length better and makes fewer mistakes:

| GPU memory (VRAM) | Model | Download |
|---|---|---|
| 8–12 GB | `qwen3.5:9b` | 6.6 GB |
| 24 GB | `qwen3.5:27b` | 17 GB |

```bash
ollama pull qwen3.5:9b
```

Then in `config.toml`:

```toml
[llm]
model = "qwen3.5:9b"
num_ctx = 16384        # more context fits on a GPU...
chunk_chars = 12000    # ...so long documents need fewer summaries
```

While a script is being written, `ollama ps` should show `100% GPU`. If it shows part CPU,
the model plus its context doesn't fit in GPU memory: set `num_ctx` back to `8192`. You can try a model
for one run without changing the file: `uv run podcastify paper.pdf --model qwen3.5:9b`.

### 2. Voices and speech-to-text: install the GPU build of onnxruntime

Kokoro, Parakeet and PyMuPDF's layout model all run on **onnxruntime**. It comes in a CPU
package (`onnxruntime`) and a GPU package (`onnxruntime-gpu`). Both provide the same Python
module, so the CPU one has to be removed when the GPU one is added. Edit `pyproject.toml`:

1. In `dependencies`, change `"pykokoro[cpu]>=0.10.10"` to `"pykokoro[gpu]>=0.10.10"`.
2. Change `"onnx-asr[cpu,hub]>=0.12.0"` to `"onnx-asr[gpu,hub]>=0.12.0"`.
3. Add this line to `dependencies`, which also installs CUDA and cuDNN as Python packages,
   so you don't need NVIDIA's separate CUDA installer:
   ```toml
       "onnxruntime-gpu[cuda,cudnn]>=1.24",
   ```
   If `nvidia-smi` shows a CUDA Version **below 13** (a CUDA 12 driver), use
   `"onnxruntime-gpu[cuda,cudnn]>=1.24,<1.27"` instead.
4. At the end of the file, add:
   ```toml
   [tool.uv]
   # The GPU build of onnxruntime replaces the CPU one (both provide the same module).
   exclude-dependencies = ["onnxruntime"]
   ```

Then install:

```bash
uv sync
```

Check that the GPU is visible (it should list `CUDAExecutionProvider`):

```bash
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Finally, set the provider in `config.toml`:

```toml
[tts.kokoro]
provider = "cuda"

[asr]
provider = "cuda"
```

The run log then shows `TTS: kokoro v1.0/fp32 on cuda`. Podcastify loads the CUDA
libraries from the pip packages automatically. If `provider = "cuda"` is set but the GPU
build isn't installed, it prints a warning and uses the CPU instead of failing.

Both versions of this change were checked to resolve on Windows with uv 0.11 (CUDA 13:
onnxruntime-gpu 1.31; CUDA 12: onnxruntime-gpu 1.24). They couldn't be run without an
NVIDIA card, so if something fails, the error usually names the missing CUDA library.
To undo the change, restore `pyproject.toml` and `uv.lock` (e.g. `git checkout pyproject.toml uv.lock`)
and run `uv sync`.

---

## Adding a new voice engine (e.g. VibeVoice)

The rest of the app only talks to the `TTSBackend` class in
[`src/podcastify/tts/base.py`](src/podcastify/tts/base.py), so a new engine is one new file
plus two small edits:

**1. Write `src/podcastify/tts/vibevoice_backend.py`:**

```python
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import AudioClip, ProgressFn, ScriptLine, TTSBackend, TTSError


class VibeVoiceBackend(TTSBackend):
    name = "vibevoice"

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)          # options = the [tts.vibevoice] section
        self.voices = {"A": options["voice_a"], "B": options["voice_b"]}
        # Load the model here, e.g. from options["model_path"] onto options.get("device", "cuda").
        # Raise TTSError("...") with a clear message if it can't be loaded.

    def synthesize(self, text: str, speaker: str) -> AudioClip:
        """Required: one line for speaker "A" or "B"."""
        samples, sample_rate = ..., 24000  # call the model here
        return AudioClip(np.asarray(samples, dtype=np.float32), sample_rate)  # mono, -1..1

    def synthesize_script(
        self, lines: Sequence[ScriptLine], on_progress: ProgressFn | None = None
    ) -> list[AudioClip]:
        """Optional: multi-speaker models sound most natural when they render the whole
        conversation in one go. Return [one_clip]; the MP3 step then only adds a short
        silence at the start and end. Without this method, synthesize() is called per line."""
        return super().synthesize_script(lines, on_progress)

    def describe(self) -> str:
        return f"vibevoice | A={self.voices['A']} B={self.voices['B']}"

    def close(self) -> None:
        """Free the model / GPU memory."""
```

**2. Register it** in [`src/podcastify/tts/__init__.py`](src/podcastify/tts/__init__.py)
(the line is already there, commented out):

```python
_REGISTRY: dict[str, str] = {
    "kokoro": ".kokoro_backend:KokoroBackend",
    "vibevoice": ".vibevoice_backend:VibeVoiceBackend",
}
```

**3. Configure it** in `config.toml`:

```toml
[tts]
backend = "vibevoice"

[tts.vibevoice]
voice_a = "..."
voice_b = "..."
```

Try it for one run with `--tts vibevoice` before changing the default. Backends are only
imported when selected, so a GPU-only engine never loads on a CPU machine.

**About VibeVoice today:** Microsoft removed the VibeVoice **TTS** code from its official
repository ([microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)) over
responsible-AI concerns. The multi-speaker model is still available through a community
fork ([vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice)).
Check its current state, license and terms of use before building on it. The steps above
work the same for any other engine.

---

## Project layout

```
config.toml                 all settings
samples/                    a sample article to try
scripts/                    try_voices.py (hear two voices), try_script.py (script only)
src/podcastify/
  cli.py                    `podcastify` command
  app.py                    `podcastify-ui` web page (Gradio)
  pipeline.py               runs the stages, caching and timings
  extractors/               pdf.py, web.py, youtube.py, textfile.py -> plain text
  asr.py                    local speech-to-text (Parakeet via onnx-asr)
  script_writer.py          prompts, chunking + summaries, JSON checks and retries
  tts/                      base.py (interface), kokoro_backend.py
  audio.py                  trimming, loudness, pauses, MP3 export
  config.py                 reads config.toml
  cache.py, script.py, system.py, timing.py, textutil.py, errors.py
```

## Built with

[Ollama](https://ollama.com) and [Qwen3.5](https://ollama.com/library/qwen3.5) ·
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via [pykokoro](https://pypi.org/project/pykokoro/) ·
[PyMuPDF](https://pymupdf.readthedocs.io) · [trafilatura](https://trafilatura.readthedocs.io) ·
[youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) ·
[yt-dlp](https://github.com/yt-dlp/yt-dlp) ·
[onnx-asr](https://github.com/istupakov/onnx-asr) with [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) ·
[pydub](https://github.com/jiaaro/pydub) and [FFmpeg](https://ffmpeg.org) · [Gradio](https://www.gradio.app).
Each has its own license; check them before publishing episodes commercially.
