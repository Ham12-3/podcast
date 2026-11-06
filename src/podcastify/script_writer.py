"""Write a two-host podcast script with a local LLM served by Ollama.

Flow:
  1. Source text longer than `chunk_chars` is split into chunks; each chunk is
     summarised into notes (and the notes again if still too long), so everything
     fits a small model's context window.
  2. Short scripts are written in one call. Medium/long scripts first get an outline
     of N parts, then each part is written separately with the end of the previous
     part as context. Small models stay much closer to the target length this way.
  3. Every reply is forced into a JSON schema by Ollama, then validated and cleaned.
     If it is still broken, we try a local fix, then retry once with a repair prompt.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from ollama import Client, ResponseError

from .cache import Cache, make_key
from .config import LLMSettings, ScriptSettings
from .errors import LLMError
from .script import SPEAKERS, ScriptLine
from .textutil import normalize_text

# Bump when prompts change so cached scripts/summaries are not reused.
PROMPT_VERSION = "3"
WORDS_PER_LINE = 28  # average; used to suggest a line count to the model

# Progress inside the script stage: (fraction 0..1, message)
ProgressFn = Callable[[float, str], None]

def lines_schema(min_items: int = 1) -> dict[str, Any]:
    """JSON schema Ollama enforces while generating. minItems stops small models from
    closing the array after a single line (Ollama's grammar honours it)."""
    return {
        "type": "array",
        "minItems": max(1, min_items),
        "items": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string", "enum": list(SPEAKERS)},
                "text": {"type": "string"},
            },
            "required": ["speaker", "text"],
        },
    }
OUTLINE_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

# --------------------------------------------------------------------------- prompts

SYSTEM_PROMPT = """You write scripts for "{show}", a friendly podcast where two hosts explain a topic to a general audience.

The hosts:
- Host A is {a}, the explainer. {a} has studied the material and explains it clearly, one idea at a time. {a}'s lines are usually two to four sentences.
- Host B is {b}, the curious co-host. {b} asks the questions a listener would ask, reacts naturally ("Oh, interesting!", "Wait, so..."), and adds simple everyday examples or comparisons. {b}'s lines are usually one or two sentences.
- {b} asks and {a} answers. {a} never asks {b} to explain the topic.

Rules:
- Plain, simple English. Short sentences. Talk like real people, using contractions.
- When a technical word is needed, explain it straight away in simple words.
- The hosts take turns speaking.
- Use only facts from the source. Do not make up numbers, names, dates or quotes.
- Write only spoken words: no stage directions, sound effects, emojis, markdown, or name labels inside the text.

When asked for dialogue, output only a JSON array where each item is {{"speaker": "A" or "B", "text": "what they say"}}."""

OUTLINE_PROMPT = """SOURCE:
\"\"\"
{notes}
\"\"\"

We will turn this source into one podcast episode in {n} parts.
Plan the parts in a logical order for a beginner, so together they cover the most important ideas without repeating each other.
Return only a JSON array of {n} strings. Each string is one sentence saying what that part covers."""

PART_PROMPT = """SOURCE:
\"\"\"
{notes}
\"\"\"

{plan}{so_far}Write {what}.
- Length: about {words} words, around {lines} lines.
- {position}
Return only the JSON array."""

OPENING = (
    "Open naturally: {a} welcomes listeners to {show}, says hello to {b} and introduces "
    "today's topic; {b} reacts with curiosity about why it matters."
)
WRAP_UP = (
    "End with a short wrap-up: recap the two or three key takeaways, thank the "
    "listeners and say goodbye."
)
POSITION_RULES = {
    "single": OPENING + " Then explain the main ideas in a logical order. " + WRAP_UP,
    "first": OPENING + " Then start on this part's topic. Do not say goodbye; more parts follow.",
    "middle": "Carry on smoothly from where the conversation stopped. Cover only this part's topic. No greetings and no goodbyes.",
    "last": "Carry on smoothly from where the conversation stopped and cover this part's topic. " + WRAP_UP,
}

SUMMARY_SYSTEM = "You take careful, accurate notes for a podcast research team."

SUMMARY_PROMPT = """Below is part {i} of {n} of a longer document.

\"\"\"
{chunk}
\"\"\"

Write clear notes on this part in about {words} words: the main ideas, key facts and numbers, names, examples and anything surprising.
Use short bullet points in plain English. Only use what is in the text. Write only the notes."""

REPAIR_PROMPT = """Your previous answer was supposed to be a JSON array of objects like {{"speaker": "A", "text": "..."}}, with speaker "A" or "B", but it has a problem: {error}

Previous answer:
{raw}

Return the corrected JSON array only. Keep the same dialogue."""

# Used when the JSON was fine but the content was not (too short, one host only).
# Sent as a follow-up in the same conversation so the source stays in context.
REDO_PROMPT = """That answer has a problem: {error}.
Write the complete dialogue again: about {words} words in around {lines} lines, with {a} (speaker "A") and {b} (speaker "B") taking turns.
Return only the JSON array."""


# --------------------------------------------------------------------------- helpers


class ScriptFormatError(ValueError):
    """The model's reply is not a usable script (broken JSON or wrong structure)."""


class ScriptContentError(ScriptFormatError):
    """The JSON is fine but the dialogue is not (too short, only one host)."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_STAGE_RE = re.compile(
    r"\[[^\]]*\]|\((?:[^)]*\b(?:laugh|chuckl|giggl|sigh|music|pause|smil|cough|beat|intro|outro)\w*[^)]*)\)",
    re.I,
)
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs (or sentences, for huge paragraphs) into chunks <= max_chars."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            while len(sentence) > max_chars:  # pathological: no sentence breaks
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def clean_line_text(text: str, names: dict[str, str]) -> str:
    text = _STAGE_RE.sub(" ", text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"[*_#`]+", "", text)
    # Drop a leading "Ava:" / "Host A:" label the model sometimes adds.
    labels = [re.escape(n) for n in names.values() if n] + [r"Host [AB]", r"[AB]"]
    text = re.sub(rf"^\s*(?:{'|'.join(labels)})\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _speaker_aliases(names: dict[str, str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for spk, name in names.items():
        for alias in (spk, f"host {spk}", f"host_{spk}", f"host{spk}", name):
            if alias:
                aliases[alias.lower()] = spk
    return aliases


def parse_lines(raw: str, names: dict[str, str], min_lines: int = 2) -> list[ScriptLine]:
    """Parse and validate the model's JSON. Raises ScriptFormatError with a clear reason."""
    raw = _FENCE_RE.sub("", strip_think(raw)).strip()
    if not raw:
        raise ScriptFormatError("the answer was empty")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScriptFormatError(f"it is not valid JSON ({exc.msg} at character {exc.pos})") from exc

    if isinstance(data, dict):  # tolerate {"lines": [...]} or similar wrappers
        data = next((v for v in data.values() if isinstance(v, list)), data)
    if not isinstance(data, list):
        raise ScriptFormatError("the top level must be a JSON array")

    aliases = _speaker_aliases(names)
    lines: list[ScriptLine] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ScriptFormatError(f"item {i} is not an object")
        speaker = aliases.get(str(item.get("speaker", "")).strip().lower())
        if speaker is None:
            raise ScriptFormatError(f'item {i} has speaker {item.get("speaker")!r}; use "A" or "B"')
        text = item.get("text")
        if not isinstance(text, str):
            raise ScriptFormatError(f'item {i} has no "text" string')
        text = clean_line_text(text, names)
        if text:
            lines.append(ScriptLine(speaker, text))

    if len(lines) < min_lines:
        raise ScriptContentError(f"it has only {len(lines)} lines, far too short")
    if len({line.speaker for line in lines}) < 2:
        raise ScriptContentError("only one host speaks; both A and B must talk")
    return lines


def salvage_truncated(raw: str) -> str | None:
    """If the JSON array was cut off mid-way, keep every complete object."""
    raw = _FENCE_RE.sub("", strip_think(raw)).strip()
    if not raw.startswith("["):
        return None
    end = len(raw)
    for _ in range(8):
        end = raw.rfind("}", 0, end)
        if end == -1:
            return None
        candidate = raw[: end + 1] + "]"
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def merge_same_speaker(lines: list[ScriptLine]) -> list[ScriptLine]:
    merged: list[ScriptLine] = []
    for line in lines:
        if merged and merged[-1].speaker == line.speaker:
            merged[-1] = ScriptLine(line.speaker, f"{merged[-1].text} {line.text}")
        else:
            merged.append(ScriptLine(line.speaker, line.text))
    return merged


# --------------------------------------------------------------------------- writer


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    output_seconds: float = 0.0
    wall_seconds: float = 0.0

    def summary(self) -> str:
        tps = self.output_tokens / self.output_seconds if self.output_seconds else 0.0
        return (
            f"{self.calls} LLM calls, {self.prompt_tokens} prompt tokens, "
            f"{self.output_tokens} output tokens at {tps:.1f} tok/s"
        )


@dataclass
class ScriptResult:
    lines: list[ScriptLine]
    outline: list[str] = field(default_factory=list)
    summarised: bool = False
    notes_chars: int = 0
    stats: LLMStats = field(default_factory=LLMStats)


class ScriptWriter:
    def __init__(
        self,
        llm: LLMSettings,
        script: ScriptSettings,
        cache: Cache | None = None,
        log: Callable[[str], None] = print,
    ):
        self.llm = llm
        self.cfg = script
        self.cache = cache
        self.log = log
        self.names = {"A": script.host_a_name, "B": script.host_b_name}
        self.client = Client(host=llm.host, timeout=httpx.Timeout(llm.request_timeout, connect=10))
        self._think: bool | None = llm.think
        self.system = SYSTEM_PROMPT.format(
            show=script.show_name, a=self.names["A"], b=self.names["B"]
        )

    # ---- readiness ------------------------------------------------------------
    def check_ready(self) -> None:
        """Fail early with a clear message if Ollama is down or the model is missing."""
        try:
            listing = self.client.list()
        except (ConnectionError, httpx.ConnectError, httpx.TimeoutException):
            raise LLMError(
                f"Can't reach Ollama at {self.llm.host}. Start the Ollama app "
                "(or run `ollama serve` in a terminal) and try again."
            ) from None
        names = {m.model for m in listing.models}
        wanted = self.llm.model
        if wanted not in names and f"{wanted}:latest" not in names:
            have = ", ".join(sorted(n for n in names if n)) or "none"
            raise LLMError(
                f"The model {wanted!r} is not downloaded. Run:  ollama pull {wanted}\n"
                f"(Models you have: {have}. You can change [llm] model in config.toml.)"
            )

    def unload(self) -> None:
        """Ask Ollama to free the model's RAM now (it reloads on the next request)."""
        try:
            self.client.generate(model=self.llm.model, keep_alive=0)
            self.log(f"  Unloaded {self.llm.model} from memory to give TTS more RAM")
        except Exception:
            pass

    # ---- low-level chat -------------------------------------------------------
    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        fmt: dict[str, Any] | None,
        num_predict: int,
        label: str,
        stats: LLMStats,
        on_tokens: Callable[[int], None] | None = None,
    ) -> tuple[str, str | None]:
        """Stream one chat call. Returns (content, done_reason)."""
        options = {
            "temperature": self.llm.temperature,
            "num_ctx": self.llm.num_ctx,
            "num_predict": num_predict,
        }
        start = time.perf_counter()
        try:
            stream = self.client.chat(
                model=self.llm.model,
                messages=messages,
                stream=True,
                think=self._think,
                format=fmt,
                options=options,
                keep_alive=self.llm.keep_alive,
            )
            pieces: list[str] = []
            last = None
            count = 0
            for part in stream:
                last = part
                if part.message.content:
                    pieces.append(part.message.content)
                    count += 1
                    if on_tokens and count % 16 == 0:
                        on_tokens(count)
        except ResponseError as exc:
            msg = str(exc.error)
            if self._think is not None and "think" in msg.lower():
                # This model has no thinking switch; retry without the flag.
                self._think = None
                return self._chat(
                    messages, fmt=fmt, num_predict=num_predict, label=label,
                    stats=stats, on_tokens=on_tokens,
                )
            if exc.status_code == 404:
                raise LLMError(
                    f"Ollama says model {self.llm.model!r} was not found. Run:  ollama pull {self.llm.model}"
                ) from None
            raise LLMError(f"Ollama returned an error during {label}: {msg}") from None
        except (ConnectionError, httpx.ConnectError):
            raise LLMError(
                f"Lost connection to Ollama at {self.llm.host} during {label}. Is it still running?"
            ) from None
        except httpx.TimeoutException:
            raise LLMError(
                f"Ollama took longer than {self.llm.request_timeout}s during {label}. "
                "Raise [llm] request_timeout in config.toml or use a smaller model."
            ) from None

        wall = time.perf_counter() - start
        stats.calls += 1
        stats.wall_seconds += wall
        if last is not None:
            stats.prompt_tokens += last.prompt_eval_count or 0
            stats.output_tokens += last.eval_count or 0
            stats.output_seconds += (last.eval_duration or 0) / 1e9
            tps = (last.eval_count or 0) / ((last.eval_duration or 1) / 1e9)
            self.log(
                f"    {label}: {last.eval_count or 0} tokens in {wall:.0f}s "
                f"({tps:.1f} tok/s, prompt {last.prompt_eval_count or 0} tokens)"
            )
        done_reason = last.done_reason if last is not None else None
        return strip_think("".join(pieces)), done_reason

    # ---- step 1: notes --------------------------------------------------------
    def prepare_notes(self, text: str, stats: LLMStats, progress: ProgressFn) -> tuple[str, bool]:
        text = normalize_text(text)
        limit = self.llm.chunk_chars
        if len(text) <= int(limit * 1.25):
            return text, False

        # Aim for combined notes of about `limit` characters (~6 chars per word).
        budget_words = max(150, int(limit * 1.25 / 6))
        notes = text
        for round_no in range(1, 4):
            chunks = split_into_chunks(notes, limit)
            # Each summary must be clearly shorter than its chunk, or rounds never converge.
            chunk_words = max(1, min(len(c.split()) for c in chunks))
            words = max(40, min(250, budget_words // len(chunks), int(chunk_words * 0.4)))
            self.log(f"  Summarising {len(chunks)} chunks (round {round_no}, ~{words} words each)")
            summaries = []
            for i, chunk in enumerate(chunks, start=1):
                base, span = 0.3 * (i - 1) / len(chunks), 0.3 / len(chunks)
                msg = f"Summarising chunk {i}/{len(chunks)}" + (f" (round {round_no})" if round_no > 1 else "")
                progress(base, msg)
                expected = words * 1.4  # rough tokens per summary
                summaries.append(self._summarise(
                    chunk, i, len(chunks), words, stats,
                    on_tokens=lambda c, b=base, s=span, m=msg: progress(b + s * min(c / expected, 0.95), m),
                ))
            previous, notes = notes, "\n\n".join(summaries)
            self.log(f"  Notes: {len(previous)} -> {len(notes)} characters")
            if len(notes) <= int(limit * 1.25):
                break
            if len(notes) > 0.8 * len(previous):
                self.log("  Summaries are not getting shorter; continuing with these notes")
                break
        return notes, True

    def _summarise(
        self, chunk: str, i: int, n: int, words: int, stats: LLMStats,
        on_tokens: Callable[[int], None] | None = None,
    ) -> str:
        key = make_key("summary", PROMPT_VERSION, self.llm.model, words, chunk)
        if self.cache and (hit := self.cache.get_text("summaries", key)):
            self.log(f"    chunk {i}/{n}: summary from cache")
            return hit
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": SUMMARY_PROMPT.format(i=i, n=n, chunk=chunk, words=words)},
        ]
        content, _ = self._chat(
            messages, fmt=None, num_predict=int(words * 2.5) + 100,
            label=f"summary {i}/{n}", stats=stats, on_tokens=on_tokens,
        )
        content = content.strip()
        if not content:
            raise LLMError(f"The model returned an empty summary for chunk {i}/{n}.")
        if self.cache:
            self.cache.put_text("summaries", key, content)
        return content

    # ---- step 2: outline ------------------------------------------------------
    def make_outline(self, notes: str, n: int, stats: LLMStats) -> list[str]:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": OUTLINE_PROMPT.format(notes=notes, n=n)},
        ]
        content, _ = self._chat(
            messages, fmt=OUTLINE_SCHEMA, num_predict=60 * n + 100, label="outline", stats=stats
        )
        try:
            items = [
                re.sub(r"^\s*(?:part\s*\d+\s*[:.)-]|\d+\s*[:.)-])\s*", "", str(x), flags=re.I).strip()
                for x in json.loads(content)
            ]
            items = [x for x in items if x]
        except (json.JSONDecodeError, TypeError):
            items = []
        if not items:
            self.log("    outline was unusable; using a generic plan")
            items = ["The main ideas, explained step by step."]
        while len(items) < n:  # pad if the model gave too few parts
            items.append("More detail, examples and why it matters.")
        return items[:n]

    # ---- step 3: dialogue -----------------------------------------------------
    def _part_prompt(
        self, notes: str, outline: list[str], index: int, words: int, so_far: list[ScriptLine]
    ) -> str:
        n = len(outline)
        if n == 1:
            position, what, plan = "single", "the dialogue for the full episode", ""
        else:
            position = "first" if index == 0 else "last" if index == n - 1 else "middle"
            what = (
                f"the dialogue for part {index + 1} of {n} of the episode. "
                f"This part covers: {outline[index]}"
            )
            plan = "EPISODE PLAN:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(outline)) + "\n\n"
        so_far_text = ""
        if so_far:
            recent = "\n".join(f"{self.names[l.speaker]}: {l.text}" for l in so_far[-4:])
            so_far_text = f"THE CONVERSATION SO FAR ENDS WITH:\n{recent}\n\n"
        return PART_PROMPT.format(
            notes=notes,
            plan=plan,
            so_far=so_far_text,
            what=what,
            words=words,
            lines=max(4, round(words / WORDS_PER_LINE)),
            position=POSITION_RULES[position].format(
                a=self.names["A"], b=self.names["B"], show=self.cfg.show_name
            ),
        )

    def _generate_lines(
        self, prompt: str, words: int, label: str, stats: LLMStats, progress: Callable[[float], None]
    ) -> list[ScriptLine]:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        expected_tokens = words * 1.4 + (words / WORDS_PER_LINE) * 12
        num_predict = int(expected_tokens * 1.8) + 200

        def on_tokens(count: int) -> None:
            progress(min(count / expected_tokens, 0.98))

        n_lines = max(4, round(words / WORDS_PER_LINE))
        min_lines = max(4, round(n_lines / 3))
        schema = lines_schema(max(4, round(n_lines * 0.6)))
        raw, reason = self._chat(
            messages, fmt=schema, num_predict=num_predict, label=label,
            stats=stats, on_tokens=on_tokens,
        )
        try:
            return parse_lines(raw, self.names, min_lines)
        except ScriptFormatError as first_error:
            error = first_error

        # Local fix first: a reply cut off at the token limit usually just needs closing.
        if not isinstance(error, ScriptContentError) and (fixed := salvage_truncated(raw)):
            try:
                lines = parse_lines(fixed, self.names, min_lines)
                self.log(f"    {label}: reply was cut off ({reason}); kept {len(lines)} complete lines")
                return lines
            except ScriptFormatError:
                pass

        snippet = re.sub(r"\s+", " ", raw.strip())[:160]
        self.log(f"    {label}: unusable reply ({error}). Start of reply: {snippet!r}")
        if isinstance(error, ScriptContentError) or len(raw.strip()) <= 40:
            # Valid JSON but bad dialogue (or nothing at all): ask again in the same
            # conversation, so the model still sees the source and its own attempt.
            self.log(f"    {label}: asking the model once more for the full dialogue")
            retry_messages = messages + [
                {"role": "assistant", "content": raw or "[]"},
                {
                    "role": "user",
                    "content": REDO_PROMPT.format(
                        error=error, words=words, lines=n_lines,
                        a=self.names["A"], b=self.names["B"],
                    ),
                },
            ]
        else:
            self.log(f"    {label}: retrying once with a JSON repair prompt")
            retry_messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": REPAIR_PROMPT.format(error=error, raw=raw[:12000])},
            ]
        raw2, _ = self._chat(
            retry_messages, fmt=schema, num_predict=num_predict,
            label=f"{label} (repair)", stats=stats, on_tokens=on_tokens,
        )
        for candidate in (raw2, salvage_truncated(raw2)):
            if candidate:
                try:
                    return parse_lines(candidate, self.names, min_lines)
                except ScriptFormatError as exc:
                    error = exc
        raise LLMError(
            f"The model did not return a valid script for {label}, even after a repair attempt "
            f"({error}). Try again, or set a bigger [llm] model in config.toml."
        )

    # ---- public entry point ---------------------------------------------------
    def write(
        self, text: str, length: str, progress: ProgressFn | None = None
    ) -> ScriptResult:
        progress = progress or (lambda f, m: None)
        if not text or not text.strip():
            raise LLMError("There is no text to turn into a script.")
        self.check_ready()
        stats = LLMStats()

        target = self.cfg.target_words(length)
        n_parts = max(1, round(target / max(150, self.llm.words_per_part)))
        words_per_part = round(target / n_parts)
        self.log(
            f"  Target: {length} = ~{target} words, {n_parts} part(s) of ~{words_per_part} words "
            f"(model {self.llm.model})"
        )

        progress(0.0, "Reading the source")
        notes, summarised = self.prepare_notes(text, stats, progress)
        base = 0.3 if summarised else 0.0
        source_words = len(normalize_text(text).split())
        if source_words < target * 0.6:
            self.log(
                f"  Note: the source has only ~{source_words} words but a {length} episode "
                f"needs ~{target}. A small model may repeat itself or invent details; "
                "a shorter length will be more accurate."
            )

        if n_parts > 1:
            progress(base, "Planning the episode")
            outline = self.make_outline(notes, n_parts, stats)
            for i, item in enumerate(outline, start=1):
                self.log(f"    part {i}: {item}")
        else:
            outline = ["The whole episode."]

        lines: list[ScriptLine] = []
        span = (1.0 - base) / n_parts
        for i in range(n_parts):
            label = f"part {i + 1}/{n_parts}" if n_parts > 1 else "script"
            start = base + i * span
            progress(start, f"Writing {label}")
            prompt = self._part_prompt(notes, outline, i, words_per_part, lines)
            part_lines = self._generate_lines(
                prompt, words_per_part, label, stats,
                lambda f, s=start, lab=label: progress(s + f * span, f"Writing {lab}"),
            )
            lines.extend(part_lines)

        lines = merge_same_speaker(lines)
        progress(1.0, "Script ready")
        return ScriptResult(
            lines=lines,
            outline=outline if n_parts > 1 else [],
            summarised=summarised,
            notes_chars=len(notes),
            stats=stats,
        )
