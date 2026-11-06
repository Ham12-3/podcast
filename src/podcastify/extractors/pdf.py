"""PDF files, read with PyMuPDF.

Two ways to read the text:

* layout mode (default, pymupdf4llm): understands the page layout, so two-column papers
  come out in reading order, words split by a hyphen at the end of a line are joined,
  and running headers, footers and page numbers are dropped. About 1 s per page on CPU.
* plain mode (pdf_layout = false, or if layout mode fails): much faster, rougher text.

Afterwards the text is tidied for the LLM: figure labels, tables, citation markers like
[12] and the References section are removed.
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import pymupdf

from ..errors import ExtractionError
from ..textutil import normalize_text, word_count
from . import Extracted, ExtractOptions

LAYOUT_BATCH = 8  # pages per layout call, so progress can be shown on long PDFs
MIN_WORDS_PER_TEXT_PAGE = 5

_REFERENCES_HEADING = re.compile(
    r"^\s*(?:[0-9IVX]+\.?\s+)?(references|bibliography|works cited|literature cited|"
    r"reference list|sources|citations)\s*$",
    re.I | re.M,
)
_BAD_META_TITLE = re.compile(r"^(untitled|title|document\d*|microsoft word\b.*|.*\.(docx?|pdf|tex|indd))$", re.I)


# --------------------------------------------------------------------- cleanup
def _clean_markdown(md: str, keep_tables: bool) -> str:
    """Turn pymupdf4llm Markdown into plain prose."""
    md = re.sub(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", "\n", md, flags=re.S | re.I)
    md = re.sub(r"<!--.*?-->", "\n", md, flags=re.S)                      # other HTML comments
    md = re.sub(r"^.*==>\s*picture.*<==.*$", "", md, flags=re.M | re.I)   # "picture omitted" notes
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)                         # images
    if keep_tables:
        md = re.sub(r"^\|?\s*:?-{3,}.*$", "", md, flags=re.M)             # table separator rows
    else:
        md = re.sub(r"^\s*\|.*\|\s*$", "", md, flags=re.M)                # table rows
    md = re.sub(r"<sup>.*?</sup>", "", md, flags=re.S | re.I)             # footnote markers
    md = re.sub(r"<br\s*/?>", " ", md, flags=re.I)
    md = re.sub(r"</?[a-zA-Z][^>]*>", "", md)                             # any other tags
    md = re.sub(r"\[([^\]]+)\]\((?:https?|mailto):[^)]*\)", r"\1", md)    # links -> text
    md = re.sub(r"^\s{0,3}#{1,6}\s*", "", md, flags=re.M)                 # heading marks
    md = re.sub(r"^\s*>\s?", "", md, flags=re.M)                          # block quotes
    md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.M)                      # bullets
    md = md.replace("**", "").replace("~~", "")
    md = re.sub(r"(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])", "", md)           # _italics_
    md = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", md)               # markdown escapes
    return md


def _tidy_prose(text: str) -> str:
    """Cleanup shared by both reading modes."""
    text = unicodedata.normalize("NFKC", text)   # ligatures (ﬁ -> fi), odd spaces
    text = text.replace("�", "")
    # numeric citations inside sentences: "deep nets [12]", "[3, 7]", "[4-6]"
    text = re.sub(r"(?<=\S)[ \t]*\[\d+(?:\s*[-–,]\s*\d+)*\]", "", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # drop lines that are mostly a URL (footnotes), just a number (axes, page
        # numbers) or a short line with an e-mail address (author details)
        urls = sum(len(u) for u in re.findall(r"(?:https?://|www\.)\S+", stripped))
        if stripped and (
            urls > 0.5 * len(stripped)
            or re.fullmatch(r"[\d.,%\s()+-]+", stripped)
            or (len(stripped) < 150 and re.search(r"@\s?[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", stripped))
        ):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)  # space left before punctuation
    return normalize_text(text)


def _drop_references(text: str, log) -> str:
    """Cut the References/Bibliography section (and anything after it, like appendices)."""
    matches = [m for m in _REFERENCES_HEADING.finditer(text) if m.start() > len(text) * 0.5]
    if not matches:
        return text
    cut = matches[0].start()
    dropped = word_count(text[cut:])
    log(f"  Skipped the {matches[0].group(1).title()} section and anything after it ({dropped} words)")
    return text[:cut].rstrip()


# --------------------------------------------------------------- read modes
def _plain_text(doc: pymupdf.Document) -> str:
    """Fast read: one paragraph per text block, lines joined, hyphenation undone."""
    paragraphs = []
    for page in doc:
        for block in page.get_text("blocks", sort=True):
            if block[6] != 0:  # 0 = text block, 1 = image
                continue
            para = re.sub(r"(\w)-\n(\w)", r"\1\2", block[4].strip())
            para = re.sub(r"\s*\n\s*", " ", para)
            if para:
                paragraphs.append(para)
    return "\n\n".join(paragraphs)


def _layout_text(doc: pymupdf.Document, pages: list[int], opts: ExtractOptions) -> str:
    import pymupdf4llm  # imported here: loads the layout model (onnxruntime)

    parts = []
    started = time.perf_counter()
    for i in range(0, len(pages), LAYOUT_BATCH):
        batch = pages[i : i + LAYOUT_BATCH]
        parts.append(
            pymupdf4llm.to_markdown(doc, pages=batch, header=False, footer=False, use_ocr=False)
        )
        done = i + len(batch)
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (len(pages) - done)
        opts.progress(done / len(pages), f"Reading PDF page {done}/{len(pages)} (about {eta:.0f}s left)")
    return _clean_markdown("\n\n".join(parts), opts.settings.keep_tables)


# -------------------------------------------------------------------- title
def _guess_title(doc: pymupdf.Document, path: Path) -> str:
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if 4 <= len(meta_title) <= 200 and not _BAD_META_TITLE.match(meta_title):
        return meta_title
    # Otherwise: the biggest horizontal text in the top half of the first page with text
    # (rotated text is skipped: arXiv prints a big sideways stamp in the margin).
    for page in list(doc)[:3]:
        spans = [
            (round(span["size"], 1), span["text"].strip())
            for block in page.get_text("dict")["blocks"] if block.get("type") == 0
            for line in block["lines"]
            if line["bbox"][1] < page.rect.height * 0.5 and abs(line["dir"][1]) < 0.01
            for span in line["spans"] if span["text"].strip()
        ]
        if not spans:
            continue
        biggest = max(size for size, _ in spans)
        title = re.sub(r"\s+", " ", " ".join(t for s, t in spans if s == biggest)).strip()
        if 4 <= len(title) <= 200 and re.search(r"[A-Za-z]{3}", title):
            return title
        break
    return path.stem.replace("_", " ").replace("-", " ")


# --------------------------------------------------------------------- main
def _open(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path)
    except pymupdf.EmptyFileError as exc:
        raise ExtractionError(f"The PDF file is empty (0 bytes): {path}") from exc
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ExtractionError(f"Could not open the PDF. It may be damaged or not really a PDF: {path}") from exc
    if doc.needs_pass:
        raise ExtractionError(
            f"The PDF is password-protected: {path}. Open it in a PDF viewer and save or print "
            "a copy without a password, then try that copy."
        )
    if doc.page_count == 0:
        raise ExtractionError(f"The PDF has no pages: {path}")
    return doc


def extract_pdf(source: str, opts: ExtractOptions | None = None) -> Extracted:
    opts = opts or ExtractOptions()
    log, settings = opts.log, opts.settings
    path = Path(source)
    doc = _open(path)
    try:
        # A quick plain read tells us which pages have real (selectable) text.
        page_words = [len(page.get_text("text").split()) for page in doc]
        text_pages = [i for i, n in enumerate(page_words) if n >= MIN_WORDS_PER_TEXT_PAGE]
        if not text_pages:
            raise ExtractionError(
                f"No text found in the PDF: {path}. It is probably scanned (pictures of pages) "
                "or empty. Podcastify needs a PDF with selectable text: run it through OCR first "
                "(for example 'Save as searchable PDF' in your scanner app), or use a text version."
            )
        empty = doc.page_count - len(text_pages)
        if empty:
            log(f"  Warning: {empty} of {doc.page_count} pages have no text (scanned images?) and were skipped")

        title = _guess_title(doc, path)
        text = ""
        if settings.pdf_layout:
            n = len(text_pages)
            log(f"  Reading {n} page{'s' * (n != 1)} with layout analysis (about {n}s on CPU)")
            try:
                text = _layout_text(doc, text_pages, opts)
            except Exception as exc:  # layout model problems should never block a run
                log(f"  Layout analysis failed ({type(exc).__name__}: {exc}); using plain text instead")
                text = ""
        if word_count(text) < 0.5 * sum(page_words[i] for i in text_pages):
            if settings.pdf_layout and text:
                log("  Layout analysis lost a lot of text; using plain text instead")
            text = _plain_text(doc)
    finally:
        doc.close()

    text = _tidy_prose(text)
    if settings.skip_references:
        text = _drop_references(text, log)
    if not text:
        raise ExtractionError(f"No usable text found in the PDF: {path}")
    return Extracted(text=text, title=title, source_type="pdf", source=str(path.resolve()))
