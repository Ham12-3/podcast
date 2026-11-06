"""Blog posts and articles from a URL.

The page is downloaded with httpx (so every failure gets a clear message), then
trafilatura finds the main article text and drops menus, ads, footers and comments.
A link that points straight to a PDF is handed to the PDF extractor.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura

from ..errors import ExtractionError
from ..textutil import normalize_text, word_count
from . import Extracted, ExtractOptions

# A normal browser-like User-Agent: some sites refuse requests that look like scripts.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
)
MAX_BYTES = 80 * 1024 * 1024
LOW_WORDS = 150  # below this, try a second, more generous extraction pass

# Headings of "read next" blocks that some sites put right after the article.
_TAIL_HEADING = re.compile(
    r"^(more recent articles|recent (posts|articles)|related (posts|articles|stories|reading)|"
    r"you (might|may) also (like|enjoy)|read (more|next)|more from .{1,40}|popular (posts|articles)|"
    r"more (posts|articles|stories)|share this( post| article)?:?|subscribe.{0,40})$",
    re.I,
)

_SAVE_AS_PDF_TIP = (
    "Tip: open the page in your browser, press Ctrl+P and choose 'Save as PDF', "
    "then give Podcastify that PDF file instead."
)


def _download(url: str, timeout: int, log) -> tuple[bytes, str, str]:
    """Return (body, content_type, final_url) or raise a friendly ExtractionError."""
    host = urlparse(url).hostname or url
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"}
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(timeout), headers=headers) as client:
            with client.stream("GET", url) as resp:
                status = resp.status_code
                if status in (404, 410):
                    raise ExtractionError(f"Page not found (HTTP {status}): {url}. Check the address.")
                if status in (401, 403, 451):
                    raise ExtractionError(
                        f"The site refused the download (HTTP {status}): {url}. Some sites block apps "
                        f"or need a login. {_SAVE_AS_PDF_TIP}"
                    )
                if status == 429:
                    raise ExtractionError(f"{host} is limiting requests (HTTP 429). Wait a few minutes and try again.")
                if status >= 500:
                    raise ExtractionError(f"{host} had a server problem (HTTP {status}). Try again later.")
                if status >= 400:
                    raise ExtractionError(f"Could not download {url} (HTTP {status}).")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        raise ExtractionError(f"The page at {url} is too large (over {MAX_BYTES // 2**20} MB).")
                ctype = resp.headers.get("content-type", "").lower()
                final_url = str(resp.url)
    except ExtractionError:
        raise
    except (httpx.UnsupportedProtocol, httpx.InvalidURL) as exc:
        raise ExtractionError(f"That is not a valid web address: {url}") from exc
    except httpx.TimeoutException as exc:
        raise ExtractionError(f"{host} did not respond within {timeout} seconds. Check your internet connection or try again.") from exc
    except httpx.TooManyRedirects as exc:
        raise ExtractionError(f"{url} redirects in a loop, so it can't be downloaded.") from exc
    except httpx.ConnectError as exc:
        msg = str(exc)
        if "getaddrinfo" in msg or "11001" in msg or "Name or service not known" in msg or "nodename" in msg:
            raise ExtractionError(
                f"Could not find the website '{host}'. Check the spelling of the address and that you are online."
            ) from exc
        if "CERTIFICATE" in msg.upper():
            raise ExtractionError(f"The security certificate of {host} could not be verified, so it was not downloaded.") from exc
        raise ExtractionError(f"Could not connect to {host}: {msg}") from exc
    except httpx.HTTPError as exc:
        raise ExtractionError(f"Download of {url} failed: {type(exc).__name__}: {exc}") from exc
    log(f"  Downloaded {len(body) / 1024:.0f} KB from {host}")
    return bytes(body), ctype, final_url


def _from_pdf(body: bytes, url: str, opts: ExtractOptions) -> Extracted:
    from .pdf import extract_pdf

    folder = opts.work_dir or Path(tempfile.gettempdir()) / "podcastify"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.pdf"
    path.write_bytes(body)
    opts.log("  The link is a PDF file; reading it as a PDF")
    result = extract_pdf(str(path), opts)
    result.source = url
    return result


def _article(html: bytes, url: str, keep_tables: bool):
    common = dict(
        url=url, with_metadata=True, include_comments=False, include_tables=keep_tables,
        include_images=False, include_links=False, include_formatting=False, deduplicate=True,
    )
    doc = trafilatura.bare_extraction(html, **common)
    if doc is None or word_count(doc.text or "") < LOW_WORDS:
        # Second try: keep more of the page (helps on unusual layouts).
        generous = trafilatura.bare_extraction(html, favor_recall=True, **common)
        if generous is not None and word_count(generous.text or "") > word_count((doc.text if doc else "") or ""):
            doc = generous
    return doc


def _trim_tail(text: str, log) -> str:
    """Drop a trailing 'More recent articles' style list of links."""
    lines = text.split("\n")
    start = int(len(lines) * 0.7)
    for i in range(start, len(lines)):
        if _TAIL_HEADING.match(lines[i].strip()) and all(len(l) < 200 for l in lines[i + 1 :]):
            log(f"  Skipped a '{lines[i].strip()}' list at the end of the page")
            return "\n".join(lines[:i]).rstrip()
    return text


def extract_web(source: str, opts: ExtractOptions | None = None) -> Extracted:
    opts = opts or ExtractOptions()
    url = source.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or "." not in parsed.hostname:
        raise ExtractionError(f"That is not a valid web address: {url}")

    opts.progress(0.1, f"Downloading {parsed.hostname}")
    body, ctype, final_url = _download(url, opts.settings.web_timeout, opts.log)

    if "pdf" in ctype or body[:5] == b"%PDF-":
        return _from_pdf(body, url, opts)
    if ctype.startswith("text/plain"):
        text = normalize_text(body.decode("utf-8", errors="replace"))
        title = Path(parsed.path).stem or parsed.hostname
        return Extracted(text=text, title=title, source_type="web", source=url)
    looks_like_html = "html" in ctype or "xml" in ctype or body.lstrip()[:1] == b"<"
    if not looks_like_html:
        kind = ctype.split(";")[0] or "unknown"
        raise ExtractionError(f"The link points to a file of type '{kind}', not a web page or PDF: {url}")

    opts.progress(0.6, "Finding the article text")
    doc = _article(body, final_url, opts.settings.keep_tables)
    text = _trim_tail(normalize_text((doc.text if doc else "") or ""), opts.log)
    if not text:
        raise ExtractionError(
            f"Couldn't find any article text on {url}. The page may need JavaScript to load, "
            f"or sit behind a login or paywall. {_SAVE_AS_PDF_TIP}"
        )
    title = (doc.title or doc.sitename or parsed.hostname or "web page").strip()
    opts.log(f"  Article: {word_count(text)} words" + (f" by {doc.author}" if doc.author else ""))
    return Extracted(text=text, title=title, source_type="web", source=url)
