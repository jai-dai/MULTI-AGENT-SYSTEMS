"""Tools available to the research agent.

Every tool is defensive: it returns a readable ``ERROR: ...`` string instead of
raising, so a failure becomes context the model can react to (retry with other
arguments, or continue without that source) rather than a crashed run.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool
from trafilatura.settings import use_config

from config import settings

# trafilatura is chatty about every failed download; keep the REPL readable.
logging.getLogger("trafilatura").setLevel(logging.ERROR)

_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(settings.request_timeout))

# A search snippet longer than this adds noise, not signal.
_MAX_SNIPPET_LENGTH = 400


def _truncate(text: str, limit: int) -> str:
    """Cut text to ``limit`` characters and tell the model that it was cut."""
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[TRUNCATED: showed first {limit} of {len(text)} characters]"
    )


def _output_dir() -> Path:
    """Resolve (and create) the reports directory next to this file."""
    path = Path(__file__).parent / settings.output_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_report_path(filename: str) -> Path:
    """Map an arbitrary model-supplied filename onto a safe path inside output/."""
    name = Path(filename.strip()).name  # drops "../", absolute paths, subdirs
    if not name:
        name = "report.md"
    if not name.endswith(".md"):
        name += ".md"
    return _output_dir() / name


@tool
def web_search(query: str, max_results: int = settings.max_search_results) -> str:
    """Search the web with DuckDuckGo.

    Returns a ranked list of results with title, URL and a short snippet.
    Snippets are 1-2 sentences only — use `read_url` on a result to get the full text.

    Args:
        query: Search query. Short, specific keywords work better than full sentences.
        max_results: How many results to return (1-10).
    """
    query = query.strip()
    if not query:
        return "ERROR: empty query."

    max_results = max(1, min(max_results, 10))

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # network issues, rate limits, backend changes
        return f"ERROR: search failed ({type(exc).__name__}: {exc}). Try again with a different query."

    if not results:
        return f"No results for '{query}'. Try different keywords."

    lines = [f"Search results for '{query}':"]
    for i, item in enumerate(results, start=1):
        title = item.get("title", "").strip()
        url = item.get("href", "").strip()
        snippet = _truncate(item.get("body", "").strip(), _MAX_SNIPPET_LENGTH)
        lines.append(f"\n{i}. {title}\n   URL: {url}\n   Snippet: {snippet}")

    return "\n".join(lines)


@tool
def read_url(url: str) -> str:
    """Read the main text content of a web page.

    Strips navigation, ads and boilerplate, and truncates long pages to keep the
    context window usable. Use this after `web_search` to actually study a source.

    Args:
        url: Full http(s) URL of the page to read.
    """
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"ERROR: '{url}' is not a valid http(s) URL."

    try:
        downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CONFIG)
    except Exception as exc:
        return f"ERROR: could not download {url} ({type(exc).__name__}: {exc})."

    if not downloaded:
        return f"ERROR: page {url} is unavailable (timeout, 403/404, or blocked). Use another source."

    try:
        text = trafilatura.extract(
            downloaded,
            url=url,
            include_comments=False,
            include_tables=True,
        )
    except Exception as exc:
        return f"ERROR: could not extract text from {url} ({type(exc).__name__}: {exc})."

    if not text or not text.strip():
        return f"ERROR: no readable text extracted from {url} (JS-heavy page or PDF). Use another source."

    return f"Content of {url}:\n\n{_truncate(text.strip(), settings.max_url_content_length)}"


@tool
def write_report(filename: str, content: str) -> str:
    """Save the final Markdown report to a file in the output directory.

    Call this once the research is done, with the complete report as `content`.

    Args:
        filename: File name, e.g. "rag_approaches.md" (".md" is added if missing).
        content: Full report body in Markdown.
    """
    if not content or not content.strip():
        return "ERROR: refusing to write an empty report."

    path = _safe_report_path(filename)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not write {path} ({exc})."

    return f"Report saved to {path} ({len(content)} characters)."


@tool
def list_reports() -> str:
    """List the Markdown reports already saved in the output directory."""
    files = sorted(_output_dir().glob("*.md"))
    if not files:
        return "No reports saved yet."
    return "Saved reports:\n" + "\n".join(
        f"- {f.name} ({f.stat().st_size} bytes)" for f in files
    )


@tool
def read_report(filename: str) -> str:
    """Read back a report saved earlier, to extend or revise it.

    Args:
        filename: Name of a file listed by `list_reports`.
    """
    path = _safe_report_path(filename)
    if not path.exists():
        return f"ERROR: {path.name} not found. Use list_reports to see available files."

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not read {path} ({exc})."

    return f"Content of {path.name}:\n\n{_truncate(text, settings.max_url_content_length)}"


TOOLS = [web_search, read_url, write_report, list_reports, read_report]
