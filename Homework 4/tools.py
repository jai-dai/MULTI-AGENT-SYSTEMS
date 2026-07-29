"""Tools for the research agent.

Unlike homework-lesson-3, there is no `@tool` decorator here: every tool is a plain
Python function, and its contract for the LLM is a hand-written JSON Schema in
``TOOL_SCHEMAS`` (OpenAI tool-calling format). ``dispatch()`` maps a tool call coming
back from the API onto the actual function.

Every tool returns a string. Failures come back as ``ERROR: ...`` text instead of
exceptions, so a failure becomes context the model can react to.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from trafilatura.settings import use_config

from config import settings

logging.getLogger("trafilatura").setLevel(logging.ERROR)

_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(settings.request_timeout))

_MAX_SNIPPET_LENGTH = 400

# Below this, an "article" is almost certainly an abstract or a landing page.
_SHORT_PAGE_THRESHOLD = 1200


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _truncate(text: str, limit: int) -> str:
    """Cut text to `limit` characters and tell the model that it was cut."""
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[TRUNCATED: showed first {limit} of {len(text)} characters]"
    )


def _output_dir() -> Path:
    path = Path(__file__).parent / settings.output_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_report_path(filename: str) -> Path:
    """Map an arbitrary model-supplied filename onto a safe path inside output/."""
    name = Path(str(filename).strip()).name  # drops "../", absolute paths, subdirs
    if not name:
        name = "report.md"
    if not name.endswith(".md"):
        name += ".md"
    return _output_dir() / name


# --------------------------------------------------------------------------- #
# tool implementations
# --------------------------------------------------------------------------- #


def web_search(query: str, max_results: int = settings.max_search_results) -> str:
    query = str(query).strip()
    if not query:
        return "ERROR: empty query."

    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = settings.max_search_results

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # network, rate limit, backend change
        return (
            f"ERROR: search failed ({type(exc).__name__}: {exc}). "
            "Try again with a different query."
        )

    if not results:
        return f"No results for '{query}'. Try different keywords."

    lines = [f"Search results for '{query}':"]
    for i, item in enumerate(results, start=1):
        title = item.get("title", "").strip()
        url = item.get("href", "").strip()
        snippet = _truncate(item.get("body", "").strip(), _MAX_SNIPPET_LENGTH)
        lines.append(f"\n{i}. {title}\n   URL: {url}\n   Snippet: {snippet}")
    return "\n".join(lines)


def _fetch_text(url: str) -> tuple[str | None, str | None]:
    """Download + extract one URL. Returns (text, error_message)."""
    try:
        downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CONFIG)
    except Exception as exc:
        return None, f"could not download {url} ({type(exc).__name__}: {exc})"

    if not downloaded:
        return None, f"page {url} is unavailable (timeout, 403/404, or blocked)"

    try:
        text = trafilatura.extract(
            downloaded, url=url, include_comments=False, include_tables=True
        )
    except Exception as exc:
        return None, f"could not extract text from {url} ({type(exc).__name__}: {exc})"

    if not text or not text.strip():
        return None, f"no readable text extracted from {url} (JS-heavy page or PDF)"

    return text.strip(), None


def read_url(url: str) -> str:
    url = str(url).strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"ERROR: '{url}' is not a valid http(s) URL."

    # An arXiv /abs/ page is only the abstract. Try the full-text HTML rendering first
    # and fall back to the abstract page if the paper has no HTML version.
    fetched_url = url
    text, error = None, None
    abs_match = re.match(r"(https?://arxiv\.org)/abs/(\S+)", url, flags=re.IGNORECASE)
    if abs_match:
        html_url = f"{abs_match.group(1)}/html/{abs_match.group(2)}"
        text, _ = _fetch_text(html_url)
        if text:
            fetched_url = html_url

    if text is None:
        text, error = _fetch_text(url)

    if text is None:
        return f"ERROR: {error}. Use another source."

    note = ""
    if abs_match and fetched_url == url:
        # We asked for the HTML full text and did not get it.
        note = (
            "\n\n[ABSTRACT ONLY: arXiv has no HTML full text for this paper, so the above "
            "is the abstract page. Do not describe it as the full paper — cite it as an "
            "abstract, or find the full text elsewhere.]"
        )
    elif len(text) < _SHORT_PAGE_THRESHOLD:
        note = (
            f"\n\n[SHORT PAGE: only {len(text)} characters extracted. This is likely an "
            "abstract, a paywall or a landing page rather than the full text — treat it "
            "as a pointer, not as evidence, and look for the full version.]"
        )

    body = _truncate(text, settings.max_url_content_length)
    return f"Content of {fetched_url}:\n\n{body}{note}"


def _missing_report_sections(content: str) -> list[str]:
    """Which required sections the report is missing (EN / UA / RU headings)."""
    required = {
        "Conclusions": r"^#{1,4}\s*\d*\.?\s*(conclusions?|висновк\w*|выводы)\b",
        "Sources": r"^#{1,4}\s*\d*\.?\s*(sources?|references?|джерела|источники)\b",
    }
    return [
        name
        for name, pattern in required.items()
        if not re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    ]


def write_report(filename: str, content: str) -> str:
    content = str(content) if content is not None else ""
    if not content.strip():
        return "ERROR: refusing to write an empty report."

    # Contract check: the model is told which sections a report must have, but only a
    # check in code actually guarantees it. Rejected reports come back as an ERROR the
    # model can fix, which also forces it to rewrite the file as a whole instead of
    # patching it (patching is what produced sections numbered 9, 11, 10).
    missing = _missing_report_sections(content)
    if missing:
        return (
            f"ERROR: report is missing required section(s): {', '.join(missing)}. "
            "Rewrite the FULL report with every required section present and the "
            "section numbering consecutive, then call write_report again."
        )

    path = _safe_report_path(filename)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not write {path} ({exc})."
    return f"Report saved to {path} ({len(content)} characters)."


def list_reports() -> str:
    files = sorted(_output_dir().glob("*.md"))
    if not files:
        return "No reports saved yet."
    return "Saved reports:\n" + "\n".join(
        f"- {f.name} ({f.stat().st_size} bytes)" for f in files
    )


def read_report(filename: str) -> str:
    path = _safe_report_path(filename)
    if not path.exists():
        return f"ERROR: {path.name} not found. Use list_reports to see available files."
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not read {path} ({exc})."
    return f"Content of {path.name}:\n\n{_truncate(text, settings.max_url_content_length)}"


# --------------------------------------------------------------------------- #
# JSON Schema definitions — this is what the model actually sees
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web with DuckDuckGo. Returns a ranked list of results with "
                "title, URL and a 1-2 sentence snippet. Snippets are NOT evidence: use "
                "them only to decide which pages are worth reading with read_url."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Short, specific English keywords work far "
                            "better than a full natural-language question."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return.",
                        "minimum": 1,
                        "maximum": 10,
                        "default": settings.max_search_results,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Fetch the main text of a web page (navigation, ads and boilerplate "
                "stripped). Long pages are truncated and marked with '[TRUNCATED ...]'. "
                "Use this on URLs found by web_search to get actual facts. For arXiv "
                "links the full-text HTML version is fetched automatically when it "
                "exists; if the result is marked '[ABSTRACT ONLY]' or '[SHORT PAGE]', "
                "you have not read the full text and must not present it as such."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full http(s) URL of the page to read.",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": (
                "Save the final Markdown report to the output directory. Call this once "
                "the research is done, with the complete report as content. The report "
                "must contain a Conclusions section and a Sources section (English, "
                "Ukrainian or Russian headings) — otherwise the call is rejected. Always "
                "pass the FULL report text: writing is a whole-file overwrite, not a patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name, e.g. 'rag_approaches.md'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full report body in Markdown.",
                    },
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reports",
            "description": "List Markdown reports already saved in the output directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_report",
            "description": (
                "Read back a report saved earlier, so it can be extended or revised "
                "instead of rewritten from scratch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of a file returned by list_reports.",
                    }
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
]

# name -> implementation
TOOL_REGISTRY = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
    "list_reports": list_reports,
    "read_report": read_report,
}


def dispatch(name: str, raw_arguments: str) -> str:
    """Execute one tool call coming from the API.

    `raw_arguments` is the JSON string produced by the model. Everything that can go
    wrong here — unknown tool, malformed JSON, wrong argument names, exception inside
    the tool — is converted into an ERROR string so the loop can continue.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOL_REGISTRY)}."

    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"ERROR: arguments for '{name}' are not valid JSON ({exc})."

    if not isinstance(arguments, dict):
        return f"ERROR: arguments for '{name}' must be a JSON object."

    try:
        return func(**arguments)
    except TypeError as exc:  # wrong / missing argument names
        return f"ERROR: bad arguments for '{name}' ({exc})."
    except Exception as exc:  # anything unexpected inside the tool
        return f"ERROR: tool '{name}' crashed ({type(exc).__name__}: {exc})."
