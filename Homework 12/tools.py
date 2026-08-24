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
# The library's default user agent is blocked by a fair number of sites (403).
_TRAFILATURA_CONFIG.set(
    "DEFAULT",
    "USER_AGENTS",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)

_MAX_SNIPPET_LENGTH = 400

# Сколько писем одного контрагента показывать в ранжированном дайджесте.
PER_COUNTERPARTY_LIMIT = 2

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


def knowledge_search(query: str, top_n: int = settings.rerank_top_n,
                     source: str | None = None,
                     correspondent: str | None = None,
                     since: str | None = None, until: str | None = None) -> str:
    """Hybrid search over the local knowledge base (see retriever.py)."""
    query = str(query).strip()
    if not query:
        return "ERROR: empty query."

    try:
        top_n = max(1, min(int(top_n), 10))
    except (TypeError, ValueError):
        top_n = settings.rerank_top_n

    # Imported lazily: the index and the cross-encoder cost seconds to load,
    # and a session that never searches the base should not pay for them.
    try:
        from retriever import retrieve
    except ImportError as exc:
        return f"ERROR: retrieval module unavailable ({exc})."

    source = str(source).strip() if source else None
    correspondent = str(correspondent).strip() if correspondent else None
    since = str(since).strip() if since else None
    until = str(until).strip() if until else None
    try:
        found = retrieve(query, top_n=top_n, source=source,
                         correspondent=correspondent, since=since, until=until)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR: knowledge search failed ({type(exc).__name__}: {exc})."

    results = found["results"]
    if not results:
        if source and not found.get("matched_files"):
            # Distinct from "nothing matched the query": no FILE matched the
            # filter, so the query was never asked. Reporting them the same way
            # would teach the model that the base has no answer when it does.
            return (f"ERROR: no file in the knowledge base has '{source}' in its "
                    "name, so nothing was searched. Drop the source filter or "
                    "use a different fragment of the filename.")
        scope = f" in files matching '{source}'" if source else ""
        if since or until:
            scope += f" dated {since or '…'}..{until or '…'}"
        return (f"No passages in the knowledge base match '{query}'{scope}. "
                "Try different wording, a wider date range, or search the web "
                "instead.")

    stages = found["stages"]
    header = (f"{len(results)} passage(s) for '{query}' "
              f"(semantic {stages['semantic']} + BM25 {stages['bm25']} "
              f"→ {stages['fused']} fused → reranked)")
    if not found["confident"]:
        header += ("\nNOTE: every passage scored BELOW the relevance threshold. "
                   "The reranker may simply have missed the wording (it does not "
                   "expand acronyms). Treat these as weak evidence and verify.")

    blocks = []
    for i, r in enumerate(results, start=1):
        origin = "+".join(name for name, on in
                          (("semantic", r["in_semantic"]), ("bm25", r["in_bm25"])) if on)
        marker = " OCR" if r.get("ocr") else ""
        marker += " WEAK" if r.get("weak") else ""
        # Мова, якої реранкер не оцінює: пасаж лишається у видачі, але агент
        # мусить знати, що його місце в списку — не оцінка релевантності.
        note = f"\n  ⚠️  {r['language_note']}" if r.get("language_note") else ""
        # The date carries where it came from, because the two are not equally
        # trustworthy: "document" is what the author saved, "mail" is when it
        # was sent, "filesystem" is only when the file reached this disk — which
        # for a copied folder says nothing about the document at all.
        dated = (f" {r['date']} ({r['date_source']})"
                 if r.get("date") else "")
        if r.get("mail_date") and r["mail_date"] != r.get("date"):
            dated += f", sent {r['mail_date']}"
        # Письмо и документ цитируются по-разному, потому что «страница 2» у
        # письма означает второе письмо в цепочке, а не вторую страницу.
        if r.get("kind") == "mail":
            head = f"[письмо: {r['source']} #{r['page']} в цепочке]"
        else:
            head = f"[{r['source']} p.{r['page']}]"
            if r.get("kind") == "attachment":
                head += " (вложение письма)"
        blocks.append(
            f"\n{i}. {head}{dated}{marker} "
            f"score={r['score']} via {origin}{note}\n"
            f"{_truncate(r['text'], settings.max_url_content_length)}")
    return header + "\n" + "\n".join(blocks)


def list_mail(since: str | None = None, until: str | None = None,
              correspondent: str | None = None, limit: int = 20,
              rank: bool = False) -> str:
    """Перечислить письма за период — не поиск, а инвентаризация.

    Отдельный инструмент рядом с `knowledge_search`, потому что это другая
    операция. Векторный поиск отвечает «что похоже на вопрос» и возвращает
    top-K; спросить его «что приходило вчера» нельзя — письмо без словесных
    совпадений не попадёт в выдачу, даже если оно единственное за день.
    """
    try:
        from mailprep import store
    except ImportError as exc:
        return f"ERROR: mail module unavailable ({exc})."

    db = Path(__file__).parent / settings.mail_db
    if not db.exists():
        return (f"ERROR: no mail database at {db} — run "
                "`python -m mailprep.imap_fetch` first.")
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 20

    try:
        connection = store.connect(db)
        # При ранжировании читается ВЕСЬ период, а не limit последних писем.
        # Иначе сортировка врёт: в первой версии окно было limit*10, и июньская
        # цепочка с юристами не попала в топ просто потому, что не поместилась в
        # полсотни свежих. Отсортировать можно только то, что прочитал.
        rows = store.list_messages(
            connection, since=since, until=until,
            correspondent=(correspondent or "").strip(),
            limit=10_000 if rank else limit)
    except Exception as exc:
        return f"ERROR: listing mail failed ({type(exc).__name__}: {exc})."

    ranking = None
    if rank:
        import importance
        rules = importance.load()
        if not rules:
            return ("ERROR: ranking needs weights.yaml, and it is missing or "
                    "unreadable. Call list_mail without rank to get messages "
                    "by date.")
        wrote_to = {address.lower() for row in connection.execute(
            "SELECT to_json FROM messages WHERE LOWER(sender_email) = LOWER(?)",
            (settings.imap_user,))
            for entry in json.loads(row["to_json"] or "[]")
            if (address := entry.get("email"))}
        rows = store.collapse_threads(rows)
        ranking = [(importance.score(row, rules, wrote_to), row) for row in rows]
        ranking = [pair for pair in ranking if not pair[0]["noise"]]
        ranking.sort(key=lambda pair: -pair[0]["score"])

        # Не больше двух писем одного контрагента. Замер на живой скриньке: у
        # контрагента с весом 97 и восемнадцатью письмами получилось 17 строк из
        # 20, и обзор «что важного» превратился в «письма от одного человека».
        # Веса при этом были верные — ломался жанр: дайджест должен показывать
        # РАЗНОЕ важное, а не самое важное много раз.
        shown: dict[str, int] = {}
        hidden: dict[str, int] = {}
        kept = []
        for verdict, row in ranking:
            key = verdict["counterparty"] or (row["sender_email"] or "?")
            if shown.get(key, 0) >= PER_COUNTERPARTY_LIMIT:
                hidden[key] = hidden.get(key, 0) + 1
                continue
            if len(kept) >= limit:
                break
            shown[key] = shown.get(key, 0) + 1
            kept.append((verdict, row))
        ranking = kept
        rows = [row for _, row in ranking]

    period = " ".join(filter(None, [f"since {since}" if since else "",
                                    f"until {until}" if until else "",
                                    f"correspondent '{correspondent}'" if correspondent else ""]))
    if not rows:
        return (f"No messages in the local mail database{' for ' + period if period else ''}. "
                "The database is a SNAPSHOT — it holds what the last IMAP fetch "
                "brought in, not live mail.")

    order = ("by importance (see weights.yaml), threads collapsed"
             if ranking else "newest first")
    lines = [f"{len(rows)} message(s){' for ' + period if period else ''}, {order}. "
             "This is the local snapshot, not live mail."]
    for index, row in enumerate(rows):
        who = row["sender_name"] or row["sender_email"]
        mark = ""
        if ranking:
            verdict = ranking[index][0]
            thread = row.get("messages_in_thread", 1)
            key = verdict["counterparty"] or (row["sender_email"] or "?")
            # Свёрнутое показывается на ПОСЛЕДНЕЙ строке контрагента: иначе
            # отсечённое исчезает молча, а это ровно то, чего дайджест делать
            # не должен — он обязан сказать, чего в нём нет.
            rest = hidden.pop(key, 0) if index == len(rows) - 1 or (
                index + 1 < len(rows)
                and (ranking[index + 1][0]["counterparty"]
                     or rows[index + 1]["sender_email"]) != key) else 0
            # Балл идёт вместе с причинами. Число важности без объяснения
            # непроверяемо: его нельзя ни оспорить, ни поправить, и модель
            # начнёт выдавать его за факт вместо настройки в файле.
            mark = (f" | importance {verdict['score']}"
                    + (f", {thread} messages in thread" if thread > 1 else "")
                    + f"\n  why: {'; '.join(verdict['reasons'])}"
                    + (f"\n  … ещё {rest} от этого же отправителя не показано"
                       if rest else ""))
        head = (f"\n[{row['date'][:16]}] {who} <{row['sender_email']}> "
                f"→ {', '.join(row['to_emails']) or '—'}{mark}\n  {row['subject']}")
        if row["attachments"]:
            # Имена вложений — мост между двумя индексами: сами файлы лежат в
            # индексе документов, и по имени их достаёт knowledge_search.
            head += ("\n  attachments (searchable by name via knowledge_search "
                     f"source=…): {', '.join(row['attachments'])}")
        lines.append(head)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# JSON Schema definitions — this is what the model actually sees
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "Search the LOCAL knowledge base of ingested documents: hybrid "
                "retrieval (semantic embeddings + BM25 keywords), merged and "
                "reranked by a cross-encoder. Returns passages with their source "
                "file and page number, which you may cite as [file.pdf p.N]. "
                "Prefer this over web_search whenever the ingested documents "
                "could plausibly cover the question — it is cheaper and its "
                "sources are precise. Phrase the query the way the DOCUMENTS "
                "would, and expand acronyms: 'retrieval-augmented generation' "
                "retrieves what 'RAG' alone can miss."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for, in the vocabulary of the documents. "
                            "One sub-topic per call."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many passages to return after reranking.",
                        "minimum": 1,
                        "maximum": 10,
                        "default": settings.rerank_top_n,
                    },
                    "correspondent": {
                        "type": "string",
                        "description": (
                            "Optional: restrict to messages where this fragment "
                            "appears in the sender, To or Cc — an address or a "
                            "domain, e.g. 'l*w-l*n', 'a**n.ua', 'p******a'. Use "
                            "it for questions about WHO a message involves; the "
                            "`query` still has to say WHAT it is about. A "
                            "correspondent filter with a vague query returns the "
                            "right messages with low scores, because the "
                            "reranker judges topical relevance, not metadata."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Optional: restrict the search to files whose NAME "
                            "contains this fragment, case-insensitive — e.g. "
                            "'invoice', 'EDCF', 'payroll'. Use it when the "
                            "question is clearly about one kind of document; "
                            "it is applied before scoring, so those files get "
                            "the whole result budget instead of competing with "
                            "the entire corpus. Omit it when unsure — a filter "
                            "that matches no file returns nothing at all."
                        ),
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Optional: only passages dated on or after this day, "
                            "YYYY-MM-DD. The date is the document's own where it "
                            "has one, the message date for mail. Passages with "
                            "no date are excluded by any date filter — they "
                            "cannot prove they belong in the period."
                        ),
                    },
                    "until": {
                        "type": "string",
                        "description": (
                            "Optional: only passages dated on or before this day, "
                            "YYYY-MM-DD (inclusive)."
                        ),
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
            "name": "list_mail",
            "description": (
                "LIST messages from the local mail snapshot, newest first, with "
                "sender, recipients, subject and attachment filenames. This is "
                "enumeration, not search: use it for 'what arrived yesterday', "
                "'what did I send to X last week', 'which documents came by "
                "mail today'. knowledge_search cannot answer those — it ranks by "
                "similarity to a query and returns only the top few, so a "
                "message that shares no wording with the question is invisible "
                "to it even when it is the only message of that day. "
                "The database is a SNAPSHOT taken by the last IMAP fetch, not "
                "live mail: say so when the answer depends on freshness. "
                "Attachment filenames returned here are searchable with "
                "knowledge_search(source=…), which is how you get from a "
                "message to the contents of what it carried."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "Only messages on or after this day, YYYY-MM-DD.",
                    },
                    "until": {
                        "type": "string",
                        "description": "Only messages on or before this day, YYYY-MM-DD.",
                    },
                    "correspondent": {
                        "type": "string",
                        "description": (
                            "Optional: fragment of an address or domain that must "
                            "appear in sender, To or Cc — e.g. 'a**n.ua'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many messages to return (default 20).",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "rank": {
                        "type": "boolean",
                        "description": (
                            "Sort by IMPORTANCE instead of date, collapsing each "
                            "thread to one line, and show why each score came "
                            "out as it did. Importance comes from weights.yaml, "
                            "written by the user: counterparty weights, topic "
                            "multipliers, and a floor for anything carrying a "
                            "deadline. Nothing is learned from the mailbox — by "
                            "message count the top of this mailbox is "
                            "newsletters. Use it for 'what matters', 'what "
                            "needs attention'; quote the stated reasons rather "
                            "than inventing your own, and say that the ordering "
                            "is a configured judgement, not a fact about the "
                            "messages."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
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
    "knowledge_search": knowledge_search,
    "list_mail": list_mail,
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
