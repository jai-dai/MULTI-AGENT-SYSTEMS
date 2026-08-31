"""DocsMCP — то, из чего BA берёт контекст: веб и внутренняя документация.

    .venv/bin/python mcp_servers/docs_mcp.py        # http://127.0.0.1:8931/mcp

Два инструмента и одна разница между ними, ради которой всё и затевалось.

`web_search` идёт в открытый интернет: там свежо, но там нет НАШИХ решений.
`knowledge_search` идёт по корпусу команды — стандарты кодирования, гайды,
документация фреймворков, которые мы действительно используем. Аналог `@docs` в
Cursor: агент ищет не «как вообще принято», а «как принято здесь».

Для бизнес-аналитика это разделение и есть работа. Спецификация, написанная по
общим знаниям модели, требует того, что команда не делает; спецификация,
свёренная с внутренним стандартом, требует того, что команда делает.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP                          # noqa: E402

import preflight                                     # noqa: E402
import retriever                                     # noqa: E402
import tools                                         # noqa: E402
import vectorstore                                   # noqa: E402
from config import settings                          # noqa: E402

server = FastMCP(
    name="DocsMCP",
    instructions=(
        "Context for writing a specification. knowledge_search goes over the "
        "team's own documentation and coding standards and returns passages "
        "with file and page; web_search goes to the open web."),
)


@server.tool
def knowledge_search(query: str, top_n: int = 5) -> str:
    """Search the team's internal documentation and coding standards.

    Args:
        query: what to look for, in the language of the documents
        top_n: how many passages to return
    """
    return tools.knowledge_search(query=query, top_n=top_n)


@server.tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the open web. Returns titles, URLs and snippets."""
    return tools.web_search(query=query, max_results=max_results)


@server.tool
def read_url(url: str) -> str:
    """Fetch one web page and return its readable text."""
    return tools.read_url(url=url)


@server.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """Что лежит в корпусе команды: сколько документов, какие, за какой период.

    Ресурс, а не инструмент: инструмент ДЕЛАЕТ, ресурс ОПИСЫВАЕТ. Аналитику
    незачем тратить шаг рассуждения на вызов «а что у вас есть» — это читают до
    начала работы, чтобы решить, идти ли в корпус вообще.
    """
    out: dict = {"indexes": [], "backend": settings.vector_backend}
    for name in retriever.index_names():
        directory = vectorstore.index_dir(name)
        chunks_file = directory / "chunks.json"
        if not chunks_file.exists():
            out["indexes"].append({"name": name, "error": "нет chunks.json"})
            continue
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        sources = sorted({c.get("source", "") for c in chunks})
        out["indexes"].append({"name": name, "chunks": len(chunks),
                               "documents": len(sources), "sources": sources[:30]})
    return json.dumps(out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    preflight.guard()
    print(f"DocsMCP → http://{settings.protocol_host}:{settings.docs_mcp_port}/mcp",
          flush=True)
    server.run(transport="http", host=settings.protocol_host,
               port=settings.docs_mcp_port)
