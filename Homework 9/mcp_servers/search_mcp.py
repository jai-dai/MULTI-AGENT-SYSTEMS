"""SearchMCP — поиск как протокол, а не как импорт.

    .venv/bin/python mcp_servers/search_mcp.py        # http://127.0.0.1:8901/mcp

Те же три инструмента, что были у агентов в hw8, но теперь их вызывают по сети.
Логика внутри не изменилась ни на строку: `knowledge_search` тянет за собой весь
гибридный конвейер (bge-m3 → RRF → кросс-энкодер), и переписывать его ради
протокола незачем.

# Зачем это, если раньше работало

Выигрыш не в модности протокола, а в том, что кросс-энкодер на 1.1 ГБ грузится
ТЕПЕРЬ ОДИН РАЗ — здесь. В hw8 три агента жили в одном процессе и делили его
случайно; вынеси их в разные процессы, как требует ACP, и каждый потянул бы
свою копию модели. На восьмигигабайтной машине это решает, поедет система или
нет.

Вторая половина ответа — Qdrant. До перехода на локальный сервер встроенный
режим держал блокировку каталога, и второй процесс к индексу просто не
подошёл бы. Четыре процесса hw9 существуют только потому, что хранилище стало
общим.

# Почему инструменты объявлены дважды

MCP описывает инструмент своей схемой, наш `tools.py` — своей (в формате
OpenAI, который `llm.py` переводит в оба протокола). Здесь они сходятся:
сигнатура функции ниже становится MCP-схемой автоматически, а тело просто
зовёт нашу реализацию. Дублируется описание, но не логика — и это осознанный
размен: MCP-клиент должен видеть инструмент, не зная про наш `TOOL_SCHEMAS`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer          # noqa: E402

import preflight                                    # noqa: E402
import retriever                                    # noqa: E402
import tools                                        # noqa: E402
import vectorstore                                  # noqa: E402
from config import settings                         # noqa: E402

server = MCPServer(
    name="SearchMCP",
    instructions=(
        "Search over a local knowledge base and the open web. "
        "knowledge_search returns passages with exact citations (file, page); "
        "web_search returns titles, URLs and snippets; read_url fetches one page."),
)


@server.tool()
def knowledge_search(query: str, top_n: int = 5,
                     source: str | None = None,
                     since: str | None = None,
                     until: str | None = None) -> str:
    """Search the local knowledge base. Returns passages with file and page.

    Args:
        query: what to look for, in the language of the documents
        top_n: how many passages to return
        source: restrict to files whose name contains this
        since: only documents dated on or after YYYY-MM-DD
        until: only documents dated on or before YYYY-MM-DD
    """
    return tools.knowledge_search(query=query, top_n=top_n, source=source,
                                  since=since, until=until)


@server.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the open web. Returns titles, URLs and snippets."""
    return tools.web_search(query=query, max_results=max_results)


@server.tool()
def read_url(url: str) -> str:
    """Fetch one web page and return its readable text."""
    return tools.read_url(url=url)


@server.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """Что вообще лежит в базе знаний — размер, источники, дата.

    Ресурс, а не инструмент, и разница здесь содержательная: инструмент
    ДЕЛАЕТ, ресурс ОПИСЫВАЕТ. Агенту незачем тратить шаг рассуждения на вызов
    «а что у тебя есть»; клиент читает это до начала работы и знает, стоит ли
    вообще идти в локальную базу или сразу в веб.
    """
    out: dict = {"indexes": [], "backend": settings.vector_backend,
                 "qdrant": settings.qdrant_url or "embedded"}
    for name in retriever.index_names():
        directory = vectorstore.index_dir(name)
        chunks_file = directory / "chunks.json"
        if not chunks_file.exists():
            out["indexes"].append({"name": name, "error": "нет chunks.json"})
            continue
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        sources = sorted({c.get("source", "") for c in chunks})
        dates = sorted(c["date"] for c in chunks if c.get("date"))
        out["indexes"].append({
            "name": name,
            "chunks": len(chunks),
            "documents": len(sources),
            "sources": sources[:20],
            "oldest": dates[0] if dates else None,
            "newest": dates[-1] if dates else None,
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Взвешивание машины переехало сюда из REPL, и переезд не косметический.
    # 1.1 ГБ кросс-энкодера, torch и индекс в памяти — всё это расходуется в
    # ЭТОМ процессе; в hw8 они жили вместе с диалогом, поэтому и проверка стояла
    # там. Оставь её на прежнем месте — и она уверенно называла бы 1.7 ГБ для
    # процесса, которому хватает мегабайтов, а настоящий едок стартовал бы
    # молча. Проверка обязана стоять там, где тратится память.
    preflight.guard()
    print(f"SearchMCP → http://{settings.protocol_host}:{settings.search_mcp_port}/mcp",
          flush=True)
    server.run(transport="streamable-http",
               host=settings.protocol_host, port=settings.search_mcp_port)
