"""ReportMCP — единственная операция записи в системе.

    .venv/bin/python mcp_servers/report_mcp.py       # http://127.0.0.1:8902/mcp

# Где на самом деле стоит человек

Подтверждение живёт НЕ здесь, а у супервизора: он спрашивает человека перед
тем, как сделать этот вызов. Сервер о человеке не знает и знать не должен —
иначе подтверждение пришлось бы тащить через протокол, а «спросить» это
интерактивное действие, которого у HTTP-запроса нет.

Отсюда честное следствие, и его лучше назвать вслух: **сервер выполнит запись
для любого, кто до него дозвонится**. Для локальной однопользовательской
установки на `127.0.0.1` это приемлемо, но это именно допущение, а не защита.
Если серверы однажды выедут за пределы машины, ворота придётся ставить и здесь.

# Почему проверка разделов осталась в инструменте

`write_report` отказывается писать отчёт без «Conclusions» и «Sources» — и эта
проверка переехала сюда вместе с ним. Соблазн был вынести её к супервизору,
раз тот и так формирует отчёт, но тогда договор соблюдался бы только пока все
клиенты вежливы. Контракт сервера обязан держать сам сервер.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer          # noqa: E402

import tools                                        # noqa: E402
from config import settings                         # noqa: E402

server = MCPServer(
    name="ReportMCP",
    instructions=(
        "Save a finished markdown report to disk. The report must contain a "
        "'Conclusions' section and a 'Sources' section, otherwise the call is "
        "rejected with an explanation."),
)


@server.tool()
def save_report(filename: str, content: str) -> str:
    """Save a markdown report to the output directory.

    Args:
        filename: file name, e.g. rag_comparison.md
        content: full markdown text, with Conclusions and Sources sections
    """
    return tools.write_report(filename=filename, content=content)


@server.resource("resource://output-dir")
def output_dir() -> str:
    """Куда пишутся отчёты и что там уже лежит."""
    directory = Path(tools._output_dir())
    files = sorted(directory.glob("*.md")) if directory.exists() else []
    return json.dumps({
        "path": str(directory),
        "exists": directory.exists(),
        "reports": [{"name": f.name, "bytes": f.stat().st_size,
                     "modified": f.stat().st_mtime} for f in files],
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(f"ReportMCP → http://{settings.protocol_host}:{settings.report_mcp_port}/mcp",
          flush=True)
    server.run(transport="streamable-http",
               host=settings.protocol_host, port=settings.report_mcp_port)
