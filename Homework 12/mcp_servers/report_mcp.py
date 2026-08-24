"""ReportMCP — единственная операция записи в системе, отдельным сервером.

    .venv/bin/python mcp_servers/report_mcp.py        # http://127.0.0.1:8902/mcp

Отдельный сервер не ради симметрии с поиском. У чтения и записи разные права и
разная цена ошибки: неудачный поиск стоит токенов, неудачная запись — файла на
диске пользователя. Разные адреса делают эту разницу видимой в архитектуре, а
не только в голове у автора.

# Контракт отчёта держит сервер, а не вызывающий

`save_report` отказывает, если в тексте нет разделов Conclusions и Sources.
Проверка стоит ЗДЕСЬ, потому что это свойство отчёта, а не свойство того, кто
его пишет: завтра писать будет другой агент, а требование останется тем же.

# Чего этот сервер НЕ делает

Он не спрашивает человека. Подтверждение живёт у супервизора
(`HumanInTheLoopMiddleware`), и сервер о нём не знает — он выполнит запись для
любого, кто до него дозвонится. Для установки на 127.0.0.1 это приемлемо, но
это допущение, а не защита, и записать его надо прямо.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP                          # noqa: E402

import tools                                         # noqa: E402
from config import settings                          # noqa: E402

server = FastMCP(
    name="ReportMCP",
    instructions=(
        "Save a finished markdown report to disk. The report must contain a "
        "'Conclusions' section and a 'Sources' section, otherwise the call is "
        "rejected with an explanation."),
)


@server.tool
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
    server.run(transport="http", host=settings.protocol_host,
               port=settings.report_mcp_port)
