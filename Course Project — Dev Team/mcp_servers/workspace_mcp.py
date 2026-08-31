"""WorkspaceMCP — руки команды: файлы на диске и запуск кода в песочнице.

    .venv/bin/python mcp_servers/workspace_mcp.py   # http://127.0.0.1:8932/mcp

# Почему запуск кода вынесен за границу процесса

Песочница (`sandbox.py`) и так исполняет код в отдельном процессе. Но сам
ЗАПУСК живёт здесь, в MCP-сервере, а не внутри графа — и это вторая граница,
поставленная намеренно.

Граф с агентами и сервер, исполняющий чужой код, — разные роли и разная цена
ошибки. Держи их в одном процессе, и любая беда исполнения оказывается на
расстоянии одного стека от оркестрации. Разные адреса делают эту разницу видимой
в архитектуре, а не только в намерении автора.

# Почему запись и чтение — разные инструменты

`write_file` есть только у разработчика, `read_file` — у обоих. Это не удобство,
а границы ролей: QA, который может переписать код, перестаёт быть проверяющим.
Набор инструментов — единственное место, где такое разделение вообще проводится,
и фильтр применяется при сборке агента.

# Чего этот сервер НЕ делает

Он не спрашивает человека и не проверяет, кто к нему обратился: выполнит запись
для любого, кто дозвонится. Для установки на 127.0.0.1 приемлемо, но это
допущение, а не защита, и записать его надо прямо.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP                          # noqa: E402

import sandbox                                       # noqa: E402
from config import settings                          # noqa: E402

server = FastMCP(
    name="WorkspaceMCP",
    instructions=(
        "The project workspace. write_file and read_file work on files under "
        "the workspace directory; run_python executes code in a sandboxed "
        "subprocess with a timeout and a memory limit."),
)

ROOT = Path(settings.workspace_dir).resolve()


def _resolve(relative: str) -> Path | str:
    """Путь внутри рабочего каталога, либо текст ошибки.

    Проверка не формальная: `../../.env` — это ровно то, что напишет модель,
    которой показалось, что нужный файл лежит выше. Ошибка возвращается ТЕКСТОМ,
    потому что агент должен прочитать её и попробовать иначе.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return (f"ERROR: путь '{relative}' ведёт за пределы рабочего каталога. "
                f"Файлы проекта живут внутри {ROOT.name}/.")
    return candidate


@server.tool
def write_file(path: str, content: str) -> str:
    """Write a project file. The path is relative to the workspace root.

    Args:
        path: e.g. src/main.py or tests/test_main.py
        content: full file contents
    """
    target = _resolve(path)
    if isinstance(target, str):
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"ERROR: не удалось записать '{path}' ({type(exc).__name__}: {exc})."
    return (f"written: {path} ({len(content.splitlines())} строк, "
            f"{len(content)} символов)")


@server.tool
def read_file(path: str) -> str:
    """Read a project file. The path is relative to the workspace root."""
    target = _resolve(path)
    if isinstance(target, str):
        return target
    if not target.exists():
        return (f"ERROR: файла '{path}' нет. Существующие: "
                f"{', '.join(list_files().splitlines()[:10]) or '(пусто)'}")
    try:
        return target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"ERROR: не удалось прочитать '{path}' ({type(exc).__name__}: {exc})."


@server.tool
def list_files() -> str:
    """List every file currently in the project workspace."""
    ROOT.mkdir(parents=True, exist_ok=True)
    found = sorted(p.relative_to(ROOT).as_posix()
                   for p in ROOT.rglob("*") if p.is_file())
    return "\n".join(found) if found else "(рабочий каталог пуст)"


@server.tool
def run_python(code: str) -> str:
    """Run Python code in a sandboxed subprocess and return what it printed.

    The code runs in the project workspace, so it can import the files written
    there. It has a timeout and a memory limit, and cannot import os,
    subprocess, shutil, socket or other system modules.

    Args:
        code: the program to execute
    """
    return sandbox.run(code, ROOT).as_text()


@server.resource("resource://workspace")
def workspace_state() -> str:
    """Что сейчас лежит в рабочем каталоге и каковы ограничения песочницы."""
    ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(p.relative_to(ROOT).as_posix()
                   for p in ROOT.rglob("*") if p.is_file())
    return json.dumps({
        "path": str(ROOT),
        "files": files,
        "sandbox": {
            "timeout_seconds": sandbox.TIMEOUT_SECONDS,
            "memory_limit_mb": sandbox.MEMORY_LIMIT_MB,
            "output_limit_chars": sandbox.OUTPUT_LIMIT_CHARS,
            "blocked_modules": list(sandbox.BLOCKED),
        },
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(f"WorkspaceMCP → http://{settings.protocol_host}:"
          f"{settings.workspace_mcp_port}/mcp   (workspace: {ROOT})", flush=True)
    server.run(transport="http", host=settings.protocol_host,
               port=settings.workspace_mcp_port)
