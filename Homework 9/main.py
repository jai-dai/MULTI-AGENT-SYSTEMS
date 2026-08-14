"""REPL мультиагентной системы. Здесь же живёт человек в контуре.

    .venv/bin/python main.py        # сначала ./start_servers.sh

# Как устроен HITL, когда инструмент уехал на другой сервер

`save_report` — единственная операция записи, и она не исполняется, пока её не
увидел человек. Механизм тот же, что в hw8: цикл ReAct спрашивает `interceptor`
ПЕРЕД исполнением инструмента, и если тот вернул строку, она уходит модели как
результат вызова.

Переезд на MCP ничего здесь не сломал, и это стоит назвать вслух. Крючок стоит
между МОДЕЛЬЮ и вызовом, а не между вызовом и файлом. Поэтому неважно, что за
именем `save_report` теперь HTTP-запрос к ReportMCP: до сервера дело просто не
доходит, пока человек не сказал «да». Три действия человека выражаются одним и
тем же каналом:

    approve  → зовём ReportMCP, модель получает «сохранено»
    edit     → сервер НЕ зовём, модель получает замечания и правит
    reject   → сервер НЕ зовём, модель получает отказ и молчит

Оборотная сторона честно записана в шапке `report_mcp.py`: сервер выполнит
запись для любого, кто до него дозвонится. Ворота стоят у супервизора, а не у
двери. Для локальной установки на 127.0.0.1 это допущение приемлемо, но это
допущение, а не защита.

Ни checkpointer, ни возобновление графа по-прежнему не нужны: цикл синхронный,
состояние — список сообщений в живом процессе, а «приостановка» это `input()`.
"""
from __future__ import annotations

import json
import socket

from config import settings
from supervisor import Supervisor

PREVIEW_LINES = 15

# # Куда делся preflight
#
# В hw8 REPL первым делом взвешивал машину: кросс-энкодер на 1.1 ГБ грузился
# здесь же, и запуск на восьми гигабайтах стоило проверить заранее.
#
# Теперь этот процесс не грузит НИЧЕГО — ни реранкер, ни torch, ни индекс. Он
# умеет только разговаривать по сети, а модели живут в SearchMCP. Оставленная
# на прежнем месте проверка не просто стала лишней: она уверенно называла
# 1.7 ГБ там, где расходуются мегабайты, — то есть врала. Поэтому она уехала
# в `mcp_servers/search_mcp.py`, к моделям, а здесь проверяется единственное,
# от чего REPL действительно зависит, — что серверы подняты.

# Порт -> что на нём и чем поднимается. Проверяется до старта, потому что
# «система не работает» и «ты забыл поднять серверы» — разные новости.
REQUIRED = [
    (settings.search_mcp_port, "SearchMCP"),
    (settings.report_mcp_port, "ReportMCP"),
    (settings.acp_port, "ACP-сервер"),
]


def _servers_ready() -> bool:
    missing = []
    for port, name in REQUIRED:
        with socket.socket() as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((settings.protocol_host, port)) != 0:
                missing.append(f"{name} (порт {port})")
    if missing:
        print("Не подняты: " + ", ".join(missing))
        print("Запусти:  ./start_servers.sh")
        return False
    return True


def _preview_report(arguments: str) -> tuple[str, str]:
    """(имя файла, превью) из аргументов вызова. Кривые аргументы не роняют REPL."""
    try:
        data = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "(не разобрано)", arguments[:400]
    content = str(data.get("content", ""))
    lines = content.splitlines()
    head = "\n".join(lines[:PREVIEW_LINES])
    if len(lines) > PREVIEW_LINES:
        head += f"\n… ещё {len(lines) - PREVIEW_LINES} строк"
    # Размер — от ПОЛНОГО отчёта, а не от превью: человек подтверждает запись
    # целого файла, и знать он должен про файл.
    size = f"{len(lines)} строк, {len(content)} символов"
    return f"{data.get('filename', '(без имени)')}  ({size})", head


def make_approval_gate(save):
    """Крючок перед исполнением инструмента. `save` — вызов ReportMCP.

    Фабрика, а не функция модуля, потому что записывать теперь умеет не
    `tools`, а конкретное соединение с конкретным сервером. Крючок получает его
    снаружи и остаётся тем же, чем был: местом, где спрашивают человека.
    """
    def approval_gate(name: str, arguments: str) -> str | None:
        if name != "save_report":
            return None                        # None — исполнять как обычно

        filename, preview = _preview_report(arguments)
        print("\n" + "=" * 60)
        print("⏸️  ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ")
        print("=" * 60)
        print(f"  файл: {filename}")
        print()
        print("\n".join("  " + line for line in preview.splitlines()))
        print()

        while True:
            try:
                choice = input("  👉 approve / edit / reject: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                choice = "reject"

            if choice in ("approve", "a", "y", "yes", "да"):
                try:
                    data = json.loads(arguments or "{}")
                except json.JSONDecodeError as exc:
                    return f"ERROR: arguments are not valid JSON ({exc})."
                result = save(**data)          # только здесь дело доходит до MCP
                print(f"  ✅ {result}")
                return result
            if choice in ("edit", "e", "правка"):
                try:
                    feedback = input("  ✏️  что изменить: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    feedback = ""
                if not feedback:
                    print("  (пусто — считаю отказом)")
                    return ("The user declined to save the report and gave no "
                            "feedback. Stop and report that nothing was saved.")
                print("  ↩️  отправляю на доработку")
                # Замечания возвращаются как результат инструмента — для модели
                # это неотличимо от «инструмент отработал и вот что он сказал»,
                # и она переделывает отчёт, не зная, что между ними был человек.
                return ("The user did NOT approve the report and asks for "
                        f"changes: {feedback}\n"
                        "Revise the FULL report accordingly and call save_report "
                        "again. The file has not been written.")
            if choice in ("reject", "r", "n", "no", "нет"):
                print("  ❌ отменено")
                return ("The user rejected saving the report. The file was NOT "
                        "written. Do not try again under a different name — stop "
                        "and tell the user the report was not saved.")
            print("  не понял — approve, edit или reject")

    return approval_gate


def main() -> None:
    if not _servers_ready():
        return

    supervisor = Supervisor()
    # Крючок ставится после сборки: он зовёт ReportMCP, а соединение с ним
    # заводит сам супервизор. Отдать соединение до того, как оно открыто,
    # нельзя — вот и весь порядок.
    supervisor.agent.interceptor = make_approval_gate(
        supervisor.reports.registry["save_report"])

    import llm
    import supervisor as supervisor_module
    print("Мультиагентная система по протоколам: MCP (инструменты) + ACP (агенты)")
    print(f"  ACP {supervisor_module.ACP_URL} → {', '.join(supervisor.acp.names())}")
    print(f"  MCP {supervisor_module.REPORT_MCP_URL} → save_report")
    print(f"model: {llm.describe()} | доработок максимум: "
          f"{supervisor_module.MAX_REVISIONS}")
    print("exit — выйти, reset — забыть диалог")
    print("-" * 60)

    try:
        while True:
            try:
                request = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not request:
                continue
            if request.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if request.lower() == "reset":
                supervisor.reset()
                print("Память очищена — и здесь, и на ACP-сервере.")
                continue

            try:
                answer = supervisor.run(request)
            except KeyboardInterrupt:
                print("\n[прервано]")
                continue
            except RuntimeError as exc:
                print(f"\n[ошибка] {exc}")
                continue
            print(f"\nAgent: {answer}")
    finally:
        # Соединения держат фоновые потоки с циклами событий. Без закрытия
        # процесс не завершится сам.
        supervisor.close()


if __name__ == "__main__":
    main()
