"""REPL мультиагентной системы. Здесь же живёт человек в контуре.

    .venv/bin/python main.py        # сначала поднять MCP и A2A серверы

# Как устроен HITL на LangGraph, и чем он отличается от своего цикла

`HumanInTheLoopMiddleware` не спрашивает человека сам. Он ОСТАНАВЛИВАЕТ ГРАФ:
`ainvoke` возвращается, не закончив работу, и в состоянии лежит `__interrupt__` с
описанием того, что агент собирался сделать. Дальше дело вызывающего — показать
это человеку и запустить граф заново командой `Command(resume=...)`, которая
попадает ровно в ту точку, где он встал.

Отсюда две вещи, которых не было в версии со своим синхронным циклом:

1. **Checkpointer обязателен.** Продолжить остановленный граф можно только если
   его состояние где-то лежит. В своём цикле «прерывание» было вызовом `input()`
   внутри функции — стек никуда не девался, и восстанавливать было нечего.
2. **Цикл здесь, а не внутри.** `ainvoke` вызывается в `while`, потому что
   подтверждений за один запрос может быть несколько: человек попросил правку,
   агент переписал отчёт и снова зовёт `save_report`.

Что это даёт взамен: остановленный граф можно продолжить в ДРУГОМ процессе — на
том и стоит вся конструкция LangGraph. Здесь эта возможность не нужна (REPL
живёт в одном процессе), и честно сказать это прямо: механизм мощнее задачи.

Три ответа человека выражаются одним каналом:

    approve  → инструмент исполняется, идёт вызов ReportMCP
    edit     → сервер НЕ зовём, модель получает замечания и правит
    reject   → сервер НЕ зовём, модель получает отказ и останавливается
"""
from __future__ import annotations

import asyncio
import json
import socket

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import observability
from config import settings
from supervisor import MAX_REVISIONS, Supervisor

PREVIEW_LINES = 15

REQUIRED = [
    (settings.search_mcp_port, "SearchMCP"),
    (settings.report_mcp_port, "ReportMCP"),
    (settings.planner_port, "A2A planner"),
    (settings.researcher_port, "A2A researcher"),
    (settings.critic_port, "A2A critic"),
]


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _servers_ready() -> bool:
    """Проверяем ДО старта: «система не работает» и «ты забыл поднять серверы» —
    разные новости, и путать их дорого."""
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


def _preview(arguments: dict) -> tuple[str, str]:
    """(имя файла, превью) из аргументов вызова."""
    content = str(arguments.get("content", ""))
    lines = content.splitlines()
    head = "\n".join(lines[:PREVIEW_LINES])
    if len(lines) > PREVIEW_LINES:
        head += f"\n… ещё {len(lines) - PREVIEW_LINES} строк"
    # Размер — от ПОЛНОГО отчёта: человек подтверждает запись целого файла.
    size = f"{len(lines)} строк, {len(content)} символов"
    return f"{arguments.get('filename', '(без имени)')}  ({size})", head


def _interrupt_payload(state) -> dict | None:
    """Достать `HITLRequest` из остановленного графа.

    Форма зафиксирована в `human_in_the_loop.py`:
    `{"action_requests": [{"name", "args", "description"}], "review_configs": [...]}`.
    Запросов может быть несколько — модель вправе позвать два инструмента за
    один ход, и человек тогда решает по каждому.
    """
    interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
    if not interrupts:
        return None
    value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
    return value if isinstance(value, dict) else {"action_requests": []}


def _ask_human(action: dict):
    """Показать ОДНО действие и вернуть решение в формате middleware.

    Возвращается `Decision`: `{"type": "approve"}` либо
    `{"type": "reject", "message": ...}`. Замечания человека едут именно
    отказом с текстом, а не отдельным типом: инструмент не исполняется, а
    модель получает объяснение и переписывает отчёт.
    """
    arguments = action.get("args") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"content": arguments}
    filename, head = _preview(arguments)

    print("\n" + "=" * 60)
    print("⏸️  ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ")
    print("=" * 60)
    print(f"  файл: {filename}")
    print()
    print("\n".join("  " + line for line in head.splitlines()))
    print()

    while True:
        try:
            choice = input("  👉 approve / edit / reject: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            choice = "reject"

        if choice in ("approve", "a", "y", "yes", "да"):
            return {"type": "accept"}
        if choice in ("edit", "e", "правка"):
            try:
                feedback = input("  ✏️  что изменить: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                feedback = ""
            if not feedback:
                print("  (пусто — считаю отказом)")
                return {"type": "reject", "message": (
                    "The user declined to save the report and gave no feedback. "
                    "Stop and report that nothing was saved.")}
            print("  ↩️  отправляю на доработку")
            # Замечания возвращаются модели как результат вызова — для неё это
            # неотличимо от «инструмент отработал и вот что он сказал».
            return {"type": "reject", "message": (
                f"The user did NOT approve the report and asks for changes: "
                f"{feedback}\nRevise the FULL report accordingly and call "
                "save_report again. The file has not been written.")}
        if choice in ("reject", "r", "n", "no", "нет"):
            print("  ❌ отменено")
            return {"type": "reject", "message": (
                "The user rejected saving the report. The file was NOT written. "
                "Do not try again under a different name — stop and tell the "
                "user the report was not saved.")}
        print("  не понял — approve, edit или reject")


async def run_once(supervisor: Supervisor, request: str, thread: str,
                   session_id: str | None = None) -> str:
    """Один запрос до конца, сколько бы подтверждений он ни потребовал.

    Здесь же рождается КОРЕНЬ трассы. `propagate_attributes` навешивает
    `session_id`, `user_id` и теги на всё дерево разом — включая то, что
    построится в других процессах: они подцепятся к этому же `trace_id`.

    Обрати внимание на область: контекст держится вокруг ВСЕГО цикла
    подтверждений, а не вокруг одного `ainvoke`. Иначе правка отчёта человеком
    порождала бы второй trace, и в интерфейсе один разговор выглядел бы как два
    несвязанных запуска.
    """
    supervisor.request = request
    supervisor.reset()
    payload: dict | Command = {"messages": [HumanMessage(content=request)]}
    config_ = {"configurable": {"thread_id": thread}, "recursion_limit": 60,
               "callbacks": observability.callbacks()}

    # Два вложенных контекста, и оба обязательны:
    #   propagate_attributes — вешает session/user/теги на всё дерево;
    #   root_span            — ВЛАДЕЕТ контекстом OTel, чтобы `carrier()` внутри
    #                          асинхронных инструментов не вернул пустоту.
    with _traced(request, session_id):
        with observability.root_span("multi-agent-research", input=request):
            # Снимаем контекст ЗДЕСЬ — это единственное место, где он гарантированно
            # есть. Дальше он едет параметром, а не через окружение.
            supervisor.trace_carrier = observability.carrier()
            return await _loop(supervisor, payload, config_)


def _traced(request: str, session_id: str | None):
    """Контекст трассировки на весь запрос, либо пустышка без ключей."""
    import contextlib

    if not observability.enabled():
        return contextlib.nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(
        session_id=session_id,
        user_id=settings.langfuse_user_id,
        trace_name="multi-agent-research",
        tags=["hw12", "a2a", "mcp"],
        metadata={"request": request[:200]},
        # Связь трассы с ВЕРСИЕЙ промпта супервизора. Без неё видно «ответ стал
        # хуже», но не видно, что накануне поменяли третью строку промпта.
        prompt=observability.prompt_object("supervisor"),
    )


async def _loop(supervisor: Supervisor, payload, config_) -> str:
    while True:
        state = await supervisor.agent.ainvoke(payload, config_)
        interrupt = _interrupt_payload(state)
        if interrupt is None:
            messages = state.get("messages", [])
            return messages[-1].content if messages else ""
        # Решений должно быть РОВНО столько, сколько запросов: middleware
        # сверяет длины и падает с понятной ошибкой, если не сошлось.
        decisions = [_ask_human(a) for a in interrupt.get("action_requests", [])]
        payload = Command(resume={"decisions": decisions})


async def main() -> None:
    if not _servers_ready():
        return

    supervisor = Supervisor()
    await supervisor.build()

    print("Мультиагентная система по протоколам: MCP (инструменты) + A2A (агенты)")
    for name, url in (("planner", settings.planner_port),
                      ("researcher", settings.researcher_port),
                      ("critic", settings.critic_port)):
        print(f"  A2A {name:11} → http://{settings.protocol_host}:{url}/")
    print(f"  MCP отчёты      → http://{settings.protocol_host}:"
          f"{settings.report_mcp_port}/mcp")
    print(f"model: {settings.model_name} | доработок максимум: {MAX_REVISIONS}")
    print("exit — выйти, reset — забыть диалог")
    print("-" * 60)

    # Одна сессия на запуск REPL: задание требует, чтобы трейсы группировались,
    # и группировать их по РАЗГОВОРУ — единственное осмысленное деление. Дата в
    # имени, чтобы вчерашняя сессия не смешалась с сегодняшней.
    session = f"{settings.langfuse_session_prefix}-{_stamp()}"
    if observability.enabled():
        print(f"трейсы: {settings.langfuse_base_url} | session={session} | "
              f"user={settings.langfuse_user_id}")
    else:
        print("Langfuse не настроен — работаем без трассировки "
              "(впиши ключи в .env)")

    round_no = 0
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
            round_no += 1
            supervisor.reset()
            print("Память очищена — новый thread для графа.")
            continue

        try:
            answer = await run_once(supervisor, request, f"repl-{round_no}",
                                    session_id=session)
        except KeyboardInterrupt:
            print("\n[прервано]")
            continue
        except Exception as exc:
            print(f"\n[ошибка] {type(exc).__name__}: {exc}")
            continue
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
