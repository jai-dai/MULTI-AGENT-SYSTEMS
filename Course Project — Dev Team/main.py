"""REPL команды разработки. Здесь же живёт человек в контуре.

    .venv/bin/python main.py        # сначала ./start_servers.sh

# Как устроена остановка на человеке

Узел `gate` вызывает `interrupt` — граф ОСТАНАВЛИВАЕТСЯ, `ainvoke` возвращается
не закончив, и в состоянии лежит описание того, что нужно решить. Дальше дело
вызывающего: показать спецификацию, спросить человека и запустить граф заново
через `Command(resume=…)`, который попадает ровно в ту точку.

Отсюда две вещи, которых не было бы в синхронном цикле: обязательный
checkpointer (продолжить остановленный граф можно, только если его состояние
где-то лежит) и цикл `while` здесь, снаружи, — потому что остановок за один
запрос может быть несколько, если человек возвращает спецификацию на доработку.
"""
from __future__ import annotations

import asyncio
import socket
from datetime import datetime

from langgraph.types import Command

import observability
from config import settings
from team import build

PREVIEW_LINES = 40

REQUIRED = [
    (settings.docs_mcp_port, "DocsMCP"),
    (settings.workspace_mcp_port, "WorkspaceMCP"),
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


# Слово, которым утверждается спецификация. Ровно одно, целиком и без
# сокращений — см. `ask_human`.
APPROVE_WORD = "APPROVE"
REVISE_WORD = "REVISE"


def ask_human(payload: dict) -> dict:
    """Показать спецификацию и вернуть решение человека.

    # Почему утверждение — это одно точное слово

    Здесь стоял список: `approve`, `a`, `y`, `yes`, `да`. То есть случайное «a»
    и Enter запускали разработку по неутверждённым требованиям — ровно та
    ошибка, от которой ворота и защищают. Короткое подтверждение экономит
    полсекунды и стоит всех итераций, которые пойдут по неверной спецификации.

    Теперь утверждает ТОЛЬКО слово `APPROVE`, набранное целиком. Ни `a`, ни `y`,
    ни пустая строка, ни `К`, случайно набранное в русской раскладке.

    # Почему непонятный ввод не считается отказом

    Соблазн есть: «всё, что не approve, — revise». Но тогда опечатка молча
    отправляет аналитика на второй круг, и человек не понимает, почему.
    Непонятный ввод — это НЕ РЕШЕНИЕ, и правильная реакция на него —
    переспросить, а не выбрать за человека.
    """
    print("\n" + "=" * 64)
    print("⏸️  СПЕЦИФИКАЦИЯ НА УТВЕРЖДЕНИЕ")
    print("=" * 64)
    lines = str(payload.get("spec", "")).splitlines()
    print("\n".join("  " + l for l in lines[:PREVIEW_LINES]))
    if len(lines) > PREVIEW_LINES:
        print(f"  … ещё {len(lines) - PREVIEW_LINES} строк")
    if payload.get("saved_to"):
        print(f"\n  спецификация сохранена: {payload['saved_to']}")
    print()
    print(f"  Наберите {APPROVE_WORD} — передать в разработку")
    print(f"          {REVISE_WORD}  — вернуть аналитику с замечаниями")

    while True:
        try:
            choice = input(f"  👉 {APPROVE_WORD} / {REVISE_WORD}: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Обрыв ввода — это не согласие. Молчание не утверждает.
            print()
            return {"decision": "revise",
                    "feedback": "Ввод прерван, спецификация не утверждена."}

        if choice.upper() == APPROVE_WORD:
            return {"decision": "approve"}
        if choice.upper() == REVISE_WORD:
            try:
                feedback = input("  ✏️  что исправить: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                feedback = ""
            return {"decision": "revise", "feedback": feedback}

        if not choice:
            print(f"  пустая строка ничего не утверждает — наберите "
                  f"{APPROVE_WORD} или {REVISE_WORD}")
        else:
            print(f"  '{choice}' — не команда. Нужно ровно {APPROVE_WORD} "
                  f"или {REVISE_WORD}, целиком.")


def _interrupt_of(state) -> dict | None:
    interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
    if not interrupts:
        return None
    value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
    return value if isinstance(value, dict) else {"spec": str(value)}


async def run_once(team, story: str, thread: str, session: str | None = None):
    """Одна user story до конца, сколько бы остановок она ни потребовала."""
    payload = {"user_story": story, "iteration": 0, "scores": []}
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 60}

    # Два вложенных контекста, и оба обязательны.
    #
    # `propagate_attributes` навешивает session/user/теги — но НЕ создаёт спана.
    # Без собственного корня каждый вызов агента заводит СВОЙ trace, и вместо
    # одного дерева на прогон в интерфейсе оказывается пять несвязанных: ba,
    # developer, qa по отдельности. Замерено: первый прогон с трассировкой дал
    # ровно это, с корнем `qa` вместо `dev-team`.
    #
    # `root_span` этим корнем и становится: всё, что построят агенты внутри,
    # встанет под него ветками.
    with _traced(story, session):
        with observability.root_span("dev-team", input=story) as root:
            state = await _loop(team, payload, config)
            if root is not None and hasattr(root, "update"):
                review = state.get("review")
                root.update(output={
                    "verdict": getattr(review, "verdict", None),
                    "score": getattr(review, "score", None),
                    "iterations": state.get("iteration", 0) + 1,
                    "files": getattr(state.get("code"), "files_created", []),
                })
            return state


async def _loop(team, payload, config):
    while True:
        state = await team.ainvoke(payload, config)
        pending = _interrupt_of(state)
        if pending is None:
            return state
        payload = Command(resume=ask_human(pending))


def _traced(story: str, session: str | None):
    import contextlib

    if not observability.enabled():
        return contextlib.nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(
        session_id=session, user_id=settings.langfuse_user_id,
        trace_name="dev-team", tags=["dev-team", "langgraph", "mcp"],
        metadata={"user_story": story[:200]})


def _report(state: dict) -> None:
    review = state.get("review")
    code = state.get("code")
    scores = state.get("scores", [])
    print("\n" + "=" * 64)
    if review is None:
        print("Работа не дошла до ревью.")
        return
    print(f"ИТОГ: {review.verdict}   score={review.score:.2f}   "
          f"итераций: {state.get('iteration', 0) + 1}")
    if len(scores) > 1:
        print("динамика score: " + " → ".join(f"{s:.2f}" for s in scores))
    if code and code.files_created:
        print("файлы: " + ", ".join(code.files_created))
    if review.verdict != "APPROVED" and review.issues:
        print("осталось нерешённым:")
        for i, issue in enumerate(review.issues, 1):
            print(f"  {i}. {issue}")


async def main() -> None:
    if not _servers_ready():
        return

    team = await build()
    session = f"team-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("Команда разработки: BA → человек → Developer ⇄ QA")
    print(f"  DocsMCP      {settings.docs_mcp_port}   (RAG + web для аналитика)")
    print(f"  WorkspaceMCP {settings.workspace_mcp_port}   (файлы + песочница)")
    print(f"model: {settings.model_name} | предел итераций ревью: "
          f"{settings.max_review_iterations}")
    if observability.enabled():
        print(f"трейсы: {settings.langfuse_base_url} | session={session}")
    print("exit — выйти")
    print("-" * 64)

    number = 0
    while True:
        try:
            story = input("\nUser story: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not story:
            continue
        if story.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        number += 1
        try:
            state = await run_once(team, story, f"story-{number}", session)
            _report(state)
        except KeyboardInterrupt:
            print("\n[прервано]")
        except Exception as exc:
            print(f"\n[ошибка] {type(exc).__name__}: {exc}")
        finally:
            observability.flush()


if __name__ == "__main__":
    asyncio.run(main())
