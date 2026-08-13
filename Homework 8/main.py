"""REPL мультиагентной системы. Здесь же живёт человек в контуре.

# Как устроен HITL без приостановки графа

`save_report` — единственная операция записи, и она не исполняется, пока её не
увидел человек. Механизм простой: цикл ReAct спрашивает `interceptor` ПЕРЕД
исполнением инструмента, и если тот вернул строку, она уходит модели как
результат вызова — сам инструмент не запускался.

Поэтому три действия человека выражаются одним и тем же каналом:

    approve  → инструмент исполняется, модель получает «сохранено»
    edit     → инструмент НЕ исполняется, модель получает замечания и правит
    reject   → инструмент НЕ исполняется, модель получает отказ и молчит

Ни checkpointer, ни возобновление графа не нужны: цикл синхронный, состояние —
это список сообщений в живом процессе, а «приостановка» это обычный `input()`.
"""
from __future__ import annotations

import json

import preflight
import tools
from supervisor import Supervisor

PREVIEW_LINES = 15


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


def approval_gate(name: str, arguments: str) -> str | None:
    """Крючок перед исполнением инструмента. None — исполнять как обычно."""
    if name != "save_report":
        return None

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
            result = tools.write_report(**json.loads(arguments))
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
            # Замечания возвращаются как результат инструмента — для модели это
            # неотличимо от «инструмент отработал и вот что он сказал», и она
            # переделывает отчёт, не зная, что между ними был человек.
            return ("The user did NOT approve the report and asks for changes: "
                    f"{feedback}\n"
                    "Revise the FULL report accordingly and call save_report "
                    "again. The file has not been written.")
        if choice in ("reject", "r", "n", "no", "нет"):
            print("  ❌ отменено")
            return ("The user rejected saving the report. The file was NOT "
                    "written. Do not try again under a different name — stop "
                    "and tell the user the report was not saved.")
        print("  не понял — approve, edit или reject")


def main() -> None:
    preflight.guard()

    supervisor = Supervisor(interceptor=approval_gate)
    print("Мультиагентная система: Supervisor + Planner / Researcher / Critic")
    import llm
    print(f"model: {llm.describe()} | доработок максимум: "
          f"{__import__('supervisor').MAX_REVISIONS}")
    print("exit — выйти, reset — забыть диалог")
    print("-" * 60)

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
            print("Память очищена.")
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


if __name__ == "__main__":
    main()
