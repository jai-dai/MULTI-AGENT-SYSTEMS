"""Проверка ответов человека против НАСТОЯЩЕГО middleware.

    .venv/bin/python check_hitl.py

# Зачем отдельная проверка на тридцать строк

Здесь была ошибка, которую сквозной прогон не поймал и поймать не мог:
`_ask_human` возвращал `{"type": "accept"}`, а middleware сверяет литерал строкой
и знает только `approve` / `edit` / `reject` / `respond`. Живой человек, нажавший
«да», получил бы `Unexpected human decision`.

Почему это пережило тестирование: прогон подменял `_ask_human` автоподтверждением,
которое возвращало ПРАВИЛЬНЫЙ литерал. То есть тест заменял ровно ту функцию, в
которой сидела ошибка, и проверял всё, кроме неё. Классическая слепая зона моков:
подменяя интерактивную часть, подменяешь и её контракт.

Отсюда правило, ради которого файл и существует: **если что-то заменяется моком
ради тестируемости, эта самая часть обязана проверяться отдельно.**

Проверка не требует ни агентов, ни серверов, ни единого токена: три строки ввода
прогоняются через настоящий `_ask_human`, а результат — через настоящий
`_process_decision`.
"""
from __future__ import annotations

import contextlib
import io
import sys

import main
from langchain.agents.middleware import HumanInTheLoopMiddleware

ACTION = {"name": "save_report",
          "args": {"filename": "report.md", "content": "# Report\nbody\n"}}

CASES = [
    ("approve\n", "approve", "инструмент исполняется"),
    ("edit\nдобавь источники\n", "reject", "не исполняется, модель правит"),
    ("reject\n", "reject", "не исполняется, модель останавливается"),
]


def run() -> int:
    middleware = HumanInTheLoopMiddleware(interrupt_on={"save_report": True})
    tool_call = {"name": "save_report", "id": "call_1", "args": ACTION["args"]}
    config = middleware.interrupt_on["save_report"]

    failures = 0
    for keys, expected, note in CASES:
        stdin, sys.stdin = sys.stdin, io.StringIO(keys)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                decision = main._ask_human(ACTION)
        finally:
            sys.stdin = stdin

        if decision.get("type") != expected:
            print(f"  ✗ {keys.split(chr(10))[0]:8} -> {decision.get('type')!r}, "
                  f"ожидался {expected!r}")
            failures += 1
            continue
        try:
            middleware._process_decision(decision, tool_call, config)
        except Exception as exc:
            print(f"  ✗ {keys.split(chr(10))[0]:8} -> middleware отверг: "
                  f"{type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"  ✓ {keys.split(chr(10))[0]:8} -> {decision['type']:8} {note}")

    print("\nHITL: все ответы человека приняты middleware" if not failures
          else f"\nHITL: провалов {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
