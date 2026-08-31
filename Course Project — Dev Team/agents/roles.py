"""Три роли: набор инструментов, промпт и форма ответа.

# Набор инструментов — это границы роли

Не удобство и не оптимизация. Аналитику не дан `write_file`: аналитик, который
пишет код, перестаёт быть аналитиком, и спецификация начинает подгоняться под уже
написанное. QA не дан `write_file` по той же причине в другую сторону —
проверяющий, который может починить, чинит вместо того, чтобы находить.

Оба MCP-сервера отдают всем одно и то же, поэтому фильтр применяется ЗДЕСЬ, при
сборке агента. Это единственное место, где границы вообще проводятся.

# Почему у Developer и QA один и тот же `run_python`

Разработчик запускает код, чтобы он заработал. QA запускает его, чтобы он
сломался. Инструмент один, намерение разное, и это правильно: дай проверяющему
другой способ запуска — и он будет проверять не то, что поедет в работу.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import prompts
from schemas import CodeOutput, ReviewOutput, SpecOutput


@dataclass(frozen=True)
class Role:
    name: str
    prompt: str
    tools: tuple[str, ...]
    output: type
    description: str


BA = Role(
    name="ba",
    prompt=prompts.BA,
    # Только чтение и поиск. Никакой записи: спецификация пишется ДО кода.
    tools=("knowledge_search", "web_search", "read_url"),
    output=SpecOutput,
    description="Turns a user story into a testable specification",
)

DEVELOPER = Role(
    name="developer",
    prompt=prompts.DEVELOPER,
    tools=("write_file", "read_file", "list_files", "run_python",
           "web_search", "read_url"),
    output=CodeOutput,
    description="Implements the specification and leaves working files on disk",
)

QA = Role(
    name="qa",
    prompt=prompts.QA,
    # Читает и запускает, но НЕ пишет. См. шапку.
    tools=("read_file", "list_files", "run_python"),
    output=ReviewOutput,
    description="Reviews the code against the spec and returns a verdict",
)

ALL = (BA, DEVELOPER, QA)


def tools_for(role: Role, available: list) -> list:
    """Отфильтровать инструменты MCP под роль.

    Молча отдать меньше, чем просили, — плохой договор: агент, у которого нет
    `run_python`, будет вести себя странно, а причина не проявится нигде. Поэтому
    недостача — это исключение при сборке, а не сюрприз во время работы.
    """
    by_name = {t.name: t for t in available}
    missing = [n for n in role.tools if n not in by_name]
    if missing:
        raise KeyError(
            f"роли '{role.name}' нужны инструменты {missing}, а MCP-серверы "
            f"отдают: {sorted(by_name)}. Проверь, что оба сервера подняты.")
    return [by_name[n] for n in role.tools]
