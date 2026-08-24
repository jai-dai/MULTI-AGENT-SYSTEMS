"""Critic — проверяет исследование ТЕМИ ЖЕ источниками, а не читает его.

У него тот же набор инструментов, что у исследователя, и это главное решение
роли: критик без доступа к источникам может оценить только связность текста, а
не его верность. «Свежесть» без собственного поиска непроверяема в принципе.

Дата подставляется в промпт при сборке: модель своей даты не знает и уверенно
считает свежим то, что было свежим на момент обучения.

Промпт больше не лежит рядом с кодом: он приезжает из Langfuse по имени
`NAME` и лейблу из настроек. Роль агента теперь описана в ДВУХ местах —
набор инструментов здесь, текст задачи там, — и это осознанный размен:
промпт стало можно править и откатывать, не трогая деплой.
"""
from __future__ import annotations

from datetime import date

from langchain.agents import create_agent

import observability
from schemas import CritiqueResult

NAME = "critic"
TOOLS = ["web_search", "read_url", "knowledge_search"]
DESCRIPTION = ("Verifies research findings against the sources and returns a "
               "verdict of APPROVE or REVISE with specific revision requests.")
SKILL_EXAMPLES = [
    "FINDINGS: ## 1) Naive RAG ... — verify freshness and completeness",
]


def build(tools, model: str, today: str | None = None):
    stamp = today or date.today().isoformat()
    return create_agent(
        model=model,
        tools=[t for t in tools if t.name in TOOLS],
        # `{{today}}` — template-переменная промпта в Langfuse: модель своей
        # даты не знает и уверенно считает свежим то, что было свежим на
        # момент обучения. Подстановка при compile, а не склейка строк.
        system_prompt=observability.prompt(NAME, today=stamp),
        response_format=CritiqueResult,
    )
