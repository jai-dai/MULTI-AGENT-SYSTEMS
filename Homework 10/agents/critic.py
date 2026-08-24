"""Critic — проверяет исследование ТЕМИ ЖЕ источниками, а не читает его.

У него тот же набор инструментов, что у исследователя, и это главное решение
роли: критик без доступа к источникам может оценить только связность текста, а
не его верность. «Свежесть» без собственного поиска непроверяема в принципе.

Дата подставляется в промпт при сборке: модель своей даты не знает и уверенно
считает свежим то, что было свежим на момент обучения.
"""
from __future__ import annotations

from datetime import date

from langchain.agents import create_agent

import config
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
        system_prompt=config.CRITIC + "\n" + config.DATE_NOTE.format(today=stamp),
        response_format=CritiqueResult,
    )
