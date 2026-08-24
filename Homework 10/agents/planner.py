"""Planner — разбирает запрос на выполнимый план.

Инструменты у него есть намеренно: план, написанный без осмотра предметной
области, отправляет исследователя за словами, которых в источниках нет. Сначала
пара поисков, потом декомпозиция.

Границы роли — это НАБОР ИНСТРУМЕНТОВ, а не просьба в промпте. Планировщику не
дан `read_url`: с ним он начинает читать статьи целиком вместо того, чтобы
составить план. Фильтр применяется здесь, при сборке агента, потому что
SearchMCP один на всех и отдаёт всем одно и то же.

`response_format=ResearchPlan` — это и есть структурированный вывод в LangChain:
модель обязана вернуть ровно эту модель данных, а не текст, который потом кто-то
разбирает регулярками.
"""
from __future__ import annotations

from langchain.agents import create_agent

import config
from schemas import ResearchPlan

NAME = "planner"
TOOLS = ["web_search", "knowledge_search"]
DESCRIPTION = ("Decomposes a research request into an executable plan: goal, "
               "specific search queries, which sources to use, and the shape of "
               "the expected answer.")
SKILL_EXAMPLES = [
    "Compare RAG approaches: naive, sentence-window, and parent-child",
    "What retrieval strategies exist beyond fixed-size chunking?",
]


def build(tools, model: str):
    return create_agent(
        model=model,
        tools=[t for t in tools if t.name in TOOLS],
        system_prompt=config.PLANNER,
        response_format=ResearchPlan,
    )
