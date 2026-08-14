"""Planner — разбирает запрос на выполнимый план.

Инструменты у него есть намеренно: план, написанный без осмотра предметной
области, отправляет исследователя за словами, которых в источниках нет. Сначала
пара поисков, потом декомпозиция — и только потом структура.

Против hw8 изменилась одна строка: инструменты приходят не из `toolbox`, а из
`McpToolset` — то есть по сети. Сам агент разницы не замечает, и это главный
вывод урока: протокол меняет доставку инструмента, а не роль агента.
"""
from __future__ import annotations

import prompts
from agents.react import ReactAgent
from mcp_utils import McpToolset
from schemas import ResearchPlan

# Границы роли. Планировщику не дан `read_url`: с ним он начинает читать статьи
# целиком вместо того, чтобы составить план.
TOOLS = ["web_search", "knowledge_search"]
MAX_STEPS = 6


def build(toolset: McpToolset, depth: int = 1) -> ReactAgent:
    return ReactAgent(
        name="planner",
        system_prompt=prompts.PLANNER,
        registry=toolset.registry,
        schemas=toolset.schemas,
        max_steps=MAX_STEPS,
        output_model=ResearchPlan,
        depth=depth,
    )
