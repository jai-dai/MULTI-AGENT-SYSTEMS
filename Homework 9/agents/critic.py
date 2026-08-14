"""Critic — проверяет исследование теми же источниками, а не читает его.

У него тот же набор инструментов, что у исследователя, и это главное решение:
критик без доступа к источникам может оценить только связность текста, а не его
верность. «Свежесть» без собственного поиска вообще непроверяема.

Дата подставляется в промпт при сборке: модель своей даты не знает и уверенно
считает свежим то, что было свежим на момент обучения.
"""
from __future__ import annotations

from datetime import date

import prompts
from agents.react import ReactAgent
from mcp_utils import McpToolset
from schemas import CritiqueResult

TOOLS = ["web_search", "read_url", "knowledge_search"]
MAX_STEPS = 10


def build(toolset: McpToolset, depth: int = 1,
          today: str | None = None) -> ReactAgent:
    stamp = today or date.today().isoformat()
    return ReactAgent(
        name="critic",
        system_prompt=(prompts.CRITIC + "\n"
                       + prompts.DATE_NOTE.format(today=stamp)),
        registry=toolset.registry,
        schemas=toolset.schemas,
        max_steps=MAX_STEPS,
        output_model=CritiqueResult,
        depth=depth,
    )
