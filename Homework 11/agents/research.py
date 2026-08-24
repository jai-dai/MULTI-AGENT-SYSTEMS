"""Researcher — тот самый агент из hw5, только промпт и роль другие.

Цикл, инструменты и весь RAG-конвейер переиспользуются как есть. Отличие от
hw5 одно: он больше не пишет отчёт. Отчёт собирает супервизор из находок, и
разделение здесь не косметическое — исследователь, который пишет итог, начинает
подгонять находки под уже написанный текст.

# Что изменилось с переездом на ACP

В hw8 экземпляр жил МЕЖДУ раундами сам собой: супервизор держал его в поле, и
во втором раунде исследователь помнил, что нашёл в первом. Здесь он живёт в
другом процессе, и «то же самое» пришлось сделать явно — по идентификатору
сессии ACP (см. `acp_server.py`). Ровно то, что раньше давалось бесплатно, стало
решением, которое надо принять и написать.
"""
from __future__ import annotations

import prompts
from agents.react import ReactAgent
from mcp_utils import McpToolset

TOOLS = ["web_search", "read_url", "knowledge_search"]
MAX_STEPS = 12


def build(toolset: McpToolset, depth: int = 1) -> ReactAgent:
    return ReactAgent(
        name="researcher",
        system_prompt=prompts.RESEARCHER,
        registry=toolset.registry,
        schemas=toolset.schemas,
        max_steps=MAX_STEPS,
        depth=depth,
    )
