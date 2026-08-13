"""Researcher — тот самый агент из hw5, только промпт и роль другие.

Цикл, инструменты и весь RAG-конвейер переиспользуются как есть. Отличие от
hw5 одно: он больше не пишет отчёт. Отчёт собирает супервизор из находок, и
разделение здесь не косметическое — исследователь, который пишет итог, начинает
подгонять находки под уже написанный текст.

Экземпляр живёт МЕЖДУ раундами: во втором раунде он помнит, что нашёл в первом,
и правка по замечаниям критика становится правкой, а не повторным поиском с
нуля.
"""
from __future__ import annotations

import prompts
import toolbox
from agents.react import ReactAgent

MAX_STEPS = 12


def build(depth: int = 1) -> ReactAgent:
    registry, schemas = toolbox.for_role("researcher")
    return ReactAgent(
        name="researcher",
        system_prompt=prompts.RESEARCHER,
        registry=registry,
        schemas=schemas,
        max_steps=MAX_STEPS,
        depth=depth,
    )
