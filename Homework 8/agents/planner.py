"""Planner — разбирает запрос на выполнимый план.

Инструменты у него есть намеренно: план, написанный без осмотра предметной
области, отправляет исследователя за словами, которых в источниках нет. Сначала
пара поисков, потом декомпозиция — и только потом структура.
"""
from __future__ import annotations

import prompts
import toolbox
from agents.react import ReactAgent
from schemas import ResearchPlan

MAX_STEPS = 6


def build(depth: int = 1) -> ReactAgent:
    registry, schemas = toolbox.for_role("planner")
    return ReactAgent(
        name="planner",
        system_prompt=prompts.PLANNER,
        registry=registry,
        schemas=schemas,
        max_steps=MAX_STEPS,
        output_model=ResearchPlan,
        depth=depth,
    )
