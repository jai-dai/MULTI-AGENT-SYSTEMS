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
import toolbox
from agents.react import ReactAgent
from schemas import CritiqueResult

MAX_STEPS = 10


def build(depth: int = 1, today: str | None = None) -> ReactAgent:
    registry, schemas = toolbox.for_role("critic")
    stamp = today or date.today().isoformat()
    return ReactAgent(
        name="critic",
        system_prompt=(prompts.CRITIC + "\n"
                       + prompts.DATE_NOTE.format(today=stamp)),
        registry=registry,
        schemas=schemas,
        max_steps=MAX_STEPS,
        output_model=CritiqueResult,
        depth=depth,
    )
