"""Структурированные ответы Planner и Critic.

Зачем модели данных, а не свободный текст: супервизору нужно ПРИНЯТЬ РЕШЕНИЕ по
ответу суб-агента — идти дальше или вернуть на доработку. Решение по свободному
тексту означает, что его снова разбирает модель, и «APPROVE» внутри фразы «я бы
не сказал APPROVE» ломает цикл. Здесь же вердикт это поле, а не догадка.

Валидация тоже не формальность: `verdict` ограничен двумя значениями, и модель,
придумавшая третье, получит ошибку и переспросит — вместо того чтобы отправить
супервизора в ветку, которой нет.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Разбор запроса на выполнимые шаги — до того, как начался поиск."""

    goal: str = Field(description="What we are trying to answer, in one sentence")
    search_queries: list[str] = Field(
        description="Specific queries to execute, in the language of the sources")
    sources_to_check: list[str] = Field(
        description="Which sources to use: 'knowledge_base', 'web', or both")
    output_format: str = Field(
        description="What the final report should look like")


class CritiqueResult(BaseModel):
    """Оценка исследования по трём измерениям плюс вердикт.

    Три булевых поля не декоративны: они заставляют критика проверить каждое
    измерение отдельно, а не вынести общее впечатление. Вердикт при этом
    отдельным полем — критик вправе одобрить работу с оговорками.
    """

    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(
        description="Is the data up-to-date and based on recent sources?")
    is_complete: bool = Field(
        description="Does the research fully cover the user's original request?")
    is_well_structured: bool = Field(
        description="Are findings logically organized and ready for a report?")
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(
        description="What is missing, outdated, or poorly structured")
    revision_requests: list[str] = Field(
        description="Specific things to fix if verdict is REVISE")
