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

from pydantic import BaseModel, Field, model_validator


class ResearchPlan(BaseModel):
    """Разбор запроса на выполнимые шаги — до того, как начался поиск.

    # Почему у плана есть ветка «исследовать нечего»

    Её не было, и это стоило денег. Тесты hw10 замерили: на односложном «why», на
    бессмысленной строке и на попытке инъекции планировщик СУДИТ ВЕРНО — его
    `goal` говорит «уточнить, что имел в виду пользователь». Но сдать он обязан
    был `ResearchPlan`, а в `ResearchPlan` не было способа сказать «ничего».
    Поэтому верное суждение превращалось в поисковые запросы вроде
    `how to ask clarifying questions when user asks "why"`, и дальше по конвейеру
    шёл исследователь их выполнять: 477 170 токенов на бессмысленную строку,
    383 366 на рецепт борща.

    Виновата была не модель и не промпт, а ОТСУТСТВУЮЩАЯ ВЕТКА В МОДЕЛИ ДАННЫХ.
    Структурированный вывод с обязательной схемой забирает возможность
    отказаться: сказать можно только то, для чего есть поле.

    Инвариант держит валидатор, а не договорённость: либо `blocked_reason` и
    пустые запросы, либо запросы и никакого `blocked_reason`. Модель, нарушившая
    его, получит текст ошибки и переспросит (см. `structured.py`) — ровно так же,
    как при любом другом несоответствии схеме.
    """

    goal: str = Field(description="What we are trying to answer, in one sentence")
    blocked_reason: str | None = Field(
        default=None,
        description=(
            "Set this INSTEAD of a plan when no research should happen at all: "
            "the request has no answerable question, is outside this system's "
            "domain and tools, asks for something that must be declined, or "
            "asks for a capability this system does not have. Say plainly why, "
            "and what the user could ask instead. Leave search_queries empty. "
            "A blocked plan is a complete, correct answer — not a failure."))
    search_queries: list[str] = Field(
        default_factory=list,
        description="Specific queries to execute, in the language of the sources. "
                    "Empty only when blocked_reason is set.")
    sources_to_check: list[str] = Field(
        default_factory=list,
        description="Which sources to use: 'knowledge_base', 'web', or both")
    output_format: str = Field(
        default="",
        description="What the final report should look like")

    @model_validator(mode="after")
    def _blocked_or_planned(self) -> "ResearchPlan":
        """Ровно одно из двух. Половинчатого состояния не существует."""
        if self.blocked_reason:
            if self.search_queries:
                raise ValueError(
                    "blocked_reason is set, so there is nothing to search for — "
                    "leave search_queries empty. If research SHOULD happen, "
                    "clear blocked_reason instead.")
            return self
        missing = [name for name, empty in (
            ("search_queries", not self.search_queries),
            ("sources_to_check", not self.sources_to_check),
            ("output_format", not self.output_format.strip()),
        ) if empty]
        if missing:
            raise ValueError(
                f"a plan that is not blocked needs {', '.join(missing)}. "
                "If no research should happen at all, set blocked_reason and "
                "leave search_queries empty instead of filling this in.")
        return self


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
