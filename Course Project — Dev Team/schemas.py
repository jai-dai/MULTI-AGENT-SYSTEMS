"""Контракты агентов. Здесь живёт то, на чём держится весь цикл.

# Почему это модели данных, а не свободный текст

Граф принимает РЕШЕНИЕ по ответу QA: вернуть работу разработчику или закончить.
Решение по свободному тексту означает, что его снова разбирает модель, и
«APPROVED» внутри фразы «я бы не сказал APPROVED» уводит граф не туда. Здесь
вердикт — это поле, а не догадка.

Валидация тоже не формальность: `verdict` ограничен двумя значениями, и модель,
придумавшая третье, получит ошибку и переспросит — вместо того чтобы отправить
условное ребро в ветку, которой нет.

# Почему у каждого поля есть description

Это не документация для человека. Описания уезжают в JSON-схему, которую видит
модель при `with_structured_output`, и они — единственное место, где сказано, что
`issues` это конкретные претензии, а не пересказ кода. Пустое описание означает,
что поле заполнят как придётся.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SpecOutput(BaseModel):
    """Спецификация: что именно надо построить, до того как написана строка кода.

    # Почему здесь есть поля сверх тех, что назвало задание

    Задание задаёт контракт: `title`, `requirements`, `acceptance_criteria`,
    `estimated_complexity`. Они все на месте — ниже добавлены ещё два, и вот
    зачем.

    На утверждение человеку уходит именно эта модель. Список требований — плохой
    материал для проверки: если аналитик понял задачу не так, это видно из
    ПЕРЕСКАЗА с первой строки, а из восьмого пункта требований — почти никогда.
    Человек утверждает не форму, а понимание.

    `restated_story` — задача своими словами. `work_plan` — как работа делится
    на шаги. Вместе они и есть то, что человек читает у ворот, и то, с чем
    разработчик сверяется потом.
    """

    title: str = Field(description="Short name of the feature, one line")
    restated_story: str = Field(
        description="What the user is asking for, in YOUR OWN words, one short "
                    "paragraph. If your reading differs from a literal one, say "
                    "so here — this is what the human approves or corrects.")
    work_plan: list[str] = Field(
        description="Ordered steps the developer will take. Each step is a unit "
                    "of work, not a requirement: 'parse the input file', "
                    "'validate the date range', 'write the tests'.")
    requirements: list[str] = Field(
        description="What the code must do. Each item is one testable statement, "
                    "not a paragraph. Include error handling and input validation "
                    "where the story implies them.")
    acceptance_criteria: list[str] = Field(
        description="How to tell the work is done. Each item must be CHECKABLE: "
                    "a concrete input and the expected outcome, not 'works well'.")
    estimated_complexity: Literal["simple", "medium", "complex"] = Field(
        description="simple: one function; medium: a few modules; "
                    "complex: needs design decisions")


class CodeOutput(BaseModel):
    """Результат разработчика: код, который уже лежит на диске."""

    source_code: str = Field(
        description="The main source file, complete and runnable as written")
    description: str = Field(
        description="What was built and which requirement each part covers")
    files_created: list[str] = Field(
        description="Paths of files actually written to disk, relative to the "
                    "workspace. Do not list files you did not create.")


class ReviewOutput(BaseModel):
    """Вердикт QA. Именно он управляет условным ребром графа.

    `score` не участвует в маршрутизации и стоит здесь намеренно: маршрут решает
    `verdict`, а число нужно, чтобы видеть ДИНАМИКУ по итерациям. Цикл, где
    оценка растёт 0.4 → 0.6 → 0.75, и цикл, где она стоит на 0.5, — разные
    болезни, и по одному вердикту их не различить.
    """

    verdict: Literal["APPROVED", "REVISION_NEEDED"] = Field(
        description="APPROVED only if the code satisfies every acceptance "
                    "criterion and runs. Otherwise REVISION_NEEDED.")
    issues: list[str] = Field(
        description="Concrete defects found. Each item must name WHAT is wrong "
                    "and WHERE. Empty only when the verdict is APPROVED.")
    suggestions: list[str] = Field(
        description="Actionable fixes. A developer must be able to act on each "
                    "one without asking what was meant.")
    score: float = Field(
        ge=0.0, le=1.0,
        description="0.0-1.0 quality estimate. Not used for routing; it makes "
                    "progress across iterations visible.")
