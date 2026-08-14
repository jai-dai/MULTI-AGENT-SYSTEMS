"""Структурированный вывод там, где его нет в протоколе — через инструменты.

# Почему не `response_format`

У OpenAI есть `response_format={"type": "json_schema"}`, у Anthropic его нет:
там структурированный вывод делается ровно тем, что здесь и написано —
инструментом с нужной схемой и запретом отвечать иначе. Строить абстракцию на
возможности, которой у половины провайдеров нет, значит потерять переключение
`MODEL_NAME` одной строкой — то самое, ради чего `llm.py` и писался.

Инструменты же есть у обоих, и `llm.py` уже переводит их схемы в оба протокола.
Поэтому «структурированный ответ» здесь — это инструмент `submit_*`, чья схема
равна JSON-схеме Pydantic-модели.

# Почему валидация всё равно нужна

Модель вызывает инструмент с аргументами, которые она СОЧЛА подходящими схеме.
Заявленная схема не гарантия: приходят лишние поля, `verdict: "APPROVED"`
вместо `"APPROVE"`, список строкой. Pydantic превращает это в ошибку с текстом,
а текст возвращается модели как результат инструмента — она переспрашивает сама.
Молча принять кривой ответ хуже: вердикт, которого нет в `Literal`, увёл бы
супервизора в несуществующую ветку.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

# Префикс имени инструмента. Отдельный, чтобы цикл агента мог отличить
# «модель сдала результат» от «модель вызвала обычный инструмент».
SUBMIT_PREFIX = "submit_"


def tool_name(model: type[BaseModel]) -> str:
    """Имя инструмента для модели данных: ResearchPlan -> submit_research_plan."""
    name = model.__name__
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
    return SUBMIT_PREFIX + snake


def tool_schema(model: type[BaseModel]) -> dict:
    """Модель данных -> схема инструмента в формате OpenAI.

    `llm.py` переводит этот формат в оба протокола, поэтому здесь он и остаётся
    каноническим — как и для всех остальных инструментов проекта.
    """
    schema = model.model_json_schema()
    doc = (model.__doc__ or "").strip().split("\n\n")[0]
    return {
        "type": "function",
        "function": {
            "name": tool_name(model),
            "description": (
                f"Submit the final result as {model.__name__}. {doc} "
                "Call this exactly once, when the work is done."),
            "parameters": schema,
        },
    }


def parse(model: type[BaseModel], raw_arguments: str) -> tuple[BaseModel | None, str]:
    """(экземпляр, "") либо (None, текст ошибки для модели).

    Текст ошибки пишется ДЛЯ МОДЕЛИ, а не для лога: он должен объяснять, что
    именно исправить, потому что вернётся ей как результат инструмента.
    """
    try:
        data = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return None, f"ERROR: arguments are not valid JSON ({exc}). Call the tool again."
    if not isinstance(data, dict):
        return None, "ERROR: arguments must be a JSON object. Call the tool again."
    try:
        return model.model_validate(data), ""
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()[:6])
        return None, (f"ERROR: the result does not match {model.__name__} — "
                      f"{problems}. Fix these fields and call the tool again.")
