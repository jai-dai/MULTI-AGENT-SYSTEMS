"""Кто чем вооружён. Наборы инструментов по агентам — в одном месте.

Инструменты переиспользуются из hw5 без изменений: `knowledge_search` тянет за
собой весь гибридный конвейер (bge-m3 + BM25 → RRF → кросс-энкодер), и трогать
его ради мультиагентности незачем. Здесь только сборка подмножеств.

# Почему у агентов РАЗНЫЕ наборы, а не один общий

Набор инструментов — это границы роли. Planner с `read_url` начал бы читать
статьи целиком вместо того, чтобы составить план: у него есть поиск, чтобы
понять предметную область, и этого достаточно. Общий набор на всех сделал бы
трёх агентов одним агентом, вызванным трижды.

# Почему здесь нет почтовых инструментов

`list_mail` из hw5 работает и остался бы полезен, но демонстрация идёт по
ПУБЛИЧНОЙ теме, а её результат попадёт в README. Инструмент, способный
принести в отчёт настоящую переписку, в такой конфигурации выключен намеренно —
не забыт. Чтобы вернуть: добавить "list_mail" в наборы ниже и включить
index_mail в SEARCH_INDEX_DIRS.
"""
from __future__ import annotations

from typing import Callable

import tools

# `save_report` из задания и `write_report` из hw5 — одно и то же действие.
# Второй реализации не нужно: там уже есть и безопасная сборка пути, и проверка
# обязательных разделов, без которой модель охотно сдаёт отчёт без выводов.
_SAVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_report",
        "description": (
            "Save the final markdown report to ./output/. Requires a "
            "'Conclusions' and a 'Sources' section. This action is shown to the "
            "user for approval before it happens."),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string",
                             "description": "File name, e.g. rag_comparison.md"},
                "content": {"type": "string",
                            "description": "Full markdown text of the report"},
            },
            "required": ["filename", "content"],
        },
    },
}

_REGISTRY: dict[str, Callable] = {
    "knowledge_search": tools.knowledge_search,
    "web_search": tools.web_search,
    "read_url": tools.read_url,
    "save_report": tools.write_report,
}

_SCHEMAS: dict[str, dict] = {
    schema["function"]["name"]: schema
    for schema in tools.TOOL_SCHEMAS
    if schema["function"]["name"] in _REGISTRY
}
_SCHEMAS["save_report"] = _SAVE_SCHEMA

# Роль -> инструменты. Planner осматривает домен, Researcher копает,
# Critic проверяет теми же источниками (иначе это рецензия текста, а не
# верификация), Supervisor не ищет вообще — он координирует.
SETS: dict[str, list[str]] = {
    "planner": ["web_search", "knowledge_search"],
    "researcher": ["web_search", "read_url", "knowledge_search"],
    "critic": ["web_search", "read_url", "knowledge_search"],
    "supervisor": ["plan", "research", "critique", "save_report"],
}


def for_role(role: str, extra: dict[str, Callable] | None = None
             ) -> tuple[dict[str, Callable], list[dict]]:
    """(реестр, схемы) для роли. `extra` — инструменты-суб-агенты супервизора.

    Неизвестное имя падает здесь, а не превращается в «модель вызвала
    инструмент, которого нет»: опечатка в наборе должна стоить исключения при
    старте, а не странного поведения на третьем шаге диалога.
    """
    extra = extra or {}
    registry: dict[str, Callable] = {}
    schemas: list[dict] = []
    for name in SETS[role]:
        if name in extra:
            registry[name] = extra[name]["func"]
            schemas.append(extra[name]["schema"])
        elif name in _REGISTRY:
            registry[name] = _REGISTRY[name]
            schemas.append(_SCHEMAS[name])
        else:
            raise KeyError(f"инструмент {name!r} для роли {role!r} не определён")
    return registry, schemas
