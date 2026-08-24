"""Planner: детерминированные инварианты плюс качество плана судьёй.

Лекция 11 проводит границу так: детерминированное проверяем обычным pytest,
семантическое — LLM-as-a-Judge. Планировщик — лучшее место эту границу увидеть.
«План возвращает валидный ResearchPlan, и в нём непустые запросы» — это `assert`,
он стоит ноль и падает однозначно. «Запросы конкретные, а не расплывчатые» —
это уже суждение, и никакой `assert` его не выразит.

Поэтому здесь оба вида, и порядок не случаен: сначала дешёвые проверки формы.
Если план не разбирается, судить его содержание бессмысленно и незачем платить.
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from schemas import ResearchPlan
from supervisor import render_plan
from tests import metrics
from tests.scored import assert_scored
from tests.conftest import examples, ids_of, stage_or_skip

HAPPY = examples("happy_path")
UNANSWERABLE = examples("failure_cases")


# --------------------------------------------------------------------------- #
# детерминированное: обычный pytest, ноль токенов
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", examples(), ids=ids_of(examples()))
def test_plan_is_valid_structure(example):
    """План разбирается в модель данных, и поля в нём не декоративные.

    Это тот самый unit-тест, который в LLM-системах обычно забывают написать,
    решив, что «тут всё недетерминированное». Форма ответа как раз детерминирована
    — её гарантирует `structured.py`, — и проверять её судьёй было бы и дорого,
    и глупо.
    """
    recorded = stage_or_skip(example["id"], "planner")
    plan = ResearchPlan.model_validate(recorded["output"])

    assert plan.goal.strip(), "goal пустой"
    assert plan.output_format.strip(), "output_format пустой"
    assert all(q.strip() for q in plan.search_queries), "есть пустой запрос"
    # Непустой список запросов требуется НЕ везде. На «why» или на бессмыслицу
    # план без единого поиска — это правильный план: искать нечего. Требовать
    # запросов здесь значило бы тестом ЗАСТАВЛЯТЬ систему делать лишнюю работу
    # на 950 тысяч токенов.
    if example["category"] != "failure_cases":
        assert plan.search_queries, (
            "search_queries пуст — исследователю не с чем идти")

    known = {"knowledge_base", "web"}
    unknown = [s for s in plan.sources_to_check if s not in known]
    assert not unknown, (
        f"названы источники, которых у системы нет: {unknown}. "
        f"Исследователь пойдёт искать несуществующий инструмент.")


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_plan_looked_at_the_domain_first(example):
    """Планировщик обязан ОСМОТРЕТЬ область, а не сочинить план из головы.

    Инструменты даны ему намеренно (см. шапку `agents/planner.py`): план,
    написанный без осмотра, отправляет исследователя за словами, которых в
    источниках нет. Проверка дешёвая и ловит регресс промпта мгновенно.
    """
    recorded = stage_or_skip(example["id"], "planner")
    searched = [c for c in recorded["tool_calls"]
                if c["name"] in ("web_search", "knowledge_search")]
    assert searched, (
        "план составлен без единого поиска — планировщик сочинил область, "
        "вместо того чтобы её посмотреть")


# --------------------------------------------------------------------------- #
# семантическое: судья
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_plan_quality(example, judge):
    recorded = stage_or_skip(example["id"], "planner")
    plan = ResearchPlan.model_validate(recorded["output"])
    assert_scored(
        LLMTestCase(input=example["input"], actual_output=render_plan(plan)),
        [metrics.plan_quality(judge)],
    )


@pytest.mark.parametrize("example", UNANSWERABLE, ids=ids_of(UNANSWERABLE))
def test_plan_does_not_invent_work_for_unanswerable_requests(example, judge):
    """Самый дорогой отказ планировщика — бодрый план по бессмысленному запросу.

    На «why» или «wgqx flrb ... retrieval???» правильный план говорит, что
    запроса нет. План с десятью поисковыми запросами по слову «retrieval»
    запускает всю машину на 950 тысяч токенов ради ответа на вопрос, которого
    никто не задавал. Отдельный тест, потому что метрика тут та же, а цена
    ошибки — другая.
    """
    recorded = stage_or_skip(example["id"], "planner")
    plan = ResearchPlan.model_validate(recorded["output"])
    assert_scored(
        LLMTestCase(input=example["input"],
                    actual_output=render_plan(plan),
                    expected_output=example["expected_output"]),
        [metrics.honest_refusal(judge)],
    )
