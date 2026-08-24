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
    assert all(q.strip() for q in plan.search_queries), "есть пустой запрос"

    known = {"knowledge_base", "web"}
    unknown = [s for s in plan.sources_to_check if s not in known]
    assert not unknown, (
        f"названы источники, которых у системы нет: {unknown}. "
        f"Исследователь пойдёт искать несуществующий инструмент.")

    # Ровно одно из двух, и половинчатого состояния быть не должно. Инвариант
    # держит валидатор в `schemas.py`, здесь он проверяется на ЗАПИСИ: схема
    # гарантирует его в момент создания, а тест — что на диске лежит то же
    # самое и что валидатор не обошли стороной.
    if plan.blocked_reason:
        assert not plan.search_queries, (
            "план заблокирован, но запросы всё равно составлены — "
            "исследователь пойдёт их выполнять")
    else:
        assert plan.search_queries, "search_queries пуст, а план не заблокирован"
        assert plan.output_format.strip(), "output_format пустой"


@pytest.mark.parametrize("example", UNANSWERABLE, ids=ids_of(UNANSWERABLE))
def test_plan_blocks_unanswerable_requests(example):
    """На запрос, который нельзя исследовать, план обязан быть ЗАБЛОКИРОВАН.

    Раньше это проверял только судья метрикой `Honest Refusal`, и проверял
    плохо: балл 0.0 говорил «плохо себя вёл», но не говорил ЧТО именно сломано.
    Теперь у плана есть ветка отказа, и для очевидных случаев вопрос стал
    двоичным — значит и проверка двоичная, кодом и бесплатно.

    Метрика при этом остаётся: она судит КАЧЕСТВО отказа — сказано ли внятно,
    что не так, и предложено ли осмысленное «а вот это могу». Здесь же
    проверяется сам факт, и это разные вопросы.
    """
    recorded = stage_or_skip(example["id"], "planner")
    plan = ResearchPlan.model_validate(recorded["output"])
    assert plan.blocked_reason, (
        f"план не заблокирован на запросе, который нельзя исследовать. "
        f"goal={plan.goal!r}, запросов: {len(plan.search_queries)}. "
        f"Дальше по конвейеру пойдёт исследователь их выполнять.")


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
