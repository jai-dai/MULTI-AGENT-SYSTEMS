"""Аналитик: структура спецификации кодом, качество — судьёй.

Граница между `assert` и судьёй проведена там же, где в предыдущих работах.
«Есть ли хоть один критерий приёмки» — детерминировано, проверяется бесплатно и
однозначно. «Проверяемый ли критерий» — суждение, и никакой `assert` его не
выразит.

Разделение важно не из аккуратности: метрика, куда свалено и то и другое, при
падении не отвечает на вопрос «что сломалось» — то ли аналитик стал
расплывчатым, то ли вернул спецификацию вообще без критериев. Первое —
деградация, второе — баг.
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from tests import metrics
from tests.conftest import ids_of, run_or_skip, stories
from tests.scored import assert_scored

ALL = stories()
SOLID = stories("happy_path", "medium")


def _render(spec: dict) -> str:
    return (f"TITLE: {spec['title']}\n"
            f"WHAT I UNDERSTOOD: {spec['restated_story']}\n"
            f"WORK PLAN:\n" + "\n".join(f"  - {s}" for s in spec["work_plan"])
            + "\nREQUIREMENTS:\n" + "\n".join(f"  - {r}" for r in spec["requirements"])
            + "\nACCEPTANCE CRITERIA:\n"
            + "\n".join(f"  - {c}" for c in spec["acceptance_criteria"]))


@pytest.mark.parametrize("story", ALL, ids=ids_of(ALL))
def test_spec_has_shape(story):
    """Спецификация заполнена целиком. Ноль токенов, однозначный ответ."""
    spec = run_or_skip(story["id"])["spec"]
    assert spec is not None, "аналитик не вернул спецификацию"
    assert spec["title"].strip(), "пустой заголовок"
    assert spec["restated_story"].strip(), (
        "нет пересказа задачи — человеку у ворот нечего проверять")
    assert spec["work_plan"], "пустой план работ"
    assert spec["requirements"], "ни одного требования"
    assert spec["estimated_complexity"] in ("simple", "medium", "complex")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_spec_has_acceptance_criteria(story):
    """У выполнимой задачи критерии приёмки обязаны быть.

    Вынесено из проверки формы намеренно: на расплывчатом запросе
    (`vague-request`) их отсутствие — правильное поведение, а не дефект. Требуй
    критерии там — и тест заставлял бы аналитика выдумывать их для задачи,
    которой ещё нет.
    """
    spec = run_or_skip(story["id"])["spec"]
    assert spec["acceptance_criteria"], (
        "нет критериев приёмки — QA нечего проверять, а разработчику непонятно, "
        "когда работа закончена")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_spec_quality(story, judge):
    record = run_or_skip(story["id"])
    assert_scored(
        LLMTestCase(input=story["user_story"], actual_output=_render(record["spec"])),
        [metrics.spec_quality(judge)])
