"""Сквозная проверка: то, что команда отдала, отвечает на исходную просьбу.

Отдельно от «код покрывает спецификацию» намеренно. Код может точно исполнить
спецификацию, а спецификация — разойтись с тем, что просил человек. Первая
метрика проверяет ИСПОЛНЕНИЕ, эта — что исполняли то самое.

Здесь же живут гарантии координации, которые держит код графа, а не промпты.
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from config import settings
from tests import metrics
from tests.conftest import ids_of, run_or_skip, stories
from tests.scored import assert_scored
from tests.context import render_files

ALL = stories()
SOLID = stories("happy_path", "medium")
VAGUE = stories("edge_case")


@pytest.mark.parametrize("story", ALL, ids=ids_of(ALL))
def test_iteration_limit_held(story):
    """Предел итераций живёт в КОДЕ, значит проверяется кодом.

    Промпт соблюдает «не больше пяти» обычно, но не всегда, а цена нарушения —
    бесконечный цикл на живых деньгах. Счётчик, ушедший выше предела, это не
    деградация качества, а прорванный предохранитель, и падать такой тест должен
    громко.
    """
    record = run_or_skip(story["id"])
    assert record["iterations"] <= settings.max_review_iterations, (
        f"итераций {record['iterations']} при пределе "
        f"{settings.max_review_iterations} — предохранитель не сработал")


@pytest.mark.parametrize("story", ALL, ids=ids_of(ALL))
def test_human_gate_was_passed(story):
    """До разработки спецификацию видел человек. Ровно один раз на версию.

    В записи это автоподтверждение, но проверяется сам факт: код не может
    появиться, минуя ворота.
    """
    record = run_or_skip(story["id"])
    assert record["approvals"], (
        "код написан, а спецификацию никто не утверждал — ворота обойдены")
    assert record["approvals"][0].get("saved_to"), (
        "спецификация не сохранена на диск")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_solves_user_story(story, judge):
    record = run_or_skip(story["id"])
    body = render_files(record["files"])
    assert_scored(
        LLMTestCase(input=story["user_story"], actual_output=body),
        [metrics.solves_user_story(judge)])


@pytest.mark.parametrize("story", VAGUE, ids=ids_of(VAGUE))
def test_vague_request_is_not_invented(story, judge):
    """На запрос без предмета аналитик не должен выдумывать систему.

    Самая дорогая ошибка всей цепочки: спецификация, придуманная над пустотой,
    запускает разработку и все её итерации. Правильное поведение — выписать
    открытые вопросы, а не угадать, чего хотел человек.
    """
    record = run_or_skip(story["id"])
    spec = record["spec"]
    rendered = (f"WHAT I UNDERSTOOD: {spec['restated_story']}\n"
                f"REQUIREMENTS:\n"
                + "\n".join(f"- {r}" for r in spec["requirements"]))
    assert_scored(
        LLMTestCase(
            input=(f"USER STORY: {story['user_story']}\n\n"
                   f"The correct behaviour for a request with no subject is to "
                   f"say plainly what is missing and list open questions, NOT to "
                   f"invent a system nobody asked for."),
            actual_output=rendered),
        [metrics.no_invention(judge)])
