"""Сквозная оценка: то, что человек в итоге получает на руки.

# Три метрики, а не одна, и почему именно эти

`Correctness` (reference-based) сверяет с эталонным эскизом из golden dataset —
это pre-deployment проверка, ради которой датасет и заводят.
`AnswerRelevancy` (referenceless) спрашивает, отвечает ли текст на вопрос вообще;
она же единственная из трёх пригодна для production-мониторинга, где эталона нет.
`CitationPresence` — своя, под правило проекта: отчёт без опор равен одиночному
агенту, только в девятнадцать раз дороже.

Лекция 11 рекомендует 1–2 системные метрики плюс 1–2 кастомные. Здесь ровно так.

# Почему провальные случаи отдельным тестом

На половине датасета правильный ответ — это отказ, и обычные метрики на нём
не просто бесполезны, а ВРЕДНЫ: relevancy у уверенно выдуманного ответа выше,
чем у честного «в корпусе этого нет», потому что выдуманный ответ адресует
вопрос. Гонять их вместе значило бы штрафовать систему за правильное поведение.

# Чего эти тесты не проверяют, и это надо знать

Человека в контуре. `save_report` в записи подтверждён автоматически (см.
`capture.py`), то есть проверено только, что супервизор ДОШЁЛ до сохранения с
готовым отчётом. Ветки «человек сказал edit» и «человек сказал reject» —
главное, что вообще есть в hw8/hw9 — тестами не покрыты и покрыты быть не могут:
тест по определению идёт без человека.
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from supervisor import MAX_REVISIONS
from tests import metrics
from tests.scored import assert_scored
from tests.conftest import examples, ids_of, stage_or_skip

HAPPY = examples("happy_path")
HARD = examples("edge_cases", "failure_cases")


# --------------------------------------------------------------------------- #
# happy path: полный отчёт
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_report_is_correct_and_relevant(example, judge):
    recorded = stage_or_skip(example["id"], "e2e")
    # Оценивается ОТЧЁТ, а не реплика супервизора. Человек уносит файл, и
    # «сохранил отчёт, вот краткое содержание» прошло бы relevancy, ничего не
    # сказав о том, что в файле.
    answer = recorded["report"] or recorded["output"]
    assert_scored(
        LLMTestCase(input=example["input"], actual_output=answer,
                    expected_output=example["expected_output"]),
        [metrics.correctness(judge), metrics.answer_relevancy(judge)],
    )


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_report_cites_its_claims(example, judge):
    recorded = stage_or_skip(example["id"], "e2e")
    answer = recorded["report"] or recorded["output"]
    assert_scored(LLMTestCase(input=example["input"], actual_output=answer),
                [metrics.citation_presence(judge)])


# --------------------------------------------------------------------------- #
# edge + failure: правильный ответ это отказ
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HARD, ids=ids_of(HARD))
def test_system_is_honest_about_what_it_cannot_do(example, judge):
    recorded = stage_or_skip(example["id"], "e2e")
    answer = recorded["report"] or recorded["output"]
    assert_scored(
        LLMTestCase(input=example["input"], actual_output=answer,
                    expected_output=example["expected_output"]),
        [metrics.honest_refusal(judge)],
    )


# --------------------------------------------------------------------------- #
# детерминированное: гарантии координации
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", examples(), ids=ids_of(examples()))
def test_revision_limit_held(example):
    """Лимит доработок живёт в КОДЕ, а не в промпте — значит проверяем кодом.

    Промпт соблюдает «максимум две доработки» обычно, но не всегда, а цена
    нарушения — бесконечный цикл на живых деньгах (см. `supervisor.py`).
    Счётчик, ушедший выше предела, — это не деградация качества, а прорванный
    предохранитель, и падать такой тест должен громко.
    """
    recorded = stage_or_skip(example["id"], "e2e")
    assert recorded["revisions"] <= MAX_REVISIONS, (
        f"доработок {recorded['revisions']} при пределе {MAX_REVISIONS} — "
        f"предохранитель не сработал")


@pytest.mark.parametrize("example", examples(), ids=ids_of(examples()))
def test_blocked_plan_stops_delegation(example):
    """Заблокированный план обязан остановить дорогое — и остановить в КОДЕ.

    Тот же довод, по которому лимит доработок живёт в счётчике: модель решает
    ЧТО делать, код гарантирует СКОЛЬКО. Промпт «не зови исследователя»
    соблюдается обычно, но не всегда, а цена нарушения тут — сотни тысяч
    токенов, ровно те, ради которых ветка отказа и заводилась.
    """
    recorded = stage_or_skip(example["id"], "e2e")
    plan = recorded.get("plan")
    if not plan or not plan.get("blocked_reason"):
        pytest.skip("план не заблокирован — проверять нечего")

    delegated = [c["name"] for c in recorded["tool_calls"]
                 if c["name"] in ("research", "critique") and not c["failed"]
                 and not str(c["result"]).startswith("The plan is blocked")]
    assert not delegated, (
        f"план заблокирован ({plan['blocked_reason'][:80]}), но делегирование "
        f"всё равно состоялось: {delegated}")
    assert recorded["tokens"]["total"] < 100_000, (
        f"заблокированный запрос стоил {recorded['tokens']['total']:,} токенов — "
        f"смысл блокировки в том, чтобы он стоил копейки")


@pytest.mark.parametrize("example", examples(), ids=ids_of(examples()))
def test_run_produced_something(example):
    recorded = stage_or_skip(example["id"], "e2e")
    answer = recorded["report"] or recorded["output"]
    assert answer and answer.strip(), "прогон закончился пустым ответом"


@pytest.mark.parametrize("example", examples(), ids=ids_of(examples()))
def test_cost_is_recorded(example):
    """Цена прогона должна быть в записи — иначе разговор о ней невозможен.

    Не проверка качества, а защита сбора данных. Токены суб-агентов приезжают
    частью `stats` в ответе ACP и складываются супервизором вручную; молча
    потерянный `stats` выглядит как «мультиагент подешевел».
    """
    recorded = stage_or_skip(example["id"], "e2e")
    tokens = recorded["tokens"]
    assert tokens["total"] > 0, "суммарные токены нулевые"

    # Суб-агентов могли и не звать: на «why» правильный ответ — переспросить, не
    # потратив ни одного делегирования. Поэтому сборка `stats` проверяется
    # только там, где делегирование ФАКТИЧЕСКИ было. Иначе тест требовал бы
    # тратить деньги ради того, чтобы было что посчитать.
    delegated = [c["name"] for c in recorded["tool_calls"]
                 if c["name"] in ("plan", "research", "critique")]
    if not delegated:
        return
    subagents = {k: v for k, v in tokens.items()
                 if k not in ("supervisor", "total")}
    assert subagents, (
        f"супервизор делегировал ({delegated}), но токены суб-агентов не "
        f"собрались — потерян `stats` из ответа ACP")
