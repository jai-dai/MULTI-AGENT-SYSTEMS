"""Critic: инварианты вердикта кодом, качество критики — судьёй.

Задание предлагает проверять GEval-метрикой в том числе такое: «если вердикт
APPROVE, список gaps должен быть пуст», «если REVISE, должен быть хотя бы один
revision_request». Это ошибка уровня, и здесь она исправлена сознательно: оба
утверждения ДЕТЕРМИНИРОВАНЫ. Их проверяет `assert` — бесплатно, однозначно и без
шанса, что судья сегодня решит иначе.

Судье достаётся то, что действительно требует суждения: конкретна ли претензия,
выполнима ли доработка, проверял ли критик факты или пересказал структуру.

Разделение важно не из аккуратности. Метрика, куда свалено и то и другое, при
падении не отвечает на вопрос «что сломалось»: то ли критик стал расплывчатым,
то ли вернул REVISE без единого требования. Первое — деградация, второе — баг.
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from schemas import CritiqueResult
from supervisor import render_critique
from tests import metrics
from tests.scored import assert_scored
from tests.conftest import examples, ids_of, stage_or_skip

HAPPY = examples("happy_path")


# --------------------------------------------------------------------------- #
# детерминированное: инварианты вердикта
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_verdict_is_actionable(example):
    """REVISE без требований — это тупик: супервизору нечего передать дальше."""
    recorded = stage_or_skip(example["id"], "critic")
    result = CritiqueResult.model_validate(recorded["output"])

    if result.verdict == "REVISE":
        assert result.revision_requests, (
            "вердикт REVISE без единого revision_request — исследователь "
            "получит «переделай» без указания что")
        assert result.gaps, "REVISE без gaps: не сказано, чего не хватает"
    else:
        assert not result.revision_requests, (
            "вердикт APPROVE, но выставлены требования к доработке — "
            "супервизор пойдёт дальше и требования потеряются")


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_verdict_agrees_with_its_own_dimensions(example):
    """Три булевых поля и вердикт не должны противоречить друг другу.

    Поля заведены как раз затем, чтобы критик проверил каждое измерение
    отдельно (см. `schemas.py`). Вердикт APPROVE при трёх `False` означает, что
    поля заполнены формально, а решение принято мимо них — то есть механизм не
    работает, хотя схема валидна.
    """
    recorded = stage_or_skip(example["id"], "critic")
    result = CritiqueResult.model_validate(recorded["output"])
    dimensions = [result.is_fresh, result.is_complete, result.is_well_structured]

    if result.verdict == "APPROVE":
        assert sum(dimensions) >= 2, (
            f"APPROVE при {sum(dimensions)} из 3 удовлетворённых измерений "
            f"(fresh={result.is_fresh} complete={result.is_complete} "
            f"structured={result.is_well_structured})")
    else:
        assert not all(dimensions), (
            "REVISE, хотя все три измерения признаны хорошими — "
            "вердикт не следует из собственной оценки")


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_critic_checked_the_sources(example):
    """Критик без собственного поиска оценивает связность, а не верность.

    Инструменты даны ему намеренно, те же, что исследователю (см. шапку
    `agents/critic.py`): «свежесть» без похода в источники непроверяема в
    принципе. Критик, не сделавший ни одного вызова, — это ревью текста.
    """
    recorded = stage_or_skip(example["id"], "critic")
    searched = [c for c in recorded["tool_calls"] if not c["failed"]]
    assert searched, (
        "критик не открыл ни одного источника — он оценил текст, а не факты")


# --------------------------------------------------------------------------- #
# семантическое: судья
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_critique_quality(example, judge):
    recorded = stage_or_skip(example["id"], "critic")
    result = CritiqueResult.model_validate(recorded["output"])
    assert_scored(
        LLMTestCase(
            # На вход судье идут находки, а не исходный запрос: критику
            # предъявляют именно их, и «конкретна ли претензия» имеет смысл
            # только относительно текста, к которому она предъявлена.
            input=recorded["input_findings"][:12000],
            actual_output=render_critique(result),
        ),
        [metrics.critique_quality(judge)],
    )
