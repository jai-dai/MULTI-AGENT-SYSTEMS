"""Researcher: обоснованность находок источниками, которые он реально открыл.

# Почему retrieval context здесь настоящий

Groundedness сравнивает утверждения с тем, что вернул поиск. Значит нужен не
корпус целиком и не «релевантные чанки», а РОВНО ТО, что агент видел: результаты
его собственных вызовов, в том виде, в каком они пришли. Всё остальное измеряет
что-то другое — например, хорош ли ретривер, — и называет это обоснованностью.

Такой контекст существует только потому, что стадия `researcher` записывается
в процессе тестов, а не через ACP: по протоколу результаты инструментов остаются
в чужом процессе и наружу не видны (см. `capture.py`).

# Зачем рядом встроенная faithfulness

Лекция 11 говорит про неё прямо: она ищет ПРОТИВОРЕЧИЯ контексту, а не
подтверждение. Утверждение, которого в источниках нет вовсе, противоречить им не
может — и проходит. Для RAG это ровно та галлюцинация, ради которой всё
затевалось.

Обе метрики гоняются на одном и том же случае намеренно. Faithfulness высокая
при низкой groundedness — это не шум, это измеренный объём «правды не из
источников»: модель отвечает из весов и не спорит с текстом, которого не читала.
"""
from __future__ import annotations

import json

import pytest
from deepeval.test_case import LLMTestCase

from tests import metrics
from tests.scored import assert_scored
from tests.conftest import examples, ids_of, stage_or_skip

HAPPY = examples("happy_path")


def _case(example, recorded) -> LLMTestCase:
    return LLMTestCase(
        input=example["input"],
        actual_output=recorded["output"],
        retrieval_context=recorded["retrieval_context"],
    )


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_research_is_grounded(example, judge):
    recorded = stage_or_skip(example["id"], "researcher")
    if not recorded["retrieval_context"]:
        pytest.fail("исследователь не сделал ни одного удачного поиска — "
                    "обосновывать находки нечем")
    assert_scored(_case(example, recorded), [metrics.groundedness(judge)])


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_research_does_not_contradict_sources(example, judge):
    """Встроенная faithfulness. Проходить должна ЛЕГЧЕ, чем groundedness выше."""
    recorded = stage_or_skip(example["id"], "researcher")
    if not recorded["retrieval_context"]:
        pytest.skip("нет контекста поиска")
    assert_scored(_case(example, recorded), [metrics.faithfulness(judge)])


# --------------------------------------------------------------------------- #
# детерминированное
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_research_actually_searched(example):
    """Находки без единого удачного поиска — это ответ из весов модели."""
    recorded = stage_or_skip(example["id"], "researcher")
    good = [c for c in recorded["tool_calls"] if not c["failed"]]
    assert good, "ни одного удачного вызова инструмента: искать агент не ходил"
    assert recorded["output"].strip(), "исследователь вернул пустоту"


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_research_did_not_go_in_circles(example):
    """Один и тот же запрос дважды — это оплаченный ноль.

    Первая версия этого теста сравнивала записанное число с `MAX_STEPS`, и это
    была ошибка: `agent.calls` считает ВЫЗОВЫ ИНСТРУМЕНТОВ, а не шаги цикла — за
    один шаг модель вправе позвать несколько. Восемнадцать вызовов при пределе в
    двенадцать шагов — совершенно здоровый прогон.

    Хождение по кругу видно иначе и надёжнее: буквально повторённый запрос к
    тому же инструменту. Он не приносит ни одного нового символа и стоит полной
    цены, потому что вся история едет в запрос на каждом шаге.
    """
    recorded = stage_or_skip(example["id"], "researcher")
    seen: dict[tuple, int] = {}
    for call in recorded["tool_calls"]:
        # Повтор ПОСЛЕ неудачи — это ретрай, и он правильный: `web_search`
        # периодически падает на ConnectError, и настаивать на том же запросе
        # разумно. Кругом считается только повтор УДАВШЕГОСЯ вызова.
        if call["failed"]:
            continue
        key = (call["name"], json.dumps(call["args"], sort_keys=True,
                                        ensure_ascii=False))
        seen[key] = seen.get(key, 0) + 1
    repeated = {name_args[0]: count for name_args, count in seen.items()
                if count > 1}
    assert not repeated, (
        f"исследователь повторил одни и те же вызовы: {repeated}. "
        f"Повтор не приносит нового, а история едет в запрос на каждом шаге — "
        f"это оплаченный ноль.")
