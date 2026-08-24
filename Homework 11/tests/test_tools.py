"""Tool correctness: правильные ли инструменты позвал агент.

# Что здесь на самом деле проверяется

Набор инструментов — это ГРАНИЦЫ РОЛИ, а не удобство. Планировщику намеренно не
дан `read_url`: с ним он начинает читать статьи целиком вместо того, чтобы
составить план (довод из hw8, повторён в шапке `agents/planner.py`). Тест на
tool correctness — единственное место, где это решение вообще проверяется:
границы живут в одной строке `TOOLS = [...]`, и стереть их можно случайно.

# Почему сверка по именам, а не по аргументам

`ToolCorrectnessMetric` умеет сверять `input_parameters` и требовать точной
последовательности. Здесь это выключено (см. `metrics.py`): формулировки
поисковых запросов агент придумывает сам, и требовать совпадения строки значило
бы тестировать угадывание формулировки. Порядок тоже свободен — план не
обязывает идти в корпус раньше веба.

Ожидаемый набор при этом НЕ равен полному списку доступных: ожидается минимум,
без которого роль не выполнена. `available_tools` передаётся отдельно и включает
вторую половину метрики — судья оценивает, оптимален ли был выбор, и итог
берётся как min(детерминированный, LLM).
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from agents import critic, planner, research
from tests import metrics
from tests.scored import assert_scored
from tests.conftest import (called_names, examples, ids_of, stage_or_skip,
                            tool_calls_as)

HAPPY = examples("happy_path")

# Инструменты SearchMCP целиком — для второй, LLM-половины метрики.
ALL_SEARCH_TOOLS = sorted(set(planner.TOOLS) | set(research.TOOLS)
                          | set(critic.TOOLS))


def _expected(names: list[str]) -> list[ToolCall]:
    return [ToolCall(name=n) for n in names]


# --------------------------------------------------------------------------- #
# 1. Planner: должен осмотреть область поисковыми инструментами
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_planner_uses_search_tools(example, judge):
    recorded = stage_or_skip(example["id"], "planner")
    assert_scored(
        LLMTestCase(
            input=example["input"],
            actual_output=str(recorded["output"]),
            tools_called=tool_calls_as(recorded),
            expected_tools=_expected(["web_search", "knowledge_search"]),
        ),
        [metrics.tool_correctness(judge, available_tools=ALL_SEARCH_TOOLS)],
    )


@pytest.mark.parametrize("example", examples(), ids=ids_of(examples()))
def test_planner_stays_inside_its_role(example):
    """Детерминированно: планировщик не может позвать то, чего у него нет.

    Проверка на первый взгляд лишняя — реестр и так собран из `planner.TOOLS`.
    Но именно поэтому она и стоит копейки, а ловит подмену набора инструментов
    при рефакторинге раньше, чем это заметит судья.
    """
    recorded = stage_or_skip(example["id"], "planner")
    outside = [n for n in called_names(recorded)
               if n not in planner.TOOLS and not n.startswith("submit_")]
    assert not outside, (
        f"планировщик позвал инструменты вне своей роли: {outside}. "
        f"Ему разрешены только {planner.TOOLS} — с `read_url` он начинает "
        f"читать статьи вместо того, чтобы планировать.")


# --------------------------------------------------------------------------- #
# 2. Researcher: должен использовать источники, названные в плане
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_researcher_follows_the_plan_sources(example, judge):
    """`sources_to_check` из плана -> ожидаемые инструменты.

    Ожидание берётся не из головы теста, а из ПЛАНА этого же прогона: план
    сказал, где искать, и проверяется, туда ли пошёл исследователь. Так тест
    остаётся верным, даже когда планировщик решит иначе, — он проверяет
    согласованность двух агентов, а не совпадение с константой в тесте.
    """
    plan_stage = stage_or_skip(example["id"], "planner")
    recorded = stage_or_skip(example["id"], "researcher")

    sources = plan_stage["output"].get("sources_to_check", [])
    expected = []
    if "knowledge_base" in sources:
        expected.append("knowledge_search")
    if "web" in sources:
        expected.append("web_search")
    if not expected:
        pytest.skip("план не назвал источников — нечего сверять")

    assert_scored(
        LLMTestCase(
            input=example["input"],
            actual_output=recorded["output"][:4000],
            tools_called=tool_calls_as(recorded),
            expected_tools=_expected(expected),
        ),
        [metrics.tool_correctness(judge, available_tools=ALL_SEARCH_TOOLS)],
    )


# --------------------------------------------------------------------------- #
# 3. Supervisor: дойдя до готового отчёта, обязан позвать save_report
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_supervisor_delegates_and_saves(example, judge):
    """Полный порядок координации: plan -> research -> critique -> save_report.

    Это единственный тест инструментов на уровне ACP: суб-агенты у супервизора
    подставлены в реестр наравне с `save_report`, и цикл ReAct не отличает
    «позвать агента по сети» от «позвать инструмент» (см. шапку
    `supervisor.py`). Значит и проверяется это одной метрикой — что и есть
    лучший аргумент в пользу того, что механизм выбран верно.
    """
    recorded = stage_or_skip(example["id"], "e2e")
    assert_scored(
        LLMTestCase(
            input=example["input"],
            actual_output=recorded["output"][:4000],
            tools_called=tool_calls_as(recorded),
            expected_tools=_expected(["plan", "research", "critique",
                                      "save_report"]),
        ),
        [metrics.tool_correctness(
            judge, available_tools=["plan", "research", "critique",
                                    "save_report"])],
    )


@pytest.mark.parametrize("example", HAPPY, ids=ids_of(HAPPY))
def test_supervisor_planned_before_researching(example):
    """Детерминированно: план идёт первым.

    Порядок здесь не вкусовщина. Исследователь получает план текстом, и вызов
    `research` раньше `plan` означает, что он пошёл искать по пересказу запроса
    супервизором — то есть стадия планирования оплачена и выброшена.
    """
    recorded = stage_or_skip(example["id"], "e2e")
    order = [n for n in called_names(recorded)
             if n in ("plan", "research", "critique", "save_report")]
    assert order, "супервизор не позвал ни одного суб-агента"
    assert order[0] == "plan", (
        f"первым вызван '{order[0]}', а не 'plan'. Порядок был: {order}")
    if "save_report" in order:
        assert order.index("save_report") > order.index("research"), (
            f"отчёт сохранён раньше, чем проведено исследование: {order}")
