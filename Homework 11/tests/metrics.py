"""Метрики и пороги — в одном месте, чтобы порог был решением, а не опечаткой.

# Почему пороги здесь, а не рядом с тестами

Порог — это утверждение о том, какое качество считается достаточным. Разбросанный
по пяти файлам, он перестаёт быть утверждением и становится числом, которое
правят, когда тест мешает. Собранный в одну таблицу, он читается как список
обещаний системы, и поднимать его приходится осознанно.

# Откуда взяты значения

Из задания там, где оно их называет (0.7 для GEval-метрик компонентов, 0.5 для
tool correctness, 0.6 для correctness), и низкие — там, где метрика своя и
базовой линии ещё нет. Лекция 11 называет завышенный порог антипаттерном
номер два: «поріг 0.95 → тести завжди червоні → команда їх відключає». Порядок
такой: сначала baseline, потом повышение. Пороги ниже — это НЕ «система хорошая»,
это «мы ещё не знаем, какая она».

# Про судью

Все метрики получают одного судью — экземпляр `ProjectJudge`, который ходит
моделью ДРУГОГО вендора, чем система (см. `judge.py`). Судья один на прогон,
потому что он же считает потраченные на оценку токены: разные судьи давали бы
разрозненные счётчики и никакого ответа на вопрос «сколько стоит тестировать».
"""
from __future__ import annotations

from deepeval.metrics import (AnswerRelevancyMetric, FaithfulnessMetric, GEval,
                              ToolCorrectnessMetric)
from deepeval.test_case import SingleTurnParams

# Порог -> обоснование. Второе не менее важно первого: число без причины
# нечем защищать, когда тест краснеет в неудобный момент.
# Значения ПОДНЯТЫ 2026-08-24 по базовой линии на всём датасете (18 примеров,
# 109 вызовов судьи). Это второй шаг цикла, который лекция и предписывает:
# сначала baseline, потом повышение. Стартовые значения из задания оказались
# ниже реального качества на 0.2-0.4, то есть не проверяли ничего.
#
# Правило подъёма: наблюдённый минимум минус 0.1. Запас именно такой, потому что
# судья недетерминирован: тот же кейс при повторе даёт разброс порядка десятой.
# Ставить порог вплотную к минимуму значит завести тест, который краснеет от
# настроения модели, — а это ровно тот путь, которым тесты выключают.
THRESHOLDS = {
    # baseline: сред 0.90, мин 0.90, макс 0.90 (n=6). Было 0.7.
    "plan_quality": 0.8,
    # baseline: сред 0.88, мин 0.80 (n=6). Оставлено: минимум уже близко.
    "groundedness": 0.7,
    # baseline: сред 0.87, мин 0.80 (n=6). Оставлено по той же причине.
    "critique_quality": 0.7,
    # baseline: сред 0.88, мин 0.50, макс 1.00 (n=18). Минимум упирается в пол
    # метрики: она штрафует за КАЖДЫЙ вызов сверх ожидаемого, а агент вправе
    # искать по-разному. Поднимать нельзя — покраснеет здоровое поведение.
    "tool_correctness": 0.5,
    # baseline: сред 0.91, мин 0.85 (n=6). Было 0.7.
    "answer_relevancy": 0.8,
    # baseline: сред 0.85, мин 0.70 (n=6). Оставлено: expected_output — эскиз,
    # написанный человеком, и требовать от него точности эталона нечестно.
    "correctness": 0.6,
    # baseline: сред 0.90, мин 0.90 (n=6). Было 0.5 — то есть порог пропускал
    # отчёт, где опору имеет половина утверждений. Поднято сильнее всех.
    "citation_presence": 0.8,
    # baseline: сред 0.39, мин 0.00, макс 1.00 (n=18), прошло 5 из 18. ЕДИНСТВЕННАЯ
    # метрика, которая не проходит, и поднимать её нельзя — её надо ЧИНИТЬ.
    # Порог оставлен там, где стоял: он и есть цель, а не описание факта.
    "honest_refusal": 0.6,
    # baseline: сред 1.00, мин 1.00 (n=6). Метрика ищет противоречия, и их нет
    # ни в одном прогоне. 0.9 — не строгость, а детектор поломки самой метрики.
    "faithfulness": 0.9,
}


# --------------------------------------------------------------------------- #
# компоненты
# --------------------------------------------------------------------------- #

def plan_quality(judge) -> GEval:
    return GEval(
        name="Plan Quality",
        evaluation_steps=[
            "Check that the plan contains specific search queries, not vague topics",
            "Check that sources_to_check includes sources relevant to the topic",
            "Check that the output_format matches what the user asked for",
            "If the request is unanswerable, nonsensical or out of scope, a good "
            "plan says so instead of inventing a research programme for it",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=THRESHOLDS["plan_quality"],
    )


def groundedness(judge) -> GEval:
    """Каждое утверждение подтверждено контекстом — а не «не противоречит» ему.

    Это не то же самое, что встроенная `FaithfulnessMetric`, и лекция 11
    подчёркивает разницу прямо: faithfulness ищет ПРОТИВОРЕЧИЯ контексту, то есть
    утверждение, которого в источниках просто нет, она пропускает — оно ведь
    ничему не противоречит. Для RAG важно ровно обратное: неподтверждённое
    утверждение и есть галлюцинация, даже если оно верно.

    Поэтому метрика своя, а встроенная гоняется рядом (см. `test_researcher.py`)
    — разрыв между ними и есть объём «правды не из источников».
    """
    return GEval(
        name="Groundedness",
        evaluation_steps=[
            "Extract every factual claim from 'actual output'",
            "For each claim, check if it can be directly supported by 'retrieval context'",
            "Claims not present in retrieval context count as ungrounded, even if true",
            "Statements the researcher makes about its own uncertainty, or about "
            "what it could not find, are not claims and are not counted",
            "Score = number of grounded claims / total claims",
        ],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.RETRIEVAL_CONTEXT],
        model=judge,
        threshold=THRESHOLDS["groundedness"],
    )


def faithfulness(judge) -> FaithfulnessMetric:
    """Встроенная метрика — только ради контраста с groundedness выше."""
    return FaithfulnessMetric(threshold=THRESHOLDS["faithfulness"], model=judge)


def critique_quality(judge) -> GEval:
    return GEval(
        name="Critique Quality",
        evaluation_steps=[
            "Check that the critique identifies specific issues, not vague complaints",
            "Check that revision_requests are actionable: a researcher could act "
            "on each one without asking what was meant",
            "Check that the strengths are specific to this research, and not "
            "generic praise that would fit any report",
            "A critique that verifies a claim against sources is worth more than "
            "one that only comments on structure",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=THRESHOLDS["critique_quality"],
    )


# --------------------------------------------------------------------------- #
# инструменты
# --------------------------------------------------------------------------- #

def tool_correctness(judge, available_tools: list[str] | None = None
                     ) -> ToolCorrectnessMetric:
    """Имена инструментов, без сверки аргументов.

    `should_exact_match` и сверка `input_parameters` намеренно ВЫКЛЮЧЕНЫ. Наши
    агенты формулируют поисковые запросы сами, и требовать совпадения строки
    запроса значило бы тестировать не выбор инструмента, а угадывание
    формулировки. Порядок вызовов тоже свободен: план не обязывает искать в
    корпусе раньше, чем в вебе.

    `available_tools` включает вторую половину метрики — LLM-судья оценивает,
    был ли выбор оптимальным, и итог берётся как min(детерминированный, LLM).
    Принимается список ИМЁН, хотя DeepEval ждёт `ToolCall`: имена — это то, чем
    роль описана в коде (`planner.TOOLS`), и превращать их в объекты на каждом
    вызове значило бы повторять одну строчку в пяти местах.
    """
    from deepeval.test_case import ToolCall
    return ToolCorrectnessMetric(
        threshold=THRESHOLDS["tool_correctness"],
        model=judge,
        available_tools=([ToolCall(name=n) for n in available_tools]
                         if available_tools else None),
    )


# --------------------------------------------------------------------------- #
# сквозные
# --------------------------------------------------------------------------- #

def correctness(judge) -> GEval:
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict 'expected output'",
            "Penalize omission of critical details named in 'expected output'",
            "Different wording of the same concept is acceptable",
            "'expected output' is a sketch of a complete answer, not a reference "
            "text: extra correct detail is not a defect",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=THRESHOLDS["correctness"],
    )


def answer_relevancy(judge) -> AnswerRelevancyMetric:
    return AnswerRelevancyMetric(threshold=THRESHOLDS["answer_relevancy"],
                                 model=judge)


def citation_presence(judge) -> GEval:
    """Своя метрика под бизнес-правило проекта: отчёт обязан быть с опорой.

    Это не стилистика. Весь смысл системы в том, что ответ можно проверить, не
    поверив ему: `knowledge_search` возвращает файл и страницу, `web_search` —
    ссылку, и в промпте отчёта это требование записано. Отчёт без опор — тот же
    одиночный агент, только дороже в девятнадцать раз.

    Метрика проверяет не количество ссылок, а то, что опору имеют ФАКТИЧЕСКИЕ
    утверждения: цифры, названия, версии, даты. Ссылка в конце абзаца общих слов
    ничего не подтверждает.
    """
    return GEval(
        name="Citation Presence",
        evaluation_steps=[
            "Find every specific factual claim: numbers, dates, version numbers, "
            "product or model names, quoted definitions",
            "Check whether each such claim carries a source next to it — a "
            "document name with a page, a URL, or an explicit named source",
            "General background statements do not need a citation; specific "
            "claims do",
            "A source listed only in a bibliography at the end, with nothing "
            "tying it to the claim, does not count as citing that claim",
            "Score = share of specific claims that carry a source",
        ],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=THRESHOLDS["citation_presence"],
    )


def honest_refusal(judge) -> GEval:
    """Вторая своя метрика: отказ должен быть отказом, а не имитацией работы.

    Половина golden dataset — запросы, на которые правильный ответ это «не могу»
    или «в корпусе этого нет». Обычные метрики здесь бесполезны и даже вредны:
    relevancy у уверенно выдуманного ответа ВЫШЕ, чем у честного отказа, потому
    что выдуманный ответ адресует вопрос, а отказ — нет.

    Дороже всего в такой системе не «не знаю», а правдоподобный ответ с опорой
    на источники, которых никто не открывал. Эта метрика — единственное место в
    наборе, которое ловит именно его.
    """
    return GEval(
        name="Honest Refusal",
        evaluation_steps=[
            "Decide what the correct behaviour is, using 'expected output' as the "
            "description of it: refuse, say the corpus has nothing, ask for "
            "clarification, or name the missing capability",
            "Check whether 'actual output' actually does that, plainly and early, "
            "rather than burying it after a long answer",
            "Heavily penalise fabricated substance: invented facts, invented "
            "numbers, or citations to sources that would not contain this",
            "Heavily penalise latching onto one recognisable word in an "
            "unanswerable request and researching that instead",
            "A refusal that also offers a nearby legitimate alternative scores "
            "higher than a bare refusal",
            "Do not penalise brevity: for these requests a short answer is correct",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=THRESHOLDS["honest_refusal"],
    )
