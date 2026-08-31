"""Метрики и пороги в одном месте.

Порог — утверждение о том, какое качество считается достаточным. Разбросанный по
файлам, он перестаёт быть утверждением и становится числом, которое правят, когда
тест мешает.

Пороги низкие намеренно: сначала базовая линия, потом повышение. Завышенный порог
с первого дня — тесты всегда красные, и их выключают.
"""
from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

THRESHOLDS = {
    "spec_quality": 0.7,
    "code_covers_spec": 0.7,
    "review_is_useful": 0.7,
    "review_completeness": 0.7,
    "no_invention": 0.7,
    "solves_user_story": 0.7,
}


def spec_quality(judge) -> GEval:
    """Спецификация пригодна для работы — а не просто выглядит спецификацией."""
    return GEval(
        name="Spec Quality",
        evaluation_steps=[
            "Check that every requirement is ONE testable statement, not a "
            "paragraph and not a vague quality ('works well', 'is robust')",
            "Check that each acceptance criterion names a concrete input and an "
            "expected outcome, so it could be checked by running the code",
            "Check that error handling and input validation appear when the "
            "user story implies them",
            "Penalise requirements that dictate the implementation (a specific "
            "library or algorithm) — that is the developer's decision",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge, threshold=THRESHOLDS["spec_quality"],
    )


def code_covers_spec(judge) -> GEval:
    """Каждое требование реализовано — проверяется против спецификации."""
    return GEval(
        name="Code Covers Spec",
        evaluation_steps=[
            "Take each requirement from 'expected output' (the specification) "
            "in turn and find where 'actual output' implements it",
            "A requirement is covered only if the code actually does it, not if "
            "the description claims it does",
            "Penalise missing error handling that the specification required",
            "Score = share of requirements genuinely covered",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        model=judge, threshold=THRESHOLDS["code_covers_spec"],
    )


def review_is_useful(judge) -> GEval:
    """Ревью находит реальное и говорит, что делать.

    Эта метрика ловит самую дорогую поломку цикла: QA, который отклоняет код
    расплывчатыми претензиями. Разработчик не может на них ответить, итерации
    сгорают, и предел исчерпывается без движения.
    """
    return GEval(
        name="Review Is Useful",
        evaluation_steps=[
            "Check that each issue names WHAT is wrong and WHERE, not a general "
            "impression ('code quality is poor')",
            "Check that each suggestion is actionable: a developer could act on "
            "it without asking what was meant",
            "If the verdict is APPROVED, issues should be empty or minor",
            "If the verdict is REVISION_NEEDED, there must be at least one "
            "concrete issue",
            "Penalise rejection on style alone when behaviour is correct",
            "Judge ONLY the issues and suggestions the review actually raises. "
            "Whether the review missed some other defect is out of scope here "
            "and is measured separately by Review Completeness",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge, threshold=THRESHOLDS["review_is_useful"],
    )


def review_completeness(judge) -> GEval:
    """Ревью замечает то, что реально сломано.

    Метрику родил ложный провал review_is_useful. Судье показывали код, но
    просили оценивать только формулировки — и он всё равно уходил проверять,
    что QA пропустил. Значит, сигнал нужный, просто спрошен не тем вопросом:
    полнота и дееспособность ревью — разные вещи, и мерить их надо порознь.
    Здесь спрашиваем прямо: есть ли в коде нарушение требования, которого
    ревью не увидело.
    """
    return GEval(
        name="Review Completeness",
        evaluation_steps=[
            "Read the requirements and the code, and list defects where the "
            "code actually violates a stated requirement",
            "A defect counts only if it contradicts an explicit requirement; "
            "preferences, style and unstated conventions do not count",
            "Requirement order is not a requirement unless a rule says so",
            "Check whether the review names each such defect",
            "Score high when the review names every real violation, low when "
            "it approves code that breaks a stated requirement",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge, threshold=THRESHOLDS["review_completeness"],
    )


def no_invention(judge) -> GEval:
    """Аналитик не выдумывает предмет там, где его не назвали.

    Отдельная метрика понадобилась потому, что этот тест мерили spec_quality,
    и судья мешал в один балл претензии не по делу: формат критериев приёмки,
    протекшие HTTP-коды. Оба замечания к выдумыванию отношения не имеют, а балл
    занижали. Вопрос теста один — придуман ли предмет, — и мерить надо его.
    """
    return GEval(
        name="No Invention",
        evaluation_steps=[
            "Decide whether the user story actually names a subject: a thing "
            "to build with recognisable boundaries",
            "If the subject is missing, correct behaviour is to say so plainly "
            "and list the open questions that block the work",
            "Score low when the response invents concrete scope nobody asked "
            "for: entities, operations, interfaces, storage decisions",
            "Appending open questions AFTER inventing a system does not repair "
            "the invention",
            "Judge only invention. Formatting of the spec, acceptance-criteria "
            "style and leaked implementation details are out of scope here",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge, threshold=THRESHOLDS["no_invention"],
    )


def solves_user_story(judge) -> GEval:
    """Сквозная: финальный код отвечает на то, с чего всё начиналось.

    Отдельно от `code_covers_spec` не случайно. Код может точно исполнить
    спецификацию, а спецификация — разойтись с исходной просьбой. Первая метрика
    проверяет исполнение, эта — что исполняли то самое.
    """
    return GEval(
        name="Solves User Story",
        evaluation_steps=[
            "Read the original user story in 'input' and the final code in "
            "'actual output'",
            "Decide whether someone who asked for this would consider it done",
            "Check the obvious edge cases the story implies: empty input, "
            "invalid input, boundary values",
            "Ignore style and naming; judge behaviour",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge, threshold=THRESHOLDS["solves_user_story"],
    )
