"""Свой ассерт вместо `assert_test`, потому что баллы нужны не меньше вердикта.

# Почему не `deepeval.assert_test`

Он делает ровно то, что обещает: измеряет и падает, если ниже порога. Но балл он
забирает из внутренней копии результата и **на переданном объекте метрики его не
оставляет** — после `assert_test(case, [metric])` у `metric.score` по-прежнему
`None`. Проверено, а не предположено.

Для тестов этого хватает: они бинарны. Для БАЗОВОЙ ЛИНИИ — нет. Лекция называет
завышенный порог антипаттерном и предписывает порядок «сначала baseline, потом
повышение», а baseline — это сами цифры. «Зелено при пороге 0.5» не отличает
0.51 от 0.98, а разница между ними и есть весь смысл замера: в первом случае
поднимать порог нельзя, во втором нужно.

Поэтому здесь метрика измеряется на месте, балл записывается, и только потом
выносится вердикт. Побочная выгода — вердикт по ВСЕМ метрикам сразу:
`assert_test` падает на первой непройденной, и вторая остаётся неизмеренной,
то есть отсутствующей в базовой линии ровно тогда, когда она интереснее всего.
"""
from __future__ import annotations

import os


def current_test() -> str:
    """Имя текущего теста. pytest кладёт его в окружение сам."""
    raw = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
    # 'tests/test_x.py::test_y[case] (call)' -> 'test_y[case]'
    return raw.split("::")[-1].split(" (")[0]


def assert_scored(test_case, metric_list) -> None:
    """Измерить все метрики, записать баллы, потом упасть — если есть на чём."""
    from tests.conftest import SCORES

    failed = []
    for metric in metric_list:
        name = getattr(metric, "__name__", type(metric).__name__)
        try:
            metric.measure(test_case)
        except Exception as exc:
            # Сбой самой метрики — это не «система плохая». Записывается
            # отдельно и валит тест с ясным текстом, а не нулевым баллом,
            # который в таблице неотличим от настоящего провала.
            SCORES.append({"test": current_test(), "metric": name, "score": None,
                           "threshold": getattr(metric, "threshold", None),
                           "passed": False,
                           "reason": f"МЕТРИКА УПАЛА: {type(exc).__name__}: {exc}"})
            failed.append(f"{name}: метрика упала — {type(exc).__name__}: {exc}")
            continue

        score = metric.score
        threshold = metric.threshold
        passed = bool(getattr(metric, "success", None)
                      if getattr(metric, "success", None) is not None
                      else (score is not None and score >= threshold))
        SCORES.append({
            "test": current_test(), "metric": name,
            "score": None if score is None else round(float(score), 3),
            "threshold": threshold, "passed": passed,
            "reason": (metric.reason or "")[:400] if getattr(metric, "reason", None) else "",
        })
        if not passed:
            failed.append(f"{name}: {score} < {threshold} — {metric.reason}")

    assert not failed, "\n".join(failed)
