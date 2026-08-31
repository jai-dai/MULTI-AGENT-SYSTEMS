"""Свой ассерт вместо `deepeval.assert_test`, потому что баллы нужны не меньше вердикта.

`assert_test` измеряет и падает, если ниже порога, но балл забирает из внутренней
копии и НА ПЕРЕДАННОМ объекте метрики его не оставляет: после вызова
`metric.score` по-прежнему `None`. Проверено в предыдущей работе.

Для бинарного «прошло / не прошло» этого хватает. Для базовой линии — нет:
«зелено при пороге 0.7» не отличает 0.71 от 0.98, а именно эта разница говорит,
можно ли поднимать порог.

Побочная выгода: вердикт выносится по ВСЕМ метрикам сразу. `assert_test` падает
на первой непройденной, и вторая остаётся неизмеренной ровно тогда, когда она
интереснее всего.
"""
from __future__ import annotations

import os


def current_test() -> str:
    raw = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
    return raw.split("::")[-1].split(" (")[0]


def assert_scored(test_case, metric_list) -> None:
    from tests.conftest import SCORES

    failed = []
    for metric in metric_list:
        name = getattr(metric, "__name__", type(metric).__name__)
        try:
            metric.measure(test_case)
        except Exception as exc:
            SCORES.append({"test": current_test(), "metric": name, "score": None,
                           "threshold": getattr(metric, "threshold", None),
                           "passed": False,
                           "reason": f"МЕТРИКА УПАЛА: {type(exc).__name__}: {exc}"})
            failed.append(f"{name}: метрика упала — {type(exc).__name__}: {exc}")
            continue

        score, threshold = metric.score, metric.threshold
        passed = bool(getattr(metric, "success", None)
                      if getattr(metric, "success", None) is not None
                      else (score is not None and score >= threshold))
        SCORES.append({
            "test": current_test(), "metric": name,
            "score": None if score is None else round(float(score), 3),
            "threshold": threshold, "passed": passed,
            "reason": (metric.reason or "")[:1500] if getattr(metric, "reason", None) else "",
        })
        if not passed:
            failed.append(f"{name}: {score} < {threshold} — {metric.reason}")

    assert not failed, "\n".join(failed)
