"""Общая обвязка тестов: судья на весь прогон, датасет и пропуски по записям.

# Почему пропуск, а не падение

Тест, для которого нет записанного прогона, не красный — он не запускался.
Разница принципиальная: красный тест означает «система стала хуже», а
отсутствие записи означает «за это ещё не заплатили». Свалить их в одну кучу
значит приучить смотреть на красное как на норму, и с этого начинается смерть
любого набора тестов.

Поэтому пропуск сообщает РОВНО ту команду, которой он лечится. Стоимость записи
названа там же: решение «записывать ли» — денежное, и принимать его вслепую
не надо.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Телеметрия DeepEval уходит на их сервер вместе с именами метрик. Выключается
# до импорта самого пакета, иначе клиент успевает подняться.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from tests import capture                                       # noqa: E402
from tests.judge import ProjectJudge                            # noqa: E402


# --------------------------------------------------------------------------- #
# судья
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def judge() -> ProjectJudge:
    """Один судья на весь прогон — он же счётчик потраченного на оценку."""
    return ProjectJudge()


@pytest.fixture(scope="session", autouse=True)
def _judge_banner(judge, request):
    """Сказать вслух, кто судит, и сколько это стоило.

    Не украшение: если судья окажется той же моделью, что и система, все оценки
    в прогоне завышены, и знать об этом надо ДО того, как читать цифры.
    """
    print(f"\nсудья: {judge.get_model_name()}")
    if judge.same_vendor_as_system():
        print("  ⚠️  судья и система у ОДНОГО вендора — self-enhancement bias, "
              "оценки завышены (лекция 11, антипаттерн 3). "
              "Задай JUDGE_MODEL_NAME в .env.")
    yield
    print(f"\nоценка обошлась в {judge.spent()}")


# --------------------------------------------------------------------------- #
# датасет и записи
# --------------------------------------------------------------------------- #

DATASET = capture.dataset()
BY_ID = {e["id"]: e for e in DATASET}


def examples(*categories: str) -> list[dict]:
    if not categories:
        return DATASET
    return [e for e in DATASET if e["category"] in categories]


def ids_of(chosen: list[dict]) -> list[str]:
    return [e["id"] for e in chosen]


def stage_or_skip(example_id: str, stage: str) -> dict:
    """Запись стадии, либо пропуск теста с командой, которая её создаст."""
    recorded = capture.load_stage(example_id, stage)
    if recorded is None:
        pytest.skip(
            f"нет записи '{stage}' для {example_id} — "
            f".venv/bin/python -m tests.capture --stage {stage} --id {example_id}")
    return recorded


def tool_calls_as(recorded: dict):
    """Записанные вызовы -> `ToolCall` DeepEval."""
    from deepeval.test_case import ToolCall
    return [
        ToolCall(name=call["name"],
                 input_parameters=call["args"],
                 output=str(call["result"])[:4000])
        for call in recorded.get("tool_calls", [])
    ]


def called_names(recorded: dict) -> list[str]:
    return [call["name"] for call in recorded.get("tool_calls", [])]


# --------------------------------------------------------------------------- #
# сбор баллов: базовая линия — это сами цифры, а не «зелено»
# --------------------------------------------------------------------------- #

# Сюда пишет `tests/scored.py` — по строке на каждую измеренную метрику.
SCORES: list[dict] = []
_BASELINE = ROOT / "runs" / "_baseline_scores.json"


def pytest_sessionfinish(session, exitstatus):
    """Свести баллы в таблицу и дописать её на диск.

    Дописать, а не перезаписать: базовая линия набирается по частям, по мере
    того как записываются прогоны. Прогон `-k rag-vs-long-context` не должен
    стирать всё, что замерено вчера.
    """
    if not SCORES:
        return

    import json
    from collections import defaultdict

    previous = []
    if _BASELINE.exists():
        previous = json.loads(_BASELINE.read_text(encoding="utf-8"))
    # Ключ — тест плюс метрика: повторный прогон того же теста ОБНОВЛЯЕТ строку,
    # а не плодит вторую. Иначе среднее поедет от количества запусков.
    merged = {(row["test"], row["metric"]): row for row in previous}
    merged.update({(row["test"], row["metric"]): row for row in SCORES})
    rows = sorted(merged.values(), key=lambda r: (r["metric"], r["test"]))
    _BASELINE.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    by_metric = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)

    print(f"\n{'метрика':26} {'порог':>6} {'n':>3} {'сред':>6} "
          f"{'мин':>6} {'макс':>6} {'прошло':>8}")
    print("-" * 66)
    for name, group in sorted(by_metric.items()):
        scores = [r["score"] for r in group]
        threshold = group[0]["threshold"]
        ok = sum(r["passed"] for r in group)
        print(f"{name[:26]:26} {threshold:>6} {len(group):>3} "
              f"{sum(scores)/len(scores):>6.2f} {min(scores):>6.2f} "
              f"{max(scores):>6.2f} {ok:>4}/{len(group):<3}")
    print(f"\nбаллы: {_BASELINE.relative_to(ROOT)}")
