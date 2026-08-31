"""Общая обвязка: судья на прогон, записи и пропуск вместо падения.

Тест, для которого нет записанного прогона, НЕ КРАСНЫЙ — он не запускался.
Разница принципиальная: красный означает «стало хуже», отсутствие записи — «за
это ещё не платили». Свалить их в одну кучу значит приучить смотреть на красное
как на норму, и с этого начинается смерть любого набора тестов.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from tests import capture                                   # noqa: E402
from tests.judge import ProjectJudge                        # noqa: E402

SCORES: list[dict] = []
_BASELINE = ROOT / "runs" / "_scores.json"

STORIES = capture.STORIES
BY_ID = {s["id"]: s for s in STORIES}


@pytest.fixture(scope="session")
def judge() -> ProjectJudge:
    return ProjectJudge()


@pytest.fixture(scope="session", autouse=True)
def _banner(judge):
    print(f"\nсудья: {judge.get_model_name()}")
    yield
    print(f"\nоценка обошлась в {judge.spent()}")


def stories(*categories: str) -> list[dict]:
    if not categories:
        return STORIES
    return [s for s in STORIES if s.get("category") in categories]


def ids_of(chosen: list[dict]) -> list[str]:
    return [s["id"] for s in chosen]


def run_or_skip(story_id: str) -> dict:
    record = capture.load(story_id)
    if record is None:
        pytest.skip(f"нет записи для {story_id} — "
                    f".venv/bin/python -m tests.capture --id {story_id}")
    return record


def pytest_sessionfinish(session, exitstatus):
    if not SCORES:
        return
    previous = json.loads(_BASELINE.read_text("utf-8")) if _BASELINE.exists() else []

    # Слияние с прошлым прогоном нужно, чтобы запуск подмножества (-k) не стирал
    # остальную базовую линию. Но у пары «тест + метрика» не было срока годности:
    # когда тест сменил метрику, старая пара осталась в файле навсегда и попала в
    # таблицу как живое измерение. В отчёте это выглядело как провал метрики,
    # которую больше никто не считает. Поэтому: тест, отработавший в этой сессии,
    # полностью переопределяет свой набор метрик, а не дополняет его.
    ran_now = {r["test"] for r in SCORES}
    merged = {(r["test"], r["metric"]): r
              for r in previous if r["test"] not in ran_now}
    merged.update({(r["test"], r["metric"]): r for r in SCORES})
    rows = sorted(merged.values(), key=lambda r: (r["metric"], r["test"]))
    _BASELINE.parent.mkdir(exist_ok=True)
    _BASELINE.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    by_metric = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)

    print(f"\n{'метрика':26} {'порог':>6} {'n':>3} {'сред':>6} {'мин':>6} "
          f"{'макс':>6} {'прошло':>8}")
    print("-" * 66)
    broken = 0
    for name, group in sorted(by_metric.items()):
        values = [r["score"] for r in group if r["score"] is not None]
        broken += len(group) - len(values)
        threshold = group[0]["threshold"]
        if not values:
            print(f"{name[:26]:26} {threshold:>6} {len(group):>3} "
                  f"{'—':>6} {'—':>6} {'—':>6} {'—':>8}")
            continue
        ok = sum(r["passed"] for r in group if r["score"] is not None)
        print(f"{name[:26]:26} {threshold:>6} {len(values):>3} "
              f"{sum(values)/len(values):>6.2f} {min(values):>6.2f} "
              f"{max(values):>6.2f} {ok:>4}/{len(values):<3}")
    if broken:
        print(f"\n⚠️  {broken} метрик НЕ ИЗМЕРЕНО — это сбой оценки, а не "
              f"результат системы.")
    print(f"\nбаллы: {_BASELINE.relative_to(ROOT)}")
