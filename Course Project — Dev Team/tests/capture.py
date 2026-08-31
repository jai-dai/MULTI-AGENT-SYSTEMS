"""Прогон команды записывается один раз; тесты читают запись.

# Почему не гонять команду в каждом тесте

Один прогон — это BA, разработчик и QA, каждый со своими инструментами, плюс
итерации ревью. Минуты и тысячи токенов. Метрик четыре, историй несколько;
прямолинейное «каждый тест поднимает команду» превращает `pytest` в получасовое
мероприятие, которое перестают запускать. Незапущенные тесты не имеют смысла.

Поэтому дорогая половина выполняется отдельной командой и ложится в
`runs/<id>.json`: спецификация, код, вердикт, история оценок, файлы. Дешёвая
половина — судьи — живёт в тестах.

Второй выигрыш важнее экономии: оценка становится ДЕТЕРМИНИРОВАННОЙ по входу.
Упавшая метрика означает, что изменилась метрика или порог, а не что разработчик
в этот раз выбрал другую структуру файлов.

Обратная сторона названа прямо: тесты проверяют ЗАПИСЬ, а не живую систему.
Запись устаревает, и обновлять её — отдельное решение.

    .venv/bin/python -m tests.capture --all
    .venv/bin/python -m tests.capture --id date-minmax --force
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as repl                                        # noqa: E402
import observability                                       # noqa: E402
from config import settings                                # noqa: E402
from team import build                                     # noqa: E402

RUNS = ROOT / "runs"
STORIES = json.loads((Path(__file__).parent / "stories.json").read_text("utf-8"))


def path_for(story_id: str) -> Path:
    return RUNS / f"{story_id}.json"


def load(story_id: str) -> dict | None:
    path = path_for(story_id)
    return json.loads(path.read_text("utf-8")) if path.exists() else None


async def capture(story: dict) -> dict:
    """Один прогон команды. Человека заменяет автоподтверждение.

    Это самое слабое место записи, и назвать его надо: HITL здесь всегда
    отвечает APPROVE, то есть проверено, что спецификация ДОШЛА до ворот, а не
    то, как система ведёт себя, когда человек возвращает её на доработку. Ту
    ветку тесты не покрывают и покрыть не могут — тест идёт без человека.
    """
    approvals: list[dict] = []

    def auto(payload: dict) -> dict:
        approvals.append({"version": payload.get("version"),
                          "saved_to": payload.get("saved_to")})
        print(f"  🤖 APPROVE (версия {payload.get('version')})", flush=True)
        return {"decision": "approve"}

    repl.ask_human = auto
    team = await build()
    state = await repl.run_once(team, story["user_story"], story["id"],
                                f"capture-{story['id']}")
    observability.flush()

    spec = state.get("spec")
    code = state.get("code")
    review = state.get("review")
    return {
        "id": story["id"],
        "category": story.get("category", ""),
        "user_story": story["user_story"],
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": settings.model_name,
        "spec": spec.model_dump() if spec else None,
        "code": code.model_dump() if code else None,
        "review": review.model_dump() if review else None,
        "iterations": state.get("iteration", 0) + 1,
        "scores": state.get("scores", []),
        "approvals": approvals,
        # Файлы читаются с диска: `files_created` — это ЗАЯВЛЕНИЕ разработчика, а
        # что лежит в каталоге — факт. Различать их важно: тест «код покрывает
        # спецификацию» должен смотреть на факт.
        "files": _workspace_files(),
    }


def _workspace_files() -> dict[str, str]:
    root = Path(settings.workspace_dir)
    if not root.exists():
        return {}
    out = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        out[path.relative_to(root).as_posix()] = path.read_text(
            "utf-8", errors="replace")[:20000]
    return out


def save(record: dict) -> Path:
    RUNS.mkdir(exist_ok=True)
    path = path_for(record["id"])
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


async def main_async(chosen: list[dict], force: bool) -> int:
    import shutil

    ok = 0
    for story in chosen:
        if not force and load(story["id"]):
            print(f"  ✓ {story['id']} — уже записан")
            ok += 1
            continue
        print(f"\n▶ {story['id']}: {story['user_story'][:70]}")
        # Чистый каталог на каждую историю: иначе файлы прошлой задачи
        # попадают в запись следующей и метрика «код покрывает спецификацию»
        # оценивает чужую работу.
        shutil.rmtree(settings.workspace_dir, ignore_errors=True)
        try:
            record = await capture(story)
        except Exception as exc:
            print(f"  ✗ упало: {type(exc).__name__}: {exc}")
            continue
        path = save(record)
        review = record["review"] or {}
        print(f"  💾 {path.name}  {review.get('verdict')} "
              f"score={review.get('score')} итераций={record['iterations']}")
        ok += 1
    print(f"\nготово: {ok} из {len(chosen)}")
    return 0 if ok == len(chosen) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Записать прогоны команды.")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.ids:
        chosen = [s for s in STORIES if s["id"] in args.ids]
    elif args.all:
        chosen = STORIES
    else:
        print("укажи --id или --all")
        return 2
    if not chosen:
        print("нечего записывать")
        return 2
    return asyncio.run(main_async(chosen, args.force))


if __name__ == "__main__":
    raise SystemExit(main())
