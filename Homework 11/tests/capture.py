"""Прогон системы записывается один раз; тесты потом читают запись.

# Почему тесты не гоняют агентов

Один сквозной прогон hw9 стоил 951 808 токенов (замерено, см. README hw9).
Golden dataset по заданию — 15–20 примеров. Прямолинейное «`deepeval test run`
прогоняет датасет» стоит, таким образом, порядка пятнадцати миллионов токенов
за КАЖДЫЙ запуск тестов, включая тот, где ты поправил опечатку в пороге.

Это не «дороговато» — это делает тесты неповторяемыми, а значит бесполезными:
regression testing состоит в том, чтобы гонять их часто.

Поэтому цена разделена надвое. Дорогая половина — сам прогон — выполняется
здесь, руками и по одному примеру, и её результат ложится в `runs/<id>.json`.
Дешёвая половина — судьи — живёт в тестах и стоит сотни токенов на метрику
(замер: 478 за один GEval). Тесты читают запись и не поднимают ни одного агента.

Побочный выигрыш важнее экономии: оценка становится ДЕТЕРМИНИРОВАННОЙ по входу.
Упавшая метрика означает, что изменилась метрика или порог, а не что агент в
этот раз погуглил иначе. Иначе первый же красный тест уходит в «перезапусти,
может пройдёт», и смысл теряется весь.

Обратная сторона названа прямо: тесты проверяют ЗАПИСЬ, а не живую систему.
Запись устаревает, и обновлять её — отдельное решение и отдельные деньги.
Поэтому в каждом файле лежит дата, модель и версия кода, а `--force` существует.

# Два уровня записи и почему они разные

`planner` / `researcher` / `critic` собираются ЗДЕСЬ, в процессе тестов, и ходят
в SearchMCP напрямую. Так видно то, чего иначе не видно: какие инструменты агент
позвал и ЧТО ЕМУ ВЕРНУЛОСЬ. Второе — это retrieval context, без которого
groundedness неизмерима в принципе, а через ACP он остаётся в чужом процессе.

`e2e` идёт как положено, через супервизора и ACP, и записывает то, что видно
снаружи: отчёт, вызовы супервизора, вердикты, доработки, токены. Инструменты
суб-агентов сюда не попадают — и это честная граница, а не недоделка: снаружи
их и не видно.

    .venv/bin/python -m tests.capture --stage planner --all
    .venv/bin/python -m tests.capture --stage e2e --id rag-vs-long-context
    .venv/bin/python -m tests.capture --stage researcher --category happy_path
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from config import settings                                    # noqa: E402

DATASET = HERE / "golden_dataset.json"
RUNS = ROOT / "runs"

SEARCH_MCP_URL = f"http://{settings.protocol_host}:{settings.search_mcp_port}/mcp"

STAGES = ("planner", "researcher", "critic", "e2e")
# Что нужно иметь записанным ДО этой стадии. Критик оценивает находки, находки
# берутся из плана — цепочка та же, что у самой системы.
DEPENDS = {"researcher": "planner", "critic": "researcher"}


# --------------------------------------------------------------------------- #
# запись вызовов инструментов
# --------------------------------------------------------------------------- #

def recording_registry(registry: dict[str, Callable],
                       sink: list[dict]) -> dict[str, Callable]:
    """Тот же реестр, но каждый вызов оседает в `sink`.

    Обёртка снаружи, а не крючок внутри `ReactAgent`, сознательно: цикл агента
    приезжает сюда из hw9 через `sync_from_hw9.sh` и перезаписывается целиком.
    Строчка, дописанная в `react.py` ради тестов, исчезла бы при первой же
    синхронизации — молча, как исчезали строки в `requirements.txt`.

    Тесты не должны требовать правок в тестируемом. Здесь это не принцип, а
    механическое следствие того, как устроена цепочка копий.
    """
    def wrap(name: str, func: Callable) -> Callable:
        def recorded(**kwargs):
            started = time.monotonic()
            result = func(**kwargs)
            sink.append({
                "name": name,
                "args": kwargs,
                "result": result,
                "failed": str(result).startswith("ERROR"),
                "seconds": round(time.monotonic() - started, 2),
            })
            return result
        return recorded

    return {name: wrap(name, func) for name, func in registry.items()}


def retrieval_context(calls: list[dict]) -> list[str]:
    """Что реально вернулось из источников — вход для groundedness.

    Берутся ТОЛЬКО удачные вызовы поиска: строка «ERROR: SearchMCP не ответил»
    в контексте превратила бы метрику в проверку того, что агент не сослался на
    текст ошибки. Формально верно, по смыслу — мусор.
    """
    return [str(c["result"]) for c in calls
            if c["name"] in ("knowledge_search", "web_search", "read_url")
            and not c["failed"]]


# --------------------------------------------------------------------------- #
# хранилище
# --------------------------------------------------------------------------- #

def dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def path_for(example_id: str) -> Path:
    return RUNS / f"{example_id}.json"


def load(example_id: str) -> dict | None:
    path = path_for(example_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage(example_id: str, stage: str) -> dict | None:
    """Запись одной стадии, либо None. Именно это зовут тесты."""
    run = load(example_id)
    if run is None:
        return None
    return run.get("stages", {}).get(stage)


def save_stage(example: dict, stage: str, payload: dict) -> None:
    """Дописать стадию, не тронув остальные.

    Стадии пишутся по отдельности и в разные дни: researcher стоит сотен тысяч
    токенов, planner — тысяч. Перезапись файла целиком означала бы, что
    обновление дешёвой стадии стирает дорогую.
    """
    RUNS.mkdir(exist_ok=True)
    path = path_for(example["id"])
    run = load(example["id"]) or {
        "id": example["id"],
        "input": example["input"],
        "category": example["category"],
        "stages": {},
    }
    # Вход мог измениться в датасете — тогда старые стадии больше не про него.
    if run["input"] != example["input"]:
        print(f"  ⚠️  вход примера изменился — прежние стадии стёрты")
        run = {"id": example["id"], "input": example["input"],
               "category": example["category"], "stages": {}}

    run["stages"][stage] = {
        **payload,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": settings.model_name,
    }
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"  💾 {path.relative_to(ROOT)} → стадия '{stage}'")


# --------------------------------------------------------------------------- #
# стадии
# --------------------------------------------------------------------------- #

def _toolset(only: list[str]):
    from mcp_utils import McpToolset
    return McpToolset(SEARCH_MCP_URL, label="SearchMCP", only=only)


# ВНИМАНИЕ на имя поля: `agent.calls` считает ВЫЗОВЫ ИНСТРУМЕНТОВ, а не шаги
# цикла — за один шаг модель вправе позвать несколько. Сравнивать записанное
# число с `MAX_STEPS` нельзя, оно легко больше при совершенно здоровом прогоне.
# Число шагов наружу не торчит вовсе: это локальная переменная цикла в
# `ReactAgent.run`, и добавлять её ради записи значило бы править код системы.


def _dump(value: Any) -> Any:
    """Модель данных -> JSON-совместимое; текст остаётся текстом."""
    return value.model_dump() if hasattr(value, "model_dump") else value


def capture_planner(example: dict) -> dict:
    from agents import planner

    toolset = _toolset(planner.TOOLS)
    try:
        calls: list[dict] = []
        agent = planner.build(toolset)
        agent.registry = recording_registry(agent.registry, calls)
        plan = agent.run(example["input"])
        return {"output": _dump(plan), "tool_calls": calls,
                "tokens": agent.tokens, "steps": agent.calls}
    finally:
        toolset.close()


def capture_researcher(example: dict, run: dict) -> dict:
    from agents import research
    from supervisor import render_plan
    from schemas import ResearchPlan

    plan = ResearchPlan.model_validate(run["stages"]["planner"]["output"])
    # Формулировка ровно та же, что у супервизора: исходный запрос едет вместе с
    # планом каждый раунд, иначе исследователь уезжает в план и теряет человека.
    instructions = (f"ORIGINAL USER REQUEST: {example['input']}\n\n"
                    f"{render_plan(plan)}")

    toolset = _toolset(research.TOOLS)
    try:
        calls: list[dict] = []
        agent = research.build(toolset)
        agent.registry = recording_registry(agent.registry, calls)
        findings = agent.run(instructions)
        return {"output": findings, "tool_calls": calls,
                "retrieval_context": retrieval_context(calls),
                "instructions": instructions,
                "tokens": agent.tokens, "steps": agent.calls}
    finally:
        toolset.close()


def capture_critic(example: dict, run: dict) -> dict:
    from agents import critic

    findings = run["stages"]["researcher"]["output"]
    request = (f"ORIGINAL USER REQUEST: {example['input']}\n\n"
               f"FINDINGS:\n{findings}")

    toolset = _toolset(critic.TOOLS)
    try:
        calls: list[dict] = []
        # Дата фиксируется в записи: критик судит о свежести, и через полгода
        # «устарело» будет означать другое. Без штампа запись стала бы
        # невоспроизводимой ровно в том поле, ради которого критик и нужен.
        today = datetime.now(timezone.utc).date().isoformat()
        agent = critic.build(toolset, today=today)
        agent.registry = recording_registry(agent.registry, calls)
        verdict = agent.run(request)
        return {"output": _dump(verdict), "tool_calls": calls,
                "input_findings": findings, "today": today,
                "tokens": agent.tokens, "steps": agent.calls}
    finally:
        toolset.close()


def capture_e2e(example: dict) -> dict:
    """Полный конвейер через ACP. Человека в контуре заменяет автоподтверждение.

    Это самая слабая точка всей затеи, и назвать её надо вслух: HITL в hw9 —
    место, где систему останавливает ЧЕЛОВЕК, а тест по определению идёт без
    человека. Автоподтверждение проверяет, что супервизор дошёл до `save_report`
    с готовым отчётом, — и ничего не говорит о том, как система ведёт себя,
    когда человек говорит «нет». Ту ветку тесты не покрывают (см. README).
    """
    from supervisor import Supervisor

    supervisor = Supervisor()
    approvals: list[dict] = []
    calls: list[dict] = []
    try:
        save = supervisor.reports.registry["save_report"]

        def auto_approve(name: str, arguments: str) -> str | None:
            if name != "save_report":
                return None                    # None — исполнять как обычно
            try:
                data = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return f"ERROR: arguments are not valid JSON ({exc})."
            result = save(**data)              # запись реальная, файл в output/
            approvals.append({"filename": data.get("filename"),
                              "content": data.get("content", ""),
                              "result": result})
            # Записать вызов ЗДЕСЬ, а не полагаться на обёртку реестра. Крючок
            # стоит ПЕРЕД реестром (`ReactAgent._dispatch`), и вернув строку, он
            # до реестра не доходит вовсе — обёртка этого вызова не увидела бы.
            # Ровно та особенность, на которой держится HITL: гейт стоит между
            # моделью и вызовом, а не между вызовом и файлом.
            calls.append({"name": name, "args": data, "result": result,
                          "failed": str(result).startswith("ERROR"),
                          "seconds": 0.0, "via": "interceptor"})
            print(f"  🤖 автоподтверждение: {result}")
            return result

        supervisor.agent.interceptor = auto_approve
        supervisor.agent.registry = recording_registry(supervisor.agent.registry,
                                                       calls)
        started = time.monotonic()
        answer = supervisor.run(example["input"])

        return {
            "output": answer,
            "tool_calls": calls,
            "approvals": approvals,
            "report": approvals[-1]["content"] if approvals else "",
            "revisions": supervisor.revisions,
            "plan": _dump(supervisor.plan) if supervisor.plan else None,
            "last_critique": (_dump(supervisor.last_critique)
                              if supervisor.last_critique else None),
            "tokens": {"supervisor": supervisor.agent.tokens,
                       **supervisor.agent_tokens,
                       "total": supervisor.total_tokens},
            "seconds": round(time.monotonic() - started, 1),
        }
    finally:
        supervisor.close()


CAPTURE = {
    "planner": lambda example, run: capture_planner(example),
    "researcher": capture_researcher,
    "critic": capture_critic,
    "e2e": lambda example, run: capture_e2e(example),
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def ensure(example: dict, stage: str, force: bool = False) -> bool:
    """Записать стадию, вместе со всем, от чего она зависит."""
    run = load(example["id"]) or {"stages": {}}
    if not force and stage in run.get("stages", {}):
        print(f"  ✓ {example['id']} / {stage} — уже записано")
        return True

    need = DEPENDS.get(stage)
    if need and need not in run.get("stages", {}):
        print(f"  ↳ сначала '{need}' — от него зависит '{stage}'")
        if not ensure(example, need):
            return False
        run = load(example["id"])

    print(f"\n▶ {example['id']} / {stage}")
    try:
        payload = CAPTURE[stage](example, run)
    except Exception as exc:
        print(f"  ✗ упало: {type(exc).__name__}: {exc}")
        return False
    save_stage(example, stage, payload)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Записать прогоны системы для тестов hw10.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--id", action="append", dest="ids",
                        help="конкретный пример (можно несколько раз)")
    parser.add_argument("--category", help="все примеры категории")
    parser.add_argument("--all", action="store_true", help="весь датасет")
    parser.add_argument("--force", action="store_true",
                        help="перезаписать уже записанное")
    args = parser.parse_args()

    examples = dataset()
    if args.ids:
        chosen = [e for e in examples if e["id"] in args.ids]
        unknown = set(args.ids) - {e["id"] for e in chosen}
        if unknown:
            print(f"нет таких примеров: {', '.join(sorted(unknown))}")
            return 2
    elif args.category:
        chosen = [e for e in examples if e["category"] == args.category]
    elif args.all:
        chosen = examples
    else:
        print("укажи --id, --category или --all")
        return 2

    if not chosen:
        print("нечего записывать")
        return 2

    print(f"стадия '{args.stage}', примеров: {len(chosen)}, модель "
          f"{settings.model_name}")
    ok = sum(ensure(e, args.stage, args.force) for e in chosen)
    print(f"\nготово: {ok} из {len(chosen)}")
    return 0 if ok == len(chosen) else 1


if __name__ == "__main__":
    raise SystemExit(main())
