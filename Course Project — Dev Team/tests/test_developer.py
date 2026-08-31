"""Разработчик: файлы существуют и импортируются — кодом; покрытие — судьёй."""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

import sandbox
from config import settings
from tests import metrics
from tests.conftest import ids_of, run_or_skip, stories
from tests.scored import assert_scored
from tests.context import render_files, render_requirements

SOLID = stories("happy_path", "medium")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_files_exist(story):
    """`files_created` — это ЗАЯВЛЕНИЕ, а каталог — факт. Сверяем их.

    Разработчик может перечислить файл, которого не записал: в записи есть и
    список, и содержимое каталога, и разойтись они могут молча.
    """
    record = run_or_skip(story["id"])
    claimed = record["code"]["files_created"]
    actual = set(record["files"])
    assert actual, "в рабочем каталоге нет ни одного .py файла"
    missing = [f for f in claimed
               if f not in actual and f.split("/")[-1] not in
               {a.split("/")[-1] for a in actual}]
    assert not missing, (
        f"разработчик заявил файлы, которых нет на диске: {missing}. "
        f"Фактически: {sorted(actual)}")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_code_imports(story):
    """Записанный код должен импортироваться. Запускаем по-настоящему.

    Проверка не теоретическая: ровно на ней первый прогон системы потерял пять
    итераций — файлы писались, но не импортировались, потому что песочница не
    клала рабочий каталог на `sys.path`.
    """
    record = run_or_skip(story["id"])
    modules = [f[:-3].replace("/", ".") for f in record["files"]
               if f.endswith(".py") and not f.startswith("test")]
    assert modules, "нет ни одного модуля, кроме тестов"

    # Файлы восстанавливаются из записи: тест не зависит от того, что сейчас
    # лежит в рабочем каталоге после других прогонов.
    from pathlib import Path
    import shutil, tempfile
    workdir = Path(tempfile.mkdtemp(prefix="devteam-test-"))
    try:
        for name, content in record["files"].items():
            target = workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        code = "\n".join(f"import {m}" for m in modules) + "\nprint('imported ok')"
        result = sandbox.run(code, workdir, timeout=20)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    assert result.ok, (f"модули не импортируются: {modules}\n"
                       f"{result.as_text()[:600]}")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_code_covers_spec(story, judge):
    record = run_or_skip(story["id"])
    spec = record["spec"]
    requirements = render_requirements(spec["requirements"])
    body = render_files(record["files"])
    assert_scored(
        LLMTestCase(input=story["user_story"], actual_output=body,
                    expected_output=requirements),
        [metrics.code_covers_spec(judge)])
