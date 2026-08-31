"""Запуск кода, который написала модель. Отдельным процессом и с ограничениями.

# Почему отдельный процесс, а не чёрный список модулей

Задание перечисляет минимум: timeout, запрет `os`/`subprocess`/`shutil`, лимит на
размер вывода. Чёрный список — самая слабая из трёх мер: `__import__("os")`,
`importlib`, `__class__.__mro__` — обходов столько, что перечислять безнадёжно.

Работает другое: код исполняется в ЧУЖОМ процессе со своим рабочим каталогом,
своим лимитом памяти и своим таймаутом. Что бы он ни сделал, он не уронит граф,
не съест память машины и не зациклится навсегда — процесс просто убивают. Это
граница, которую нельзя обойти изнутри, потому что она снаружи.

# Чего стоил слишком широкий чёрный список — замерено

Первая версия блокировала импорт ГЛОБАЛЬНО, и это сломало задачу целиком.
`unittest`, `runpy`, `tempfile`, `logging` импортируют `os` внутри себя — то есть
агент физически не мог написать тест на unittest. Вдобавок рабочий каталог не
попадал на `sys.path` (флаг `-I` убирает ""), и только что записанный файл не
импортировался.

Прогон это показал в цифрах: пять итераций подряд разработчик искал обход —
`src/`, `main.py`, даже `sitecustomize.py`, — QA честно писал «файлы не
импортируются», оценка шла 0.70 → 0.75 → 0.70 → 0.72 → 0.60. Агенты вели себя
правильно. Невыполнимой задачу сделала песочница.

Отсюда два исправления: рабочий каталог кладётся на путь импорта явно, а запрет
действует ТОЛЬКО на код агента — по вызывающему кадру, — и не мешает
стандартной библиотеке делать свои дела. `pathlib` из списка убран: он не даёт
ничего сверх `open()`.

# Чего эта песочница НЕ делает

Сеть не закрыта. Файловую систему видно за пределами рабочего каталога. Для
учебной системы, которая пишет функции по user story, это приемлемо; для запуска
чужого кода из интернета — нет. Сказать прямо честнее, чем оставить впечатление
настоящей изоляции.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_SECONDS = 20
MEMORY_LIMIT_MB = 512
OUTPUT_LIMIT_CHARS = 8000

# Недоступно КОДУ АГЕНТА. Внутренностям стандартной библиотеки — доступно.
BLOCKED = ("os", "subprocess", "shutil", "socket", "requests", "urllib",
           "http", "ctypes", "multiprocessing")

_PREAMBLE = '''
import sys as _sys
# Рабочий каталог — на путь импорта. Без этой строки файл, который агент только
# что записал, не импортируется: `-I` убирает "" из sys.path.
if "" not in _sys.path:
    _sys.path.insert(0, "")

import builtins as _b
import posixpath as _pp
_BLOCKED = {blocked!r}
_ROOT = _pp.abspath(".")
_real_import = _b.__import__

def _agent_code(depth=2):
    """Импорт идёт из кода агента, а не из недр стандартной библиотеки?"""
    try:
        frame = _sys._getframe(depth)
    except ValueError:
        return True
    name = frame.f_globals.get("__name__", "")
    path = frame.f_globals.get("__file__", "") or ""
    if name == "__main__":
        return True
    if not path:
        return False
    return _pp.abspath(path).startswith(_ROOT)

def _guarded(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _BLOCKED and _agent_code():
        raise ImportError(
            f"module {{root!r}} is not available to code running in this "
            "sandbox. Work within the current directory; the network and the "
            "process table are out of reach here.")
    return _real_import(name, *args, **kwargs)

_b.__import__ = _guarded

import resource as _r
_limit = {memory} * 1024 * 1024
try:
    _r.setrlimit(_r.RLIMIT_AS, (_limit, _limit))
except (ValueError, OSError):
    pass          # на macOS RLIMIT_AS иногда не применяется; таймаут остаётся
del _r, _limit
'''


@dataclass
class Result:
    """Что вернул запуск. Никаких исключений наружу: результат — это данные."""

    ok: bool
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    timeout_seconds: int = TIMEOUT_SECONDS

    def as_text(self) -> str:
        if self.timed_out:
            return (f"TIMEOUT: код не завершился за {self.timeout_seconds} с и "
                    f"был остановлен. Скорее всего бесконечный цикл или "
                    f"ожидание ввода.\n--- stdout ---\n{self.stdout}")
        parts = [f"exit code: {self.exit_code}"]
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        if not self.stdout and not self.stderr:
            parts.append("(ни stdout, ни stderr — код отработал молча)")
        return "\n".join(parts)


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT_CHARS:
        return text
    return text[:OUTPUT_LIMIT_CHARS] + f"\n… обрезано, всего {len(text)} символов"


def run(code: str, workdir: Path | str, timeout: int = TIMEOUT_SECONDS) -> Result:
    """Выполнить код и вернуть результат. Не бросает исключений НИКОГДА."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    program = _PREAMBLE.format(blocked=list(BLOCKED), memory=MEMORY_LIMIT_MB) \
        + "\n" + textwrap.dedent(code)

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program],
            cwd=workdir, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or b""
        return Result(ok=False, timed_out=True, exit_code=None,
                      timeout_seconds=timeout,
                      stdout=_clip(out.decode(errors="replace")
                                   if isinstance(out, bytes) else str(out)),
                      stderr="")
    except Exception as exc:
        return Result(ok=False, stdout="", exit_code=None,
                      stderr=f"песочница не смогла запустить код: "
                             f"{type(exc).__name__}: {exc}")

    return Result(ok=completed.returncode == 0,
                  stdout=_clip(completed.stdout),
                  stderr=_clip(completed.stderr),
                  exit_code=completed.returncode)
