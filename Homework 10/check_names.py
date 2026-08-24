"""Не дать личным именам уехать в публичный репозиторий.

Запускается перед push и падает, если нашёл. Список имён живёт ОТДЕЛЬНО и в
репозиторий не попадает — иначе проверка сама стала бы утечкой, аккуратно
собранной в один файл.

Почему это код, а не дисциплина: маскировку делали вручную и внимательно, и она
всё равно пропустила фамилию в склонённой форме — в списке стояла латиница, а в
комментарии кириллица с падежным окончанием. Правило, которое держится на
памяти, нарушается ровно тогда, когда о нём забыли.

Поэтому сопоставление идёт **по основе, с границей слова слева**: основа
`Іванен` находит «Іваненка», «Іваненкові», «Іваненко», но `test` не находится
внутри `latest`. Та же логика, что в importance.py, и по той же причине —
падежи перечислять руками нельзя, первый же непредусмотренный пройдёт насквозь.

    python check_names.py --names ~/.private-names.txt "путь/к/дереву"

Коды выхода: 0 — чисто, 1 — найдено, 2 — не с чем сравнивать.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "index", "mail"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".docx", ".xlsx",
                 ".pptx", ".zip", ".faiss", ".bin", ".so", ".dylib"}


def load_names(path: Path) -> list[str]:
    if not path.exists():
        print(f"ERROR: нет файла со списком имён: {path}", file=sys.stderr)
        raise SystemExit(2)
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    if not names:
        print(f"ERROR: список имён пуст: {path}", file=sys.stderr)
        raise SystemExit(2)
    return names


def scan(root: Path, names: list[str]) -> list[tuple[Path, int, str, str]]:
    patterns = [(name, re.compile(r"\b" + re.escape(name), re.IGNORECASE))
                for name in names]
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                     # бинарь или недоступен — не наше дело
        for number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in patterns:
                match = pattern.search(line)
                if match:
                    found.append((path.relative_to(root), number, name,
                                  line.strip()[:100]))
                    break                # одна находка на строку достаточно
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="каталоги для проверки")
    parser.add_argument("--names", required=True,
                        help="файл со списком имён (одно на строку, # комментарий)")
    args = parser.parse_args()

    names = load_names(Path(args.names).expanduser())
    hits = []
    for raw in args.paths:
        root = Path(raw).expanduser()
        if not root.exists():
            print(f"ERROR: нет каталога: {root}", file=sys.stderr)
            return 2
        hits += scan(root, names)

    if not hits:
        print(f"✅ проверено по {len(names)} именам — чисто")
        return 0

    print(f"⛔ НАЙДЕНО {len(hits)} совпадений — push остановлен\n", file=sys.stderr)
    for path, number, name, line in hits[:40]:
        print(f"  {path}:{number}  ({name})\n      {line}", file=sys.stderr)
    if len(hits) > 40:
        print(f"  … и ещё {len(hits) - 40}", file=sys.stderr)
    print("\nЗамаскируйте найденное или добавьте файл в исключения rsync.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
