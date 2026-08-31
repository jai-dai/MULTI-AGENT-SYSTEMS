"""Собрать корпус документации команды. Один раз, перед индексацией.

    .venv/bin/python build_corpus.py

# Что здесь считается «документацией команды»

Не всё подряд из интернета, а три вещи, которые бизнес-аналитик обязан свёрить
перед тем, как писать спецификацию:

- **стандарт кодирования** — по нему проверяют требования вроде «валидация
  входных данных» и «обработка ошибок»: в спецификации должно стоять то, что
  команда действительно делает, а не то, что модель считает хорошим тоном;
- **документация языка** — что есть в стандартной библиотеке, чтобы не
  требовать зависимости там, где хватит `dataclasses`;
- **документация фреймворка** — как выглядит типовая реализация здесь.

Корпус предыдущих работ (три статьи про RAG и LLM) для этого не годится: он
отвечает на вопрос «что такое RAG», а не «как здесь пишут код».

# Почему через собственный read_url

`tools.read_url` уже умеет то, что нужно: забрать страницу и вернуть читаемый
текст без навигации и футеров (trafilatura). Тащить ради корпуса второй
загрузчик значило бы иметь две разные чистки и однажды удивиться, почему в
индексе половина документа — меню сайта.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import tools
from config import settings

CORPUS = Path(__file__).resolve().parent / "corpus"

# `read_url` режет страницу по `max_url_content_length`, и для АГЕНТА это верно:
# длинная страница не должна съедать контекст. Для сборки корпуса — наоборот:
# нужен полный текст, потому что резать его будет индексация, на чанки с
# перекрытием. Первый прогон этого не учёл, и все двадцать документов вышли
# ровно по 8190 символов — обрезанными на одном и том же месте.
#
# Поднимаем предел только здесь, в скрипте миграции, не трогая настройку,
# которой пользуются агенты.
settings.max_url_content_length = 400_000

# Ссылки взяты из задания плюс страницы stdlib, которые реально нужны для
# типовых user story: структуры данных, типы, пути, тесты, разбор аргументов.
SOURCES = [
    ("google-python-style-guide",
     "https://google.github.io/styleguide/pyguide.html"),
    ("fastapi-tutorial-first-steps",
     "https://fastapi.tiangolo.com/tutorial/first-steps/"),
    ("fastapi-request-body",
     "https://fastapi.tiangolo.com/tutorial/body/"),
    ("fastapi-handling-errors",
     "https://fastapi.tiangolo.com/tutorial/handling-errors/"),
    ("fastapi-testing",
     "https://fastapi.tiangolo.com/tutorial/testing/"),
    ("python-dataclasses", "https://docs.python.org/3/library/dataclasses.html"),
    ("python-typing", "https://docs.python.org/3/library/typing.html"),
    ("python-pathlib", "https://docs.python.org/3/library/pathlib.html"),
    ("python-unittest", "https://docs.python.org/3/library/unittest.html"),
    ("python-argparse", "https://docs.python.org/3/library/argparse.html"),
    ("python-json", "https://docs.python.org/3/library/json.html"),
    ("python-re", "https://docs.python.org/3/library/re.html"),
    ("python-datetime", "https://docs.python.org/3/library/datetime.html"),
    ("python-collections", "https://docs.python.org/3/library/collections.html"),
    ("python-itertools", "https://docs.python.org/3/library/itertools.html"),
    ("python-logging", "https://docs.python.org/3/library/logging.html"),
    ("python-exceptions", "https://docs.python.org/3/library/exceptions.html"),
    ("python-enum", "https://docs.python.org/3/library/enum.html"),
    ("python-decimal", "https://docs.python.org/3/library/decimal.html"),
    ("python-sqlite3", "https://docs.python.org/3/library/sqlite3.html"),
]

MIN_CHARS = 800          # ниже этого — страница не загрузилась, а не «короткая»


def main() -> int:
    CORPUS.mkdir(exist_ok=True)
    written = skipped = failed = 0

    for name, url in SOURCES:
        target = CORPUS / f"{name}.md"
        if target.exists() and target.stat().st_size > MIN_CHARS:
            print(f"  ✓ {name:34} уже есть")
            skipped += 1
            continue

        text = tools.read_url(url=url)
        if text.startswith("ERROR") or len(text) < MIN_CHARS:
            print(f"  ✗ {name:34} {text[:70]}")
            failed += 1
            continue

        # Заголовок с источником: попадёт в чанк и будет виден в цитате, поэтому
        # агент сможет сказать, ОТКУДА требование, а не просто «так принято».
        body = f"# {name.replace('-', ' ')}\n\nSource: {url}\n\n{text.strip()}\n"
        body = re.sub(r"\n{4,}", "\n\n\n", body)
        target.write_text(body, encoding="utf-8")
        print(f"  ✓ {name:34} {len(body):>7} символов")
        written += 1

    total = sum(1 for _ in CORPUS.glob("*.md"))
    print(f"\nзаписано {written}, пропущено {skipped}, не удалось {failed}; "
          f"в корпусе {total} документов")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
