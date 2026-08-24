"""Залить промпты в Langfuse Prompt Management. Одноразовая миграция.

    .venv/bin/python prompts_seed.py            # создать/обновить
    .venv/bin/python prompts_seed.py --check    # только показать, что там сейчас

# Почему скриптом, а не руками в интерфейсе

Задание предлагает завести промпты через UI. Скрипт лучше по трём причинам, и
ни одна из них не про лень:

- **воспроизводимость.** Развернуть систему на чистом проекте Langfuse — одна
  команда, а не четыре формы с копипастом;
- **источник правды.** Тексты лежат в `prompts_seed.json`, в git, с историей
  изменений. Промпт, живущий только в облаке, невозможно отревьюить в pull
  request;
- **честность миграции.** Видно, ЧТО именно уехало: скрипт печатает имена и
  версии, а не оставляет догадываться.

После заливки система читает промпты ИЗ LANGFUSE и в этот файл не заглядывает.
`prompts_seed.json` — данные миграции, а не конфигурация приложения.

# Про версии и лейблы

`create_prompt` с тем же именем создаёт НОВУЮ ВЕРСИЮ, а не переписывает старую.
Лейбл `production` переезжает на неё. Старая версия остаётся, и откат — это
перевесить лейбл, а не искать текст в истории чата.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import observability
from config import settings

SEED = Path(__file__).resolve().parent / "prompts_seed.json"


def main(check_only: bool = False) -> int:
    client = observability.client()
    if client is None:
        print("Ключей Langfuse нет. Впиши LANGFUSE_PUBLIC_KEY и "
              "LANGFUSE_SECRET_KEY в .env")
        return 2

    data = json.loads(SEED.read_text(encoding="utf-8"))
    label = settings.langfuse_prompt_label

    if check_only:
        print(f"что лежит в Langfuse под лейблом '{label}':")
        for name in data["prompts"]:
            try:
                got = client.get_prompt(name, label=label)
                print(f"  ✅ {name:11} версия {got.version}, "
                      f"{len(got.prompt)} символов")
            except Exception as exc:
                print(f"  ❌ {name:11} {type(exc).__name__}: {str(exc)[:70]}")
        return 0

    print(f"заливаю в {settings.langfuse_base_url}, лейбл '{label}'")
    for name, spec in data["prompts"].items():
        created = client.create_prompt(
            name=name,
            prompt=spec["text"],
            type=spec.get("type", "text"),
            labels=[label],
            tags=["multi-agent", "hw12"],
            commit_message="seeded from prompts_seed.json",
        )
        version = getattr(created, "version", "?")
        print(f"  ✅ {name:11} версия {version}, {len(spec['text'])} символов")

    observability.flush()
    print("\nготово. Prompts → в интерфейсе Langfuse видно по одному на агента.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
