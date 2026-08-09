"""Импорт почты из выгрузки Google Takeout (.mbox) — без паролей и без сети.

    python -m mailprep.mbox_import ~/Downloads/Mail.mbox
    python -m mailprep.mbox_import ~/Downloads/Mail.mbox --labels "EDCF,Woojin"
    python -m mailprep.mbox_import ~/Downloads/Mail.mbox --since 2024-01-01

Альтернатива IMAP, а не замена. Оба пути кладут письма в одну и ту же базу, и
дальше конвейер один: `python -m mailprep.index`.

# Когда это лучше IMAP

Пароль приложения доступен не всегда: его нет при выключенной 2FA, при 2FA
только через ключ безопасности, в аккаунтах Workspace и при включённой
«Дополнительной защите». Takeout работает в любом из этих случаев.

И отдельно — по существу задачи. Нам нужна ПОЛНАЯ локальная копия для RAG, а не
живая синхронизация. Takeout отдаёт всю историю разом, вместе с ярлыками, и не
требует хранить в `.env` действующий доступ к почте. После истории с кодами
восстановления это не мелочь: секрет, которого нет, невозможно утечь.

Цена честная: инкрементальности нет. Обновление — это новая выгрузка. Для
корпуса, который на 95% историчен, размен выгодный.

# Как получить файл

takeout.google.com → снять все галочки → отметить только «Почта» → там же можно
выбрать конкретные ярлыки вместо всего ящика (рекомендуется: и быстрее, и в
индекс не попадёт лишнее) → экспорт. Придёт архив, внутри `Mail/*.mbox`.

Формат mbox читается стандартным модулем `mailbox`, а разбор письма — той же
функцией `parse()`, что и для IMAP: на вход ей идут те же байты RFC822.
"""

from __future__ import annotations

import argparse
import mailbox
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

from . import store
from .imap_fetch import parse

# Takeout кладёт ярлыки Gmail в собственный заголовок.
LABEL_HEADER = "X-Gmail-Labels"
THREAD_HEADER = "X-GM-THRID"


def import_mbox(path: str | Path, db_path: str, labels_wanted: list[str] | None = None,
                since: str | None = None, limit: int | None = None) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"файл не найден: {path}")

    conn = store.connect(db_path)
    cutoff = datetime.strptime(since, "%Y-%m-%d") if since else None
    wanted = {l.strip().lower() for l in labels_wanted or [] if l.strip()}

    box = mailbox.mbox(str(path))
    print(f"{path.name}: {len(box)} писем в выгрузке")

    new = duplicate = skipped = 0
    for position, message in enumerate(box, start=1):
        if limit and new >= limit:
            break
        labels = [l.strip() for l in
                  (message.get(LABEL_HEADER) or "").split(",") if l.strip()]
        if wanted and not ({l.lower() for l in labels} & wanted):
            skipped += 1
            continue

        # `message.as_bytes()` возвращает то же, что отдал бы IMAP FETCH RFC822,
        # поэтому разбор — та же функция, без отдельной ветки кода.
        parsed = parse(message.as_bytes(), thread_id=message.get(THREAD_HEADER, ""),
                       labels=labels)
        if parsed is None:
            skipped += 1
            continue
        if cutoff and parsed.date and parsed.date < cutoff:
            skipped += 1
            continue

        if store.save(conn, parsed, folder=f"takeout:{path.stem}", uid=position):
            new += 1
        else:
            duplicate += 1

        if position % 500 == 0:
            conn.commit()
            print(f"   {position}/{len(box)} — новых {new}, дублей {duplicate}, "
                  f"пропущено {skipped}", flush=True)

    conn.commit()
    print(f"\nимпортировано: новых {new}, дублей {duplicate}, пропущено {skipped}")
    return {**store.stats(conn), "new": new}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Импорт .mbox из Google Takeout в локальную базу.")
    parser.add_argument("mbox", help="путь к файлу .mbox")
    parser.add_argument("--labels", help="взять только эти ярлыки, через запятую")
    parser.add_argument("--since", help="только письма от этой даты, YYYY-MM-DD")
    parser.add_argument("--limit", type=int, help="взять первые N писем (для проб)")
    args = parser.parse_args()

    info = import_mbox(args.mbox, settings.mail_db,
                       labels_wanted=args.labels.split(",") if args.labels else None,
                       since=args.since, limit=args.limit)
    print(f"\nв базе: {info['messages']} писем, {info['threads']} цепочек, "
          f"{info['oldest']} — {info['newest']}")
    print("топ доменов отправителей:")
    for domain, count in list(info["top_sender_domains"].items())[:10]:
        print(f"   {count:>6}  {domain}")


if __name__ == "__main__":
    main()
