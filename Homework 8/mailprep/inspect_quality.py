"""
Инструмент для доработки паттернов очистки.

ЭТО ТО, ЧЕМ ТЫ БУДЕШЬ ПОЛЬЗОВАТЬСЯ БОЛЬШЕ ВСЕГО.

Workflow:
  1. Выгрузи 200-300 писем в JSON (dump из Gmail API)
  2. python -m mailprep.inspect_quality dump.json
  3. Смотри, где очистка сработала плохо
  4. Дополняй паттерны в clean.py
  5. Повторяй

Показывает:
  - письма, где после очистки осталось подозрительно мало (перечистили)
  - письма, где почти ничего не срезалось (недочистили — есть цитаты/подписи)
  - side-by-side diff до/после
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from datetime import datetime

from .models import Address, Attachment, RawMessage
from .pipeline import preprocess


def load_dump(path: str) -> list[RawMessage]:
    """
    Читает JSON-дамп. Ожидаемый формат — список объектов вида:
    {
      "message_id": "...", "thread_id": "...", "rfc_message_id": "...",
      "subject": "...", "from": {"email": "...", "name": "..."},
      "to": [{"email": "...", "name": "..."}], "cc": [],
      "date": "2026-05-20T10:30:00+03:00",
      "body_text": "...", "body_html": "...",
      "attachments": [{"filename": "x.pdf", "mime_type": "application/pdf", "size_bytes": 1234}],
      "labels": ["INBOX"]
    }

    Подгони под то, что реально отдаёт твой ридер.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    def addr(d) -> Address | None:
        if not d:
            return None
        return Address(email=d.get("email", ""), name=d.get("name", ""))

    out = []
    for d in data:
        dt = None
        if d.get("date"):
            try:
                dt = datetime.fromisoformat(d["date"])
            except ValueError:
                pass
        out.append(RawMessage(
            message_id=d.get("message_id", ""),
            thread_id=d.get("thread_id", ""),
            rfc_message_id=d.get("rfc_message_id", ""),
            in_reply_to=d.get("in_reply_to", ""),
            references=d.get("references", []),
            subject=d.get("subject", ""),
            sender=addr(d.get("from")),
            to=[addr(x) for x in d.get("to", []) if x],
            cc=[addr(x) for x in d.get("cc", []) if x],
            date=dt,
            body_text=d.get("body_text", ""),
            body_html=d.get("body_html", ""),
            attachments=[
                Attachment(a.get("filename", ""), a.get("mime_type", ""), a.get("size_bytes", 0))
                for a in d.get("attachments", [])
            ],
            labels=d.get("labels", []),
        ))
    return out


def report(raws: list[RawMessage], show: int = 10, verbose: bool = False) -> None:
    rows = []
    for r in raws:
        src = r.body_text.strip() or r.body_html
        clean = preprocess(r)
        before = len(src)
        after = len(clean.body)
        ratio = after / before if before else 0.0
        rows.append((ratio, before, after, r, clean))

    total = len(rows)
    empty = sum(1 for x in rows if x[4].is_empty)
    sig = sum(1 for x in rows if x[4].signature_removed)
    disc = sum(1 for x in rows if x[4].disclaimer_removed)
    untouched = sum(1 for x in rows if x[0] > 0.95)

    print("=" * 72)
    print(f"ВСЕГО ПИСЕМ: {total}")
    print(f"  подпись срезана:      {sig:>4}  ({sig/total*100:.0f}%)")
    print(f"  дисклеймер срезан:    {disc:>4}  ({disc/total*100:.0f}%)")
    print(f"  пустые после очистки: {empty:>4}  ({empty/total*100:.0f}%)   <- если много, паттерны СЛИШКОМ жадные")
    print(f"  почти не тронуто:     {untouched:>4}  ({untouched/total*100:.0f}%)   <- если много, паттернов НЕ ХВАТАЕТ")
    print("=" * 72)

    print(f"\n### ПЕРЕЧИСТИЛИ (осталось <15%) — топ {show}\n")
    for ratio, before, after, r, c in sorted(rows, key=lambda x: x[0])[:show]:
        print(f"  [{ratio:5.1%}] {before:>6} -> {after:<6} | {r.subject[:55]}")
        if verbose:
            print(f"        ДО:    {(r.body_text or '')[:160]!r}")
            print(f"        ПОСЛЕ: {c.body[:160]!r}\n")

    print(f"\n### НЕДОЧИСТИЛИ (осталось >95%) — топ {show}\n")
    for ratio, before, after, r, c in sorted(rows, key=lambda x: -x[0])[:show]:
        print(f"  [{ratio:5.1%}] {before:>6} -> {after:<6} | {r.subject[:55]}")
        if verbose:
            print(f"        ХВОСТ: {c.body[-200:]!r}\n")


def show_one(raws: list[RawMessage], msg_id: str) -> None:
    """Детальный side-by-side по одному письму."""
    for r in raws:
        if r.message_id == msg_id:
            c = preprocess(r)
            src = r.body_text.strip() or r.body_html
            print("=" * 72, "\nДО:\n", "=" * 72, sep="")
            print(src)
            print("=" * 72, "\nПОСЛЕ:\n", "=" * 72, sep="")
            print(c.body)
            print("=" * 72)
            print(f"lang={c.lang} sig={c.signature_removed} "
                  f"disc={c.disclaimer_removed} quoted_removed={c.quoted_removed_chars}")
            return
    print(f"Письмо {msg_id} не найдено", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Инспектор качества очистки почты")
    # Источник писем теперь SQLite (store.py), а JSON-дамп остался как вход для
    # чужих выгрузок. Инспектор должен смотреть на ТЕ ЖЕ данные, что уйдут в
    # индекс, иначе паттерны подгоняются под одно, а работают на другом.
    ap.add_argument("dump", nargs="?", help="путь к JSON-дампу писем")
    ap.add_argument("--db", nargs="?", const="", metavar="PATH",
                    help="брать письма из локальной базы (по умолчанию MAIL_DB)")
    ap.add_argument("--limit", type=int, help="взять первые N писем")
    ap.add_argument("-n", "--show", type=int, default=10, help="сколько примеров показать")
    ap.add_argument("-v", "--verbose", action="store_true", help="показывать текст до/после")
    ap.add_argument("--id", help="детальный разбор одного письма по message_id")
    args = ap.parse_args()

    if args.db is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import settings

        from . import store

        raws = store.load_all(store.connect(args.db or settings.mail_db),
                              limit=args.limit)
    elif args.dump:
        raws = load_dump(args.dump)
    else:
        ap.error("нужен либо путь к JSON-дампу, либо --db")
    if args.id:
        show_one(raws, args.id)
    else:
        report(raws, args.show, args.verbose)


if __name__ == "__main__":
    main()
