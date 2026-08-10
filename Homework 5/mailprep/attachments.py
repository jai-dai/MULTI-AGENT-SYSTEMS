"""Связь «файл вложения на диске» -> «письмо, которым он пришёл».

Вложения индексируются обычным `ingest.py` как документы, и это правильно: PDF
есть PDF, кто бы его ни прислал. Но документ, попавший в индекс так, теряет всё,
что о нём знала почта — отправителя, адресатов, дату, тему. А спрашивают о таких
файлах как раз почтовыми категориями: «что мне присылал Хетцнер», «что я
отправлял в АТОН у грудні». Без метаданных фильтр `correspondent` по вложениям
не работает вовсе: он смотрит на `sender_email`, которого у PDF нет.

Ключ связи — имя каталога. `imap_fetch` кладёт вложения письма в
`mail/attachments/<attachment_folder(message_id)>/`, и здесь та же функция
читается в обратную сторону, чтобы формула была одна на оба направления.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .imap_fetch import attachment_folder

# Поля письма, которые получает чанк вложения. Имена намеренно совпадают с теми,
# что кладёт в почтовые чанки mailprep/index.py: retriever фильтрует оба индекса
# одним и тем же кодом, и расхождение имён означало бы фильтр, который молча
# работает на одном индексе и не работает на другом.
FIELDS = ("message_id", "thread_id", "subject", "mail_date",
          "sender_email", "sender_name", "to_emails", "cc_emails")


def metadata_by_folder(db_path: str | Path) -> dict[str, dict]:
    """Имя каталога вложений -> метаданные письма.

    Пустой словарь, если базы нет: почта — отдельный и необязательный источник,
    и её отсутствие не должно мешать индексации обычных документов.
    """
    path = Path(db_path)
    if not path.exists():
        return {}

    # Только чтение: индексация не имеет причин писать в почтовую базу, а
    # read-only соединение делает это не вопросом дисциплины.
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT rfc_message_id, thread_id, subject, date, "
            "       sender_email, sender_name, to_json, cc_json "
            "FROM messages").fetchall()
    finally:
        connection.close()

    return {attachment_folder(row["rfc_message_id"]): {
        "message_id": row["rfc_message_id"],
        "thread_id": row["thread_id"],
        "subject": row["subject"] or "",
        # ИМЕННО mail_date, а не date. Замер на живом индексе: устав 2020 года
        # выводился как «2026-06-16», потому что письмо с ним переслали в этом
        # году. Это две разные даты — «когда документ создан» и «когда его
        # прислали», — и вторая не имеет права занимать место первой.
        "mail_date": (row["date"] or "")[:10],
        "sender_email": row["sender_email"] or "",
        "sender_name": row["sender_name"] or "",
        "to_emails": _emails(row["to_json"]),
        "cc_emails": _emails(row["cc_json"]),
    } for row in rows}


def _emails(raw: str) -> list[str]:
    """to_json/cc_json — это [{"email": ..., "name": ...}], пишет их imap_fetch."""
    return [address["email"] for address in json.loads(raw or "[]")
            if address.get("email")]
