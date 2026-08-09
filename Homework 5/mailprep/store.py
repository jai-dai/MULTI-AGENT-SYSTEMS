"""Локальное хранилище писем — SQLite.

Зачем оно вообще нужно, если письма можно эмбеддить сразу на лету:

`clean.py` будет переписываться. Это не риск, а план — паттерны цитат, подписей
и дисклеймеров подгоняются под конкретную переписку итеративно, и README пакета
прямо говорит «универсальных паттернов не существует». Без локальной копии
каждая такая итерация означала бы повторную выкачку всей почты по IMAP.
С хранилищем сырые письма скачиваются ОДИН раз, а прогон очистки заново —
секунды.

Отсюда же следует главное правило схемы: здесь лежит **сырое** письмо, до
очистки. Всё, что вычисляется (очищенный текст, язык, чанки), выводится из него
и может быть пересчитано.

Ещё две вещи, ради которых нужна именно база, а не папка с файлами:

  * дедупликация. Одно письмо приходит и в INBOX, и в SENT, и в «Все письма» —
    в Gmail это норма. `rfc_message_id` как PRIMARY KEY отсекает повтор на
    уровне схемы, а не проверкой в коде.
  * состояние синхронизации. IMAP инкрементален по UID, но UID действительны
    только в паре с UIDVALIDITY: сервер вправе её сменить, и тогда все прежние
    UID недействительны разом. Это состояние надо где-то держать.
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .models import Address, Attachment, RawMessage

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    rfc_message_id TEXT PRIMARY KEY,     -- дедуп между папками
    thread_id      TEXT NOT NULL,
    in_reply_to    TEXT DEFAULT '',
    refs           TEXT DEFAULT '[]',    -- "references" — зарезервированное слово в SQL
    subject        TEXT DEFAULT '',
    sender_email   TEXT DEFAULT '',
    sender_name    TEXT DEFAULT '',
    to_json        TEXT DEFAULT '[]',
    cc_json        TEXT DEFAULT '[]',
    date           TEXT,                 -- ISO 8601
    body_text      TEXT DEFAULT '',
    body_html      TEXT DEFAULT '',
    attachments    TEXT DEFAULT '[]',
    labels         TEXT DEFAULT '[]',
    folder         TEXT DEFAULT '',
    uid            INTEGER,
    fetched_at     TEXT,
    is_bulk        INTEGER DEFAULT 0,   -- рассылка: в базе лежит, в индекс не идёт
    list_id        TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_date   ON messages(date);
CREATE INDEX IF NOT EXISTS idx_sender ON messages(sender_email);
CREATE INDEX IF NOT EXISTS idx_bulk   ON messages(is_bulk);

-- UID имеют смысл только вместе с UIDVALIDITY: если сервер её сменил,
-- прежние UID недействительны и папку надо перечитать с нуля.
CREATE TABLE IF NOT EXISTS sync (
    folder      TEXT PRIMARY KEY,
    uidvalidity INTEGER,
    last_uid    INTEGER DEFAULT 0,
    updated_at  TEXT
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --------------------------------------------------------------------------- #
# запись
# --------------------------------------------------------------------------- #


def _addr_json(addrs: list[Address]) -> str:
    return json.dumps([{"email": a.email, "name": a.name} for a in addrs],
                      ensure_ascii=False)


def save(conn: sqlite3.Connection, msg: RawMessage, folder: str, uid: int) -> bool:
    """Сохранить письмо. Возвращает False, если оно уже было (дубль из другой папки).

    INSERT OR IGNORE, а не REPLACE: первая встреченная копия выигрывает.
    Перезапись сдвигала бы folder/uid у уже сохранённого письма и ломала бы
    инкрементальность той папки, из которой оно пришло изначально.
    """
    cur = conn.execute(
        """INSERT OR IGNORE INTO messages
           (rfc_message_id, thread_id, in_reply_to, refs, subject,
            sender_email, sender_name, to_json, cc_json, date,
            body_text, body_html, attachments, labels, folder, uid, fetched_at,
            is_bulk, list_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (msg.rfc_message_id, msg.thread_id, msg.in_reply_to,
         json.dumps(msg.references, ensure_ascii=False), msg.subject,
         msg.sender.email if msg.sender else "",
         msg.sender.name if msg.sender else "",
         _addr_json(msg.to), _addr_json(msg.cc),
         msg.date.isoformat() if msg.date else None,
         msg.body_text, msg.body_html,
         json.dumps([{"filename": a.filename, "mime_type": a.mime_type,
                      "size_bytes": a.size_bytes, "saved_path": a.saved_path}
                     for a in msg.attachments], ensure_ascii=False),
         json.dumps(msg.labels, ensure_ascii=False), folder, uid,
         datetime.now().isoformat(timespec="seconds"),
         int(msg.is_bulk), msg.list_id))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# чтение
# --------------------------------------------------------------------------- #


def _addrs(raw: str) -> list[Address]:
    return [Address(email=a.get("email", ""), name=a.get("name", ""))
            for a in json.loads(raw or "[]")]


def _row_to_raw(row: sqlite3.Row) -> RawMessage:
    sender_email = row["sender_email"]
    return RawMessage(
        message_id=row["rfc_message_id"],
        thread_id=row["thread_id"],
        rfc_message_id=row["rfc_message_id"],
        in_reply_to=row["in_reply_to"] or "",
        references=json.loads(row["refs"] or "[]"),
        subject=row["subject"] or "",
        sender=Address(email=sender_email, name=row["sender_name"] or "")
        if sender_email else None,
        to=_addrs(row["to_json"]),
        cc=_addrs(row["cc_json"]),
        date=datetime.fromisoformat(row["date"]) if row["date"] else None,
        body_text=row["body_text"] or "",
        body_html=row["body_html"] or "",
        attachments=[Attachment(**a) for a in json.loads(row["attachments"] or "[]")],
        labels=json.loads(row["labels"] or "[]"),
        is_bulk=bool(row["is_bulk"]), list_id=row["list_id"] or "",
    )


def load_all(conn: sqlite3.Connection, since: str | None = None,
             limit: int | None = None, include_bulk: bool = False,
             exclude_senders: str = "") -> list[RawMessage]:
    """Все письма как RawMessage — вход для mailprep.pipeline.process_all().

    Рассылки по умолчанию НЕ отдаются: они лежат в базе (решение обратимо без
    повторной выкачки), но в индекс идти не должны.
    """
    sql = "SELECT * FROM messages"
    where, params = [], []
    if since:
        where.append("date >= ?")
        params.append(since)
    if not include_bulk:
        where.append("is_bulk = 0")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = [_row_to_raw(r) for r in conn.execute(sql, params)]

    if exclude_senders:
        patterns = [p.strip().lower() for p in exclude_senders.split(",") if p.strip()]
        kept = [m for m in rows if not (m.sender and any(
            fnmatch.fnmatch(m.sender.email.lower(), p) for p in patterns))]
        if len(kept) < len(rows):
            print(f"   ⊘ {len(rows) - len(kept)} писем отсеяно по MAIL_EXCLUDE_SENDERS")
        rows = kept
    return rows


def stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) AS messages,
                  COUNT(DISTINCT thread_id) AS threads,
                  SUM(is_bulk) AS bulk,
                  MIN(date) AS oldest, MAX(date) AS newest
           FROM messages""").fetchone()
    folders = {r["folder"]: r["n"] for r in conn.execute(
        "SELECT folder, COUNT(*) AS n FROM messages GROUP BY folder ORDER BY n DESC")}
    domains = {r["d"]: r["n"] for r in conn.execute(
        """SELECT LOWER(SUBSTR(sender_email, INSTR(sender_email,'@')+1)) AS d,
                  COUNT(*) AS n FROM messages
           WHERE sender_email LIKE '%@%'
           GROUP BY d ORDER BY n DESC LIMIT 15""")}
    return {**dict(row), "folders": folders, "top_sender_domains": domains}


# --------------------------------------------------------------------------- #
# состояние синхронизации
# --------------------------------------------------------------------------- #


def sync_state(conn: sqlite3.Connection, folder: str) -> tuple[int | None, int]:
    row = conn.execute("SELECT uidvalidity, last_uid FROM sync WHERE folder = ?",
                       (folder,)).fetchone()
    return (row["uidvalidity"], row["last_uid"]) if row else (None, 0)


def set_sync_state(conn: sqlite3.Connection, folder: str,
                   uidvalidity: int, last_uid: int) -> None:
    conn.execute(
        """INSERT INTO sync (folder, uidvalidity, last_uid, updated_at)
           VALUES (?,?,?,?)
           ON CONFLICT(folder) DO UPDATE SET
               uidvalidity = excluded.uidvalidity,
               last_uid    = excluded.last_uid,
               updated_at  = excluded.updated_at""",
        (folder, uidvalidity, last_uid,
         datetime.now().isoformat(timespec="seconds")))
    conn.commit()
