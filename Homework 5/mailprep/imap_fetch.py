"""Выкачка почты по IMAP в локальную базу.

    python -m mailprep.imap_fetch                 # инкрементально
    python -m mailprep.imap_fetch --folders INBOX --since 2024-01-01
    python -m mailprep.imap_fetch --stats         # что уже лежит в базе

Только стандартная библиотека: `imaplib` + `email`. Gmail API потребовал бы
проекта в Google Cloud, OAuth-согласия и `google-api-python-client`; IMAP
обходится паролем приложения.

# Пароль приложения, а не пароль от почты

Google отключил вход по основному паролю. Нужен App Password: включить
двухфакторную аутентификацию, затем myaccount.google.com/apppasswords — выдаст
16 символов. Он даёт доступ ТОЛЬКО к почте и отзывается отдельно, не трогая сам
аккаунт. Лежит в .env (в .gitignore), в код не попадает.

# Инкрементальность

IMAP нумерует письма UID'ами, монотонно растущими внутри папки, поэтому «взять
всё новое» — это `UID SEARCH UID <last+1>:*`. Но UID действительны лишь в паре
с UIDVALIDITY: сервер вправе её сменить (переименование папки, восстановление
из бэкапа), и тогда прежние UID недействительны разом. Поэтому UIDVALIDITY
сверяется на каждом заходе, и при расхождении папка перечитывается с нуля —
дубли всё равно отсечёт PRIMARY KEY по Message-ID.

# Что берём сверх стандарта

Gmail отдаёт через IMAP свои расширения `X-GM-THRID` (настоящий id цепочки) и
`X-GM-LABELS` (метки). Это избавляет от догадок при склейке переписки. Если
сервер их не поддерживает — не Gmail, — thread_id выводится из References /
In-Reply-To, а при отсутствии и этого письмо становится собственной цепочкой.
"""

from __future__ import annotations

import argparse
import email
import email.utils
import imaplib
import re
import sys
from pathlib import Path
from datetime import datetime
from email.header import decode_header, make_header
from email.message import Message

from . import store
from .models import Address, Attachment, RawMessage

# Gmail разрешает большие выборки, но одно письмо с вложениями — это мегабайты.
BATCH = 50

# Формальные признаки рассылки. Именно заголовки, а не слова в тексте: их
# ставит отправляющая система, подделывать их незачем, и они не зависят от
# языка письма. "Precedence: bulk|list|junk" — старый почтовый стандарт,
# List-Unsubscribe обязателен для законных рассылок, Auto-Submitted ставят
# автоответчики и роботы.
BULK_HEADERS = ("List-Unsubscribe", "List-Id", "List-Post")
BULK_PRECEDENCE = {"bulk", "list", "junk", "auto_reply"}

# Расширения, которые умеет читать наш конвейер документов (ingest.SUPPORTED).
# Всё остальное сохранять бессмысленно: индексировать мы это всё равно не умеем.
ATTACHMENT_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xlsm", ".pptx",
                       ".txt", ".md", ".rst"}

_GM_THRID = re.compile(rb"X-GM-THRID (\d+)")
_GM_LABELS = re.compile(rb"X-GM-LABELS \(([^)]*)\)")
_UID = re.compile(rb"UID (\d+)")


# --------------------------------------------------------------------------- #
# разбор письма
# --------------------------------------------------------------------------- #


def _text(raw: str | None) -> str:
    """Заголовок в читаемый вид: MIME-кодировки, кириллица, корейский."""
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw))).strip()
    except (UnicodeDecodeError, LookupError, ValueError):
        return raw.strip()


def _addresses(raw: str | None) -> list[Address]:
    if not raw:
        return []
    out = []
    for name, addr in email.utils.getaddresses([raw]):
        if addr:
            out.append(Address(email=addr.lower().strip(), name=_text(name)))
    return out


def _payload(part: Message) -> str:
    """Текст одной части, с учётом кодировки. Битые байты не роняют выгрузку."""
    try:
        data = part.get_payload(decode=True)
    except Exception:
        return ""
    if not data:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return data.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


def _is_bulk(msg: Message) -> tuple[bool, str]:
    precedence = (msg.get("Precedence") or "").strip().lower()
    list_id = (msg.get("List-Id") or "").strip()
    bulk = (any(msg.get(h) for h in BULK_HEADERS)
            or precedence in BULK_PRECEDENCE
            or bool(msg.get("Auto-Submitted")
                    and msg.get("Auto-Submitted", "").lower() != "no"))
    return bulk, list_id


def _safe_name(raw: str) -> str:
    """Имя файла из письма — недоверенный ввод.

    Отправитель волен назвать вложение `../../.ssh/authorized_keys`, и наивная
    склейка путей запишет его туда. Берём только базовое имя и вычищаем всё,
    что может увести из каталога.
    """
    name = Path(raw or "").name.replace("\\", "_")
    name = re.sub(r"[^\w\s.\-()\[\]«»]", "_", name, flags=re.UNICODE).strip(" .")
    return name[:150] or "attachment"


def attachment_folder(message_id: str) -> str:
    """Message-ID -> имя каталога, в котором лежат его вложения.

    Единственное место, где это имя вычисляется. Оно связывает файл на диске с
    письмом в базе, а такая связь ломается тихо: разойдись формула на один
    символ между тем, кто пишет, и тем, кто читает, — вложения просто перестанут
    находить своё письмо, без единой ошибки.
    """
    return re.sub(r"[^\w.\-]", "_", message_id)[:80]


def parse(raw_bytes: bytes, thread_id: str = "", labels: list[str] | None = None,
          attachments_dir: Path | None = None, max_bytes: int = 0
          ) -> RawMessage | None:
    """RFC822 -> RawMessage. None, если письмо не разбирается вовсе.

    Если задан `attachments_dir`, вложения поддерживаемых форматов пишутся на
    диск — оттуда их забирает обычный `ingest.py`, тот же конвейер, что читает
    документы. В деловой почте главное часто именно во вложении: тело письма —
    «у вкладенні», а договор лежит файлом.
    """
    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception:
        return None

    message_id = (msg.get("Message-ID") or "").strip()
    if not message_id:
        # Без Message-ID нечем дедуплицировать. Синтезируем устойчивый ключ из
        # того, что есть, — иначе одно и то же письмо из INBOX и SENT ляжет
        # дважды и удвоит свой вес в поиске.
        stamp = f"{msg.get('Date','')}|{msg.get('From','')}|{msg.get('Subject','')}"
        message_id = f"<synthetic-{abs(hash(stamp)):016x}@local>"

    body_text, body_html, attachments = "", "", []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            disposition = str(part.get("Content-Disposition") or "")
            filename = part.get_filename()
            if "attachment" in disposition.lower() or filename:
                clean_name = _safe_name(_text(filename)) if filename else "attachment"
                data = part.get_payload(decode=True) or b""
                saved = ""
                if (attachments_dir and filename
                        and Path(clean_name).suffix.lower() in ATTACHMENT_SUFFIXES
                        and data and (not max_bytes or len(data) <= max_bytes)):
                    # Каталог на письмо: имена файлов повторяются («Договір.pdf»
                    # приходит десятками), а перезапись потеряла бы документы.
                    folder = attachments_dir / attachment_folder(message_id)
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / clean_name
                    target.write_bytes(data)
                    saved = str(target)
                attachments.append(Attachment(
                    filename=clean_name,
                    mime_type=part.get_content_type(),
                    size_bytes=len(data), saved_path=saved))
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain" and not body_text:
                body_text = _payload(part)
            elif ctype == "text/html" and not body_html:
                body_html = _payload(part)
    else:
        if msg.get_content_type() == "text/html":
            body_html = _payload(msg)
        else:
            body_text = _payload(msg)

    date = None
    if msg.get("Date"):
        try:
            date = email.utils.parsedate_to_datetime(msg["Date"])
            if date and date.tzinfo:
                date = date.astimezone().replace(tzinfo=None)
        except (TypeError, ValueError):
            date = None

    references = (msg.get("References") or "").split()
    in_reply_to = (msg.get("In-Reply-To") or "").strip()
    # Порядок падения: настоящий id цепочки от Gmail -> корень переписки из
    # References -> письмо само себе цепочка.
    thread = thread_id or (references[0] if references else in_reply_to) or message_id

    bulk, list_id = _is_bulk(msg)
    sender = _addresses(msg.get("From"))
    return RawMessage(
        message_id=message_id, thread_id=thread, rfc_message_id=message_id,
        in_reply_to=in_reply_to, references=references,
        subject=_text(msg.get("Subject")),
        sender=sender[0] if sender else None,
        to=_addresses(msg.get("To")), cc=_addresses(msg.get("Cc")),
        date=date, body_text=body_text, body_html=body_html,
        attachments=attachments, labels=labels or [],
        is_bulk=bulk, list_id=list_id,
    )


# --------------------------------------------------------------------------- #
# выкачка
# --------------------------------------------------------------------------- #


def _meta(head: bytes) -> tuple[int, str, list[str]]:
    """UID, X-GM-THRID и X-GM-LABELS из служебной строки ответа FETCH."""
    uid_match = _UID.search(head)
    uid = int(uid_match.group(1)) if uid_match else 0
    thrid_match = _GM_THRID.search(head)
    thread_id = thrid_match.group(1).decode() if thrid_match else ""
    labels: list[str] = []
    labels_match = _GM_LABELS.search(head)
    if labels_match:
        labels = [l.strip('"') for l in
                  labels_match.group(1).decode("utf-8", "replace").split()]
    return uid, thread_id, labels


def decode_mutf7(name: str) -> str:
    """Имя папки IMAP -> читаемый текст.

    IMAP хранит имена в modified UTF-7 (RFC 3501): `&` открывает
    последовательность, `-` закрывает, внутри base64 с `,` вместо `/`.
    Поэтому «Отправленные» на проводе выглядит как
    `&BB4EQgQ,BEAEMAQyBDsENQQ9BD0ESwQ1-`. В стандартной библиотеке кодека для
    этого нет, а без него список папок нечитаем.
    """
    import base64

    out, i = [], 0
    while i < len(name):
        if name[i] != "&":
            out.append(name[i]); i += 1
            continue
        end = name.find("-", i)
        if end == -1:
            out.append(name[i:]); break
        chunk = name[i + 1:end]
        if not chunk:                       # "&-" — это литеральный "&"
            out.append("&")
        else:
            padded = chunk.replace(",", "/")
            padded += "=" * (-len(padded) % 4)
            try:
                out.append(base64.b64decode(padded).decode("utf-16-be"))
            except Exception:
                out.append(name[i:end + 1])
        i = end + 1
    return "".join(out)


_LIST_LINE = re.compile(rb'\((?P<flags>[^)]*)\)\s+"[^"]*"\s+(?P<name>"[^"]*"|\S+)')


def list_folders(imap: imaplib.IMAP4_SSL) -> list[tuple[str, list[str]]]:
    """[(имя, атрибуты)] — как их видит сервер."""
    status, data = imap.list()
    if status != "OK":
        return []
    out = []
    for line in data:
        if not isinstance(line, bytes):
            continue
        m = _LIST_LINE.match(line)
        if not m:
            continue
        name = m.group("name").decode("utf-8", "replace").strip('"')
        flags = m.group("flags").decode("utf-8", "replace").split()
        out.append((name, flags))
    return out


def resolve_folders(imap: imaplib.IMAP4_SSL, wanted: list[str]) -> list[str]:
    """Имена вида `\\Sent` -> реальное имя папки на этом сервере.

    Gmail переводит служебные папки под язык интерфейса: `[Gmail]/Sent Mail`,
    `[Gmail]/Отправленные`, `[Gmail]/Надіслані` — это одна и та же папка. Имя
    зависит от локали аккаунта, а атрибут `\\Sent` в ответе LIST — нет. Поэтому
    в конфиге лучше писать атрибут, а не название.
    """
    folders = list_folders(imap)
    by_flag = {flag.lower(): name for name, flags in folders for flag in flags}
    resolved = []
    for item in wanted:
        if item.startswith("\\"):
            real = by_flag.get(item.lower())
            if real:
                print(f"   {item} -> {decode_mutf7(real)}")
                resolved.append(real)
            else:
                print(f"   ! на сервере нет папки с атрибутом {item}")
        else:
            resolved.append(item)
    return resolved


def fetch_folder(imap: imaplib.IMAP4_SSL, conn, folder: str,
                 since: str | None = None, gmail: bool = True,
                 attachments_dir: Path | None = None, max_bytes: int = 0,
                 backfill: bool = False) -> dict:
    status, data = imap.select(f'"{folder}"', readonly=True)
    if status != "OK":
        print(f"   ! папка недоступна, пропускаю: {folder}")
        return {"folder": folder, "new": 0, "seen": 0}

    validity = int(imap.response("UIDVALIDITY")[1][0])
    known_validity, last_uid = store.sync_state(conn, folder)
    if known_validity is not None and known_validity != validity:
        print(f"   ! UIDVALIDITY изменилась ({known_validity} → {validity}) — "
              f"читаю {folder} заново")
        last_uid = 0

    # Обычный режим — только то, что пришло после прошлой синхронизации.
    #
    # Догрузка НАЗАД так не работает и молча: у старых писем UID МЕНЬШЕ
    # сохранённого, они не попадают ни в `UID last+1:*`, ни под фильтр ниже,
    # и `--since 2024-09-01` вернул бы «новых писем нет» на полном ящике.
    # Поэтому отдельный флаг: он снимает нижнюю границу по UID и оставляет
    # только дату. Повторно пришедшие письма отсеет `store.save` по
    # message-id, а `highest` считается через max и назад не откатится.
    criteria = [] if backfill else [f"UID {last_uid + 1}:*"]
    if since:
        # IMAP понимает только формат 01-Jan-2024.
        criteria.append(f'SINCE {datetime.strptime(since, "%Y-%m-%d").strftime("%d-%b-%Y")}')
    status, data = imap.uid("SEARCH", None, *(criteria or ["ALL"]))
    uids = data[0].split() if status == "OK" and data and data[0] else []
    # `UID n:*` всегда возвращает хотя бы одно письмо, даже когда нового нет.
    if not backfill:
        uids = [u for u in uids if int(u) > last_uid]
    if not uids:
        print(f" = {folder}: новых писем нет")
        return {"folder": folder, "new": 0, "seen": 0}

    items = "(UID X-GM-THRID X-GM-LABELS RFC822)" if gmail else "(UID RFC822)"
    new = seen = 0
    highest = last_uid
    for start in range(0, len(uids), BATCH):
        window = uids[start:start + BATCH]
        status, chunk = imap.uid("FETCH", b",".join(window), items)
        if status != "OK":
            print(f"   ! FETCH не удался на {folder} uid {window[0]}—{window[-1]}")
            continue
        for part in chunk:
            if not isinstance(part, tuple) or len(part) < 2:
                continue
            uid, thread_id, labels = _meta(part[0])
            message = parse(part[1], thread_id, labels,
                            attachments_dir=attachments_dir, max_bytes=max_bytes)
            if message is None:
                continue
            if store.save(conn, message, folder, uid):
                new += 1
            else:
                seen += 1              # уже приходило из другой папки
            highest = max(highest, uid)
        conn.commit()
        print(f"   {folder}: {min(start + BATCH, len(uids))}/{len(uids)} "
              f"(новых {new}, дублей {seen})", flush=True)

    store.set_sync_state(conn, folder, validity, highest)
    return {"folder": folder, "new": new, "seen": seen}


def sync(host: str, user: str, password: str, folders: list[str],
         db_path: str, since: str | None = None,
         attachments_dir: str | None = None, max_mb: int = 25,
         backfill: bool = False) -> dict:
    conn = store.connect(db_path)
    gmail = "gmail" in host.lower() or "google" in host.lower()
    print(f"IMAP {user}@{host} → {db_path}")
    imap = imaplib.IMAP4_SSL(host)
    try:
        # Google shows an App Password as "abcd efgh ijkl mnop". The spaces are
        # presentational, but whether they survive to the server depends on the
        # client, so they are removed here rather than left to chance.
        imap.login(user, password.replace(" ", ""))
        folders = resolve_folders(imap, folders)
        totals = {"new": 0, "seen": 0}
        for folder in folders:
            result = fetch_folder(
                imap, conn, folder, since, gmail,
                attachments_dir=Path(attachments_dir) if attachments_dir else None,
                max_bytes=max_mb * 1024 * 1024, backfill=backfill)
            totals["new"] += result["new"]
            totals["seen"] += result["seen"]
    finally:
        try:
            imap.logout()
        except Exception:
            pass
    print(f"\nготово: новых {totals['new']}, дублей отсеяно {totals['seen']}")
    return {**totals, **store.stats(conn)}


def main() -> None:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from config import settings

    parser = argparse.ArgumentParser(description="Выкачка почты по IMAP в SQLite.")
    parser.add_argument("--folders", help="через запятую; по умолчанию из .env")
    parser.add_argument("--since", help="только письма от этой даты, YYYY-MM-DD")
    parser.add_argument("--backfill", action="store_true",
                        help="догрузить СТАРЫЕ письма: снимает границу по UID, "
                             "берёт всё от --since. Дубли отсеются по message-id")
    parser.add_argument("--stats", action="store_true", help="что уже в базе")
    parser.add_argument("--list-folders", action="store_true",
                        help="показать папки на сервере и их атрибуты")
    args = parser.parse_args()

    if args.list_folders:
        imap = imaplib.IMAP4_SSL(settings.imap_host)
        imap.login(settings.imap_user,
                   settings.imap_password.get_secret_value().replace(" ", ""))
        for name, flags in list_folders(imap):
            marker = " ".join(f for f in flags if f.startswith("\\")
                              and f not in ("\\HasNoChildren", "\\HasChildren"))
            print(f"  {decode_mutf7(name):32} {marker}")
        imap.logout()
        return

    if args.stats:
        info = store.stats(store.connect(settings.mail_db))
        for key, value in info.items():
            print(f"{key}: {value}")
        return

    if not settings.imap_user or not settings.imap_password:
        raise SystemExit(
            "нужны IMAP_USER и IMAP_PASSWORD в .env\n"
            "  IMAP_PASSWORD — это App Password, а не пароль от аккаунта:\n"
            "  включить 2FA, затем https://myaccount.google.com/apppasswords")

    folders = ([f.strip() for f in args.folders.split(",")] if args.folders
               else settings.imap_folder_list)
    info = sync(settings.imap_host, settings.imap_user,
                settings.imap_password.get_secret_value(), folders,
                settings.mail_db, args.since or (settings.imap_since or None),
                attachments_dir=settings.mail_attachments_dir,
                max_mb=settings.mail_attachment_max_mb,
                backfill=args.backfill)
    print(f"\nв базе: {info['messages']} писем, {info['threads']} цепочек, "
          f"{info['oldest']} — {info['newest']}")


if __name__ == "__main__":
    main()
