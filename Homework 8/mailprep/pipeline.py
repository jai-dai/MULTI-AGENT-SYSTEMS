"""
Пайплайн: RawMessage -> CleanMessage -> Thread -> Chunk

Чанкинг: 1 письмо = 1 чанк (плюс контекстная шапка цепочки).
Это осознанное решение — резать письмо по 512 токенов бессмысленно,
письмо и так короткое, а контекст цепочки важнее размера чанка.

Длинные письма (> MAX_CHUNK_CHARS) режутся по абзацам с overlap.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from .clean import clean_body, detect_lang, html_to_text, normalize_subject
from .models import Chunk, CleanMessage, RawMessage, Thread

MAX_CHUNK_CHARS = 4000      # bge-m3 держит 8192 токена, но короче = точнее поиск
CHUNK_OVERLAP_CHARS = 300


# --------------------------------------------------------------------------
# Шаг 1: RawMessage -> CleanMessage
# --------------------------------------------------------------------------

def preprocess(raw: RawMessage) -> CleanMessage:
    """Очищает одно письмо."""
    source = raw.body_text.strip() or html_to_text(raw.body_html)
    result = clean_body(source)

    return CleanMessage(
        message_id=raw.message_id,
        thread_id=raw.thread_id,
        subject=normalize_subject(raw.subject),
        sender=raw.sender,
        to=raw.to,
        cc=raw.cc,
        date=raw.date,
        body=result.body,
        quoted_removed_chars=result.quoted_removed_chars,
        signature_removed=result.signature_removed,
        disclaimer_removed=result.disclaimer_removed,
        attachments=raw.attachments,
        labels=raw.labels,
        lang=detect_lang(result.body),
    )


# --------------------------------------------------------------------------
# Шаг 2: список CleanMessage -> список Thread
# --------------------------------------------------------------------------

def build_threads(messages: Iterable[CleanMessage],
                  drop_empty: bool = True) -> list[Thread]:
    """
    Группирует письма в цепочки по thread_id (Gmail его уже проставил корректно).

    Если понадобится склейка по References/In-Reply-To (напр. при импорте
    из другого источника) — здесь место для union-find по rfc_message_id.
    """
    buckets: dict[str, list[CleanMessage]] = {}
    for m in messages:
        if drop_empty and m.is_empty:
            continue
        buckets.setdefault(m.thread_id, []).append(m)

    threads: list[Thread] = []
    for tid, msgs in buckets.items():
        msgs.sort(key=lambda m: (m.date is None, m.date))
        subject = next((m.subject for m in msgs if m.subject), "")
        threads.append(Thread(thread_id=tid, subject=subject, messages=msgs))

    threads.sort(key=lambda t: (t.last_at is None, t.last_at), reverse=True)
    return threads


# --------------------------------------------------------------------------
# Шаг 3: Thread -> Chunk[]
# --------------------------------------------------------------------------

def _split_long(text: str) -> list[str]:
    """Режет длинный текст по абзацам с перекрытием."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    parts, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) + 2 <= MAX_CHUNK_CHARS:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                parts.append(buf)
            tail = buf[-CHUNK_OVERLAP_CHARS:] if buf else ""
            buf = f"{tail}\n\n{para}" if tail else para
    if buf:
        parts.append(buf)
    return parts


def _context_header(thread: Thread, msg: CleanMessage, idx: int) -> str:
    """
    Шапка, которая добавляется к тексту ПЕРЕД эмбеддингом.

    Зачем: изолированное письмо "да, согласны, 14 недель" бесполезно в поиске.
    С шапкой "Тема: Membrane panels RFQ | От: O** P***r" оно находится.
    Это ключевой приём для качества RAG по почте.
    """
    who = str(msg.sender) if msg.sender else "unknown"
    when = msg.date.strftime("%Y-%m-%d") if msg.date else "?"
    # Получатели — в текст, а не только в метаданные. Замер по реальному ящику:
    # 32% писем отправлены самим владельцем, и для них «От: VL» не несёт
    # информации вообще — так подписаны все. Информативно ровно «Кому», а без
    # него запрос «что я писал в 2*****t» не с чем сопоставить: адресата нет ни
    # в одном эмбеддинге. Имена, а не только адреса: спрашивают «что писали
    # П*****й», а не «office@p******a.shoes».
    to_line = ", ".join(str(a) for a in msg.to[:6]) or "—"
    if len(msg.to) > 6:
        to_line += f" и ещё {len(msg.to) - 6}"
    cc_line = ", ".join(str(a) for a in msg.cc[:4])
    header = (
        f"Тема: {thread.subject or '(без темы)'}\n"
        f"От: {who} | Кому: {to_line}"
    )
    if cc_line:
        header += f" | Копия: {cc_line}"
    return header + (
        f"\nДата: {when} | Письмо {idx + 1} из {len(thread.messages)}\n"
        f"---\n"
    )


def to_chunks(thread: Thread) -> list[Chunk]:
    """Превращает цепочку в чанки, готовые к эмбеддингу."""
    chunks: list[Chunk] = []

    for idx, msg in enumerate(thread.messages):
        header = _context_header(thread, msg, idx)
        for part_no, part in enumerate(_split_long(msg.body)):
            raw_id = f"{msg.message_id}:{part_no}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=header + part,
                payload={
                    # идентификаторы
                    "message_id": msg.message_id,
                    "thread_id": thread.thread_id,
                    "part_no": part_no,
                    # для показа в выдаче
                    "subject": thread.subject,
                    "sender": str(msg.sender) if msg.sender else "",
                    "sender_email": msg.sender.email if msg.sender else "",
                    # Отдельно от `domains`, где отправитель и получатели
                    # свалены вместе: по общему списку нельзя отличить письмо
                    # ОТ контрагента от письма К нему.
                    "to_emails": [a.email for a in msg.to],
                    "cc_emails": [a.email for a in msg.cc],
                    "to_domains": sorted({a.domain for a in msg.to if a.domain}),
                    "date": msg.date.isoformat() if msg.date else None,
                    "snippet": part[:200],
                    # для фильтрации в Qdrant
                    "domains": sorted(msg.domains),
                    "thread_domains": sorted(thread.all_domains),
                    "lang": msg.lang,
                    "labels": msg.labels,
                    "has_attachments": bool(msg.attachments),
                    "attachment_names": [a.filename for a in msg.attachments],
                    # отладка качества очистки
                    "_quoted_removed": msg.quoted_removed_chars,
                    "_signature_removed": msg.signature_removed,
                },
            ))
    return chunks


def process_all(raws: Iterable[RawMessage]) -> tuple[list[Thread], list[Chunk]]:
    """Полный проход: сырые письма -> цепочки + чанки."""
    cleaned = [preprocess(r) for r in raws]
    threads = build_threads(cleaned)
    chunks: list[Chunk] = []
    for t in threads:
        chunks.extend(to_chunks(t))
    return threads, chunks
