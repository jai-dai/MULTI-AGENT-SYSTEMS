"""
Модели данных для препроцессора почты.

Поток: RawMessage (из Gmail API) -> CleanMessage (после очистки) -> Thread (склейка цепочки)
Thread -> чанки для индексации в Qdrant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Address:
    """Один участник переписки."""
    email: str
    name: str = ""

    @property
    def domain(self) -> str:
        return self.email.split("@")[-1].lower() if "@" in self.email else ""

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>" if self.name else self.email


@dataclass
class Attachment:
    """Метаданные вложения; saved_path заполнен, если файл сохранён на диск."""
    filename: str
    mime_type: str
    size_bytes: int = 0
    saved_path: str = ""


@dataclass
class RawMessage:
    """
    Сырое письмо как пришло из Gmail API.
    Здесь ещё есть цитаты, подписи, дисклеймеры.
    """
    message_id: str            # Gmail message id
    thread_id: str             # Gmail thread id
    rfc_message_id: str        # заголовок Message-ID (для склейки по In-Reply-To)
    in_reply_to: str = ""      # заголовок In-Reply-To
    references: list[str] = field(default_factory=list)

    subject: str = ""
    sender: Optional[Address] = None
    to: list[Address] = field(default_factory=list)
    cc: list[Address] = field(default_factory=list)
    date: Optional[datetime] = None

    body_text: str = ""        # text/plain часть
    body_html: str = ""        # text/html часть (fallback если нет plain)
    attachments: list[Attachment] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # Рассылка, а не переписка. Определяется по ФОРМАЛЬНЫМ заголовкам, а не по
    # тексту: List-Unsubscribe, Precedence: bulk, Auto-Submitted. Такие письма
    # остаются в базе, но в индекс не идут — в выборке за два месяца они дали
    # 39% "почти не тронуто", и в RAG это чистый шум, всплывающий на любой
    # запрос про оплату или пропозицию.
    is_bulk: bool = False
    list_id: str = ""          # чей это список (видно, кого отписать)


@dataclass
class CleanMessage:
    """
    Письмо после препроцессинга: только новый текст, без цитат/подписи/дисклеймера.
    Это единица чанкинга.
    """
    message_id: str
    thread_id: str
    subject: str
    sender: Optional[Address]
    to: list[Address]
    cc: list[Address]
    date: Optional[datetime]

    body: str                          # очищенный текст — только то, что написал автор
    quoted_removed_chars: int = 0      # сколько символов срезано (для отладки качества)
    signature_removed: bool = False
    disclaimer_removed: bool = False

    attachments: list[Attachment] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    lang: str = ""                     # detected: ru / uk / en / ko / mixed

    @property
    def participants(self) -> list[Address]:
        out = [self.sender] if self.sender else []
        return out + self.to + self.cc

    @property
    def domains(self) -> set[str]:
        return {a.domain for a in self.participants if a and a.domain}

    @property
    def is_empty(self) -> bool:
        """Письмо, где после очистки ничего не осталось (напр. только 'ОК' или пересылка)."""
        return len(self.body.strip()) < 3


@dataclass
class Thread:
    """
    Цепочка писем, склеенная в один документ.
    Индексируем поштучно (chunk = CleanMessage), но метаданные берём с уровня Thread.
    """
    thread_id: str
    subject: str                       # нормализованная тема (без Re:/Fwd:)
    messages: list[CleanMessage] = field(default_factory=list)

    @property
    def started_at(self) -> Optional[datetime]:
        dates = [m.date for m in self.messages if m.date]
        return min(dates) if dates else None

    @property
    def last_at(self) -> Optional[datetime]:
        dates = [m.date for m in self.messages if m.date]
        return max(dates) if dates else None

    @property
    def all_domains(self) -> set[str]:
        out: set[str] = set()
        for m in self.messages:
            out |= m.domains
        return out

    @property
    def all_participants(self) -> list[Address]:
        seen: dict[str, Address] = {}
        for m in self.messages:
            for a in m.participants:
                if a and a.email and a.email.lower() not in seen:
                    seen[a.email.lower()] = a
        return list(seen.values())


@dataclass
class Chunk:
    """
    Готовая единица для индексации в Qdrant.
    text -> bge-m3 -> вектор; payload -> фильтры при поиске.
    """
    chunk_id: str
    text: str                          # то, что пойдёт в эмбеддинг
    payload: dict                      # метаданные для фильтрации и вывода
