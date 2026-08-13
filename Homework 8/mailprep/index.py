"""Почта → индекс: SQLite → mailprep → эмбеддинги → векторное хранилище.

    python -m mailprep.index --sync          # забрать новое по IMAP и переиндексировать
    python -m mailprep.index                 # переиндексировать то, что уже в базе
    python -m mailprep.index --dry-run       # посчитать чанки, ничего не эмбеддить

Стыковка, а не второй конвейер. Письма получают ТУ ЖЕ схему чанка, что и
документы, поэтому `retriever.py`, RRF, реранкер и инструмент агента работают
без единой правки — меняется только `INDEX_DIR`.

# Почему отдельный индекс, а не общий с документами

Смешивать их в одной коллекции — значит заставить договор конкурировать с
письмом за те же три слота `RERANK_TOP_N`. У них разная плотность: письмо
короткое и конкретное, страница договора длинная и обстоятельная, и косинусная
близость у них живёт в разных диапазонах. Раздельные индексы позволяют
спрашивать «что в переписке» и «что в документах» как разные вопросы — а свести
их в один ответ агент умеет сам, у него для этого два вызова инструмента.

# Как ложится схема

    mailprep.Chunk          наш чанк
    ------------------      -----------------------------------------------
    chunk_id            ->  id
    text                ->  text        (уже с контекстной шапкой)
    payload["subject"]  ->  source      цитата читается как [тема p.N]
    payload["thread_id"]->  path
    idx письма в цепочке->  page        N — номер письма в переписке, не страница

Остальной payload переезжает как есть: он и есть то, ради чего в Qdrant
появился фильтр — домены, дата, язык, метки.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings as emb  # isort: skip  (инициализирует torch раньше faiss)
import numpy as np

import vectorstore
from config import settings

from . import pipeline, store

# Ниже этого объёма тело письма не несёт информации, которую можно найти
# поиском: 18 символов «З повагою, VL» ищутся ровно так же, как пустая строка.
MIN_BODY = 60

CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


def to_index_chunks(threads, chunks) -> list[dict]:
    """mailprep.Chunk -> схема чанка нашего конвейера."""
    # Номер письма внутри своей цепочки: он станет "страницей" в цитате.
    position = {}
    for thread in threads:
        for idx, message in enumerate(thread.messages, start=1):
            position[message.message_id] = idx

    out = []
    dropped = 0
    for chunk in chunks:
        payload = dict(chunk.payload)
        subject = payload.get("subject") or "(без темы)"
        text = chunk.text

        # Письмо-конверт: содержательного текста нет, весь смысл во вложении.
        # Порог не «пусто», а «короче MIN_BODY»: замер по реальному ящику дал
        # 35 таких чанков из 137 (26%), и почти все — пересылка документов, где
        # в теле осталось «З повагою, VL». Подпись здесь стоит СВЕРХУ, а
        # clean.py ищет её в последних 15 строках (осознанно: иначе «Regards» в
        # середине текста срезал бы полписьма). То есть это не промах паттерна,
        # а письмо, у которого текста и не было.
        body = chunk.text.split("---", 1)
        body = body[1].strip() if len(body) > 1 else ""
        if len(body) < MIN_BODY:
            names = payload.get("attachment_names") or []
            recipients = payload.get("to_emails") or []
            if not names and not recipients:
                dropped += 1
                continue
            # Имена файлов — это то немногое содержательное, что в письме есть.
            # Заодно они связывают два индекса: найдя письмо, агент узнаёт имя
            # документа, который лежит в индексе документов.
            # Тела нет, но письмо всё равно отвечает на вопрос «что и кому я
            # отправлял и когда»: тема, адресат и дата уже в шапке, остаётся
            # назвать вложения. Выбрасывать такое — терять связь
            # «документ → адресат → дата», ради которой отправленные и брались.
            if names:
                text = f"{text.rstrip()}\nВкладення: {', '.join(names)}"

        out.append({
            "id": chunk.chunk_id,
            "text": text,
            "source": subject,
            "path": payload.get("thread_id", ""),
            "page": position.get(payload.get("message_id"), 1),
            **{k: v for k, v in payload.items()
               if k not in ("subject", "thread_id")},
        })
    if dropped:
        print(f"   ⊘ {dropped} писем без текста и без вложений — в индекс не идут")
    return out


def _existing(directory) -> tuple[list[dict], dict]:
    """Уже проиндексированные чанки и манифест, или пустота."""
    chunks_path = directory / CHUNKS_FILE
    manifest_path = directory / MANIFEST_FILE
    if not chunks_path.exists() or not manifest_path.exists():
        return [], {}
    return (json.loads(chunks_path.read_text(encoding="utf-8")),
            json.loads(manifest_path.read_text(encoding="utf-8")))


def _split_new(fresh: list[dict], old: list[dict]) -> tuple[list[dict], int]:
    """Чанки, которых ещё нет в индексе, и число исчезнувших.

    Личность чанка письма — `sha1(message_id:part_no)`, она не зависит ни от
    текста, ни от порядка. Поэтому «новое» определяется точно, а не по датам:
    письмо, доехавшее задним числом в старую цепочку, тоже будет найдено.

    Исчезнувшие считаются отдельно и не удаляются. Дозапись умеет добавлять, но
    не переставлять: позиция чанка — это номер точки в хранилище, и удаление из
    середины сдвинуло бы все следующие. Если такое случилось (письмо удалили из
    ящика, поменялись правила очистки), честный ответ — пересобрать целиком, и
    об этом говорится вслух.
    """
    known = {c.get("id") for c in old}
    seen = {c.get("id") for c in fresh}
    return [c for c in fresh if c.get("id") not in known], len(known - seen)


def build(sync_first: bool = False, since: str | None = None,
          dry_run: bool = False, limit: int | None = None,
          only_new: bool = False, index_name: str | None = None) -> dict:
    """`index_name` — куда писать, если это не настроенный `INDEX_DIR`.

    Нужно для докачки при старте агента: там цель записи документов уже
    выставлена в `index_vl`, а почта должна лечь в свой индекс, и переставлять
    ради этого глобальную настройку — верный способ однажды затереть документы.
    """
    if sync_first:
        from .imap_fetch import sync as imap_sync

        if not settings.imap_user or not settings.imap_password:
            raise SystemExit("нужны IMAP_USER и IMAP_PASSWORD в .env "
                             "(IMAP_PASSWORD — App Password, не пароль аккаунта)")
        imap_sync(settings.imap_host, settings.imap_user,
                  settings.imap_password.get_secret_value(),
                  settings.imap_folder_list, settings.mail_db,
                  since or (settings.imap_since or None))

    conn = store.connect(settings.mail_db)
    raws = store.load_all(conn, since=since, limit=limit,
                          exclude_senders=settings.mail_exclude_senders)
    if not raws:
        print("в базе нет писем — сначала `python -m mailprep.imap_fetch`")
        return {"messages": 0, "threads": 0, "chunks": 0}
    print(f"писем в базе: {len(raws)}")

    threads, mail_chunks = pipeline.process_all(raws)
    chunks = to_index_chunks(threads, mail_chunks)
    print(f"цепочек: {len(threads)} | чанков: {len(chunks)}")

    # Пустые письма ("ок", "спасибо") после очистки не дают текста. Они не
    # ошибка, но и вектор от них бессмысленный — считаем и выбрасываем.
    empty = sum(1 for t in threads for m in t.messages if m.is_empty)
    if empty:
        print(f"пустых после очистки: {empty} "
              "(с вложениями — остаются под именами файлов, без — выбрасываются)")

    directory = vectorstore.index_dir(index_name)
    directory.mkdir(parents=True, exist_ok=True)

    # Конвейер прогоняется по ВСЕЙ базе всегда — он дёшев (секунды), а вот
    # эмбеддинг стоит времени. Поэтому «инкрементальность» здесь означает не
    # «обработать меньше писем», а «посчитать меньше векторов».
    old, manifest = _existing(directory) if only_new else ([], {})
    stale = 0
    if only_new and old:
        signature = emb.signature()
        was = (manifest.get("embedding") or {})
        if {k: was.get(k) for k in signature} != dict(signature):
            raise SystemExit(
                f"индекс собран другим эмбеддером ({was.get('model')}), "
                f"сейчас настроен {signature.get('model')} — дозапись невозможна, "
                "нужна полная пересборка без --only-new")
        chunks, stale = _split_new(chunks, old)
        if stale:
            print(f"   ! {stale} чанков пропали из базы, но останутся в индексе — "
                  "дозапись их не убирает, нужна полная пересборка")
        if not chunks:
            print(f"новых чанков нет — индекс уже актуален ({len(old)} чанков)")
            return {"messages": len(raws), "threads": len(threads),
                    "chunks": len(old), "embedded": 0}
        print(f"новых чанков: {len(chunks)} (в индексе уже {len(old)})")

    if dry_run:
        print("\n--dry-run: ничего не эмбеддим. Примеры чанков:")
        for chunk in chunks[:3]:
            head = " ".join(chunk["text"].split())[:150]
            print(f"  [{chunk['source'][:40]} p.{chunk['page']}] {head}…")
        return {"messages": len(raws), "threads": len(threads),
                "chunks": len(chunks), "embedded": 0}

    print(f"эмбеддинг {len(chunks)} чанков — {emb.describe(emb.signature())}")
    vectors = emb.embed_texts([c["text"] for c in chunks],
                              is_query=False, progress=True)
    vectors = np.asarray(vectors, dtype="float32")

    embedded = len(chunks)
    if only_new and old:
        vectorstore.get_store(directory=index_name).append(
            vectors, chunks, offset=len(old))
        chunks = old + chunks
    else:
        vectorstore.get_store(directory=index_name).write(vectors, chunks)
    (directory / CHUNKS_FILE).write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (directory / MANIFEST_FILE).write_text(json.dumps({
        "embedding": {**emb.signature(), "dim": int(vectors.shape[1])},
        "source": "mail",
        "messages": len(raws), "threads": len(threads),
        # `files` пуст намеренно: инкрементальность почты живёт в UID'ах IMAP,
        # а не в хешах файлов. Ключ оставлен, чтобы manifest читался тем же кодом.
        "files": {}, "no_text": [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nиндекс записан в {directory}")
    print(f"  писем: {len(raws)} | цепочек: {len(threads)} | "
          f"чанков: {len(chunks)}" + (f" (+{embedded})" if only_new and old else "")
          + f" | dim: {vectors.shape[1]}")
    return {"messages": len(raws), "threads": len(threads),
            "chunks": len(chunks), "embedded": embedded, "stale": stale}


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация почты для RAG.")
    parser.add_argument("--sync", action="store_true",
                        help="сначала забрать новое по IMAP")
    parser.add_argument("--since", help="только письма от этой даты, YYYY-MM-DD")
    parser.add_argument("--limit", type=int, help="взять первые N писем (для проб)")
    parser.add_argument("--dry-run", action="store_true",
                        help="посчитать чанки и показать примеры, не эмбеддить")
    parser.add_argument("--only-new", action="store_true",
                        help="дописать в существующий индекс только те чанки, "
                             "которых в нём ещё нет, не пересчитывая остальные")
    args = parser.parse_args()
    if settings.index_dir == "index":
        print("! INDEX_DIR=index — это индекс документов. Для почты запускать с "
              "INDEX_DIR=index_mail, иначе документы будут перезаписаны.")
        raise SystemExit(2)
    build(sync_first=args.sync, since=args.since,
          dry_run=args.dry_run, limit=args.limit, only_new=args.only_new)


if __name__ == "__main__":
    main()
