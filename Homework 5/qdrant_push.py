"""Перенести индексы из встроенного Qdrant на локальный сервер.

    .venv/bin/python qdrant_push.py --url http://localhost:6333 index_vl index_mail

Эмбеддинги НЕ пересчитываются: каждый вектор уже существует, меняется только
то, кто его обслуживает. На нашем корпусе это разница между минутой и четырьмя
часами.

Разреженные векторы пересобираются из чанков — это чистая арифметика по тексту,
без обращения к модели, и занимает секунды. Словарь при этом сверяется с тем,
что лежит на диске: если он вдруг разошёлся, значит чанки уже не те, которыми
строился индекс, и молча писать такое в новое хранилище нельзя.

Старые каталоги не трогаются. Пока `QDRANT_URL` не прописан в `.env`, всё
продолжает работать по-старому, а сервер стоит рядом с уже перенесёнными
данными — откат это удаление одной строки из конфига.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import sparse
import vectorstore
from config import settings


def read_embedded(name: str) -> tuple[np.ndarray, list[dict], list | None]:
    """Векторы, чанки и разреженные векторы одного индекса — из каталога."""
    directory = vectorstore.index_dir(name)
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))

    settings.qdrant_url = ""                     # читаем именно встроенный
    vectorstore.reset()
    source = vectorstore.get_store("qdrant", directory=name)
    if not source.exists():
        raise SystemExit(f"{name}: встроенной коллекции нет — нечего переносить")
    vectors = source.all_vectors()
    source.close()

    if vectors is None or len(vectors) != len(chunks):
        raise SystemExit(
            f"{name}: {0 if vectors is None else len(vectors)} векторов против "
            f"{len(chunks)} чанков — рассинхрон, переносить нельзя")

    sparse_vectors = None
    vocab_path = directory / sparse.VOCAB_FILE
    if vocab_path.exists():
        vocabulary, sparse_vectors = sparse.build(chunks)
        on_disk = json.loads(vocab_path.read_text(encoding="utf-8"))
        if set(vocabulary) != set(on_disk):
            raise SystemExit(
                f"{name}: словарь BM25, пересобранный из чанков, разошёлся с "
                f"{sparse.VOCAB_FILE} ({len(vocabulary)} против {len(on_disk)} "
                "терминов) — чанки уже не те, которыми строился индекс")
    return vectors, chunks, sparse_vectors


def push(name: str, url: str) -> None:
    vectors, chunks, sparse_vectors = read_embedded(name)
    print(f"{name}: {len(vectors)} векторов, {len(chunks)} чанков, "
          f"BM25 {'есть' if sparse_vectors else 'нет'}")

    settings.qdrant_url = url
    vectorstore.reset()
    target = vectorstore.get_store("qdrant", directory=name)
    target.write(vectors, chunks, sparse_vectors)
    total = target.open()
    target.close()
    print(f"   → коллекция «{name}» на сервере: {total} точек")
    if total != len(chunks):
        raise SystemExit(f"{name}: на сервере {total} точек вместо {len(chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("indexes", nargs="+", help="каталоги индексов")
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()

    for name in args.indexes:
        if not vectorstore.index_dir(name).exists():
            raise SystemExit(f"нет каталога {name}")
        push(name, args.url)

    print(f"\nготово. Чтобы агент пошёл на сервер, добавьте в .env:\n"
          f"  QDRANT_URL={args.url}")


if __name__ == "__main__":
    main()
