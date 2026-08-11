"""Замер поиска на размеченном наборе: `python evaluate.py`.

Отвечает на вопрос, который иначе решается на глаз: **какая стадия что даёт**.

Метрики две, и вторая появилась потому, что первая была несправедлива к
реранкеру. Документная спрашивает «нашли ли нужный файл»; но реранкер выбирает
не файл, а нужный кусок внутри него, и по документной метрике он не может
показать ни выигрыша, ни проигрыша. Пассажная спрашивает «несёт ли выданный
пассаж сам ответ» — там его работа наконец видна. Есть она не у всех запросов:
где вопрос звучит как «найди этот документ», любая его страница законна, и
размечать там пассажи было бы подделкой.

Один и тот же набор прогоняется четырьмя конфигурациями —

    semantic          только вектора
    bm25              только лексика
    fused             RRF-слияние, без реранкера
    reranked          полный конвейер

— и разница между `fused` и `reranked` это ровно та цена, которую реранкер
берёт своими 1.1 ГБ. Пока такого замера нет, любое утверждение о качестве
реранкера («он мешает», «нужна модель побольше») является догадкой; с ним —
измеримо.

Метрики: hit@1, hit@3 и MRR. Правильных документов у запроса может быть
несколько (см. eval_queries.json), засчитывается любой из них.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import retriever
from config import settings

QUERIES_FILE = Path(__file__).parent / "eval_queries.json"
TOP_N = 3


def load_queries() -> list[dict]:
    data = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    return data["queries"]


def rank_of_hit(sources: list[str], expect: list[str]) -> int | None:
    """Позиция первого правильного документа (1-based) или None."""
    for position, source in enumerate(sources, start=1):
        if any(needle.lower() in source.lower() for needle in expect):
            return position
    return None


def ranked_ids(query: str, mode: str) -> list[int]:
    """Номера чанков в порядке, который даёт указанная конфигурация."""
    top_k = settings.retrieval_top_k

    semantic = retriever.semantic_search(query, top_k)
    lexical = retriever.bm25_search(query, top_k)

    if mode == "semantic":
        return semantic
    if mode == "bm25":
        return lexical
    fused = retriever.reciprocal_rank_fusion([semantic, lexical])
    if mode != "reranked":
        return fused
    ranked, _ = retriever.rerank(query, fused[:max(top_k, TOP_N)], TOP_N)
    return [i for i, _ in ranked]


def sources_for(order: list[int]) -> list[str]:
    """Имена файлов без повторов.

    Дедупликация по файлу обязательна для документной метрики: три чанка одного
    документа — это один ответ, а не три. Иначе hit@3 мерил бы длину документа,
    а не качество поиска.
    """
    chunks = retriever._load()["chunks"]
    out: list[str] = []
    for i in order:
        name = chunks[i]["source"]
        if name not in out:
            out.append(name)
    return out


def passage_rank(order: list[int], item: dict) -> int | None:
    """Позиция первого пассажа, который СОДЕРЖИТ ответ.

    Пассаж засчитывается, когда он из ожидаемого документа И несёт строку-
    свидетельство. Одного свидетельства мало: «Вхідний залишок» встречается в
    41 чанке разных банковских выписок, и без имени файла метрика засчитала бы
    выписку чужой компании.
    """
    chunks = retriever._load()["chunks"]
    for position, i in enumerate(order[:TOP_N], start=1):
        chunk = chunks[i]
        if not any(n.lower() in chunk["source"].lower() for n in item["expect"]):
            continue
        if any(e.lower() in chunk["text"].lower() for e in item["evidence"]):
            return position
    return None


def report(title: str, modes: list[str], rows: dict[str, list]) -> None:
    print(f"\n{title}")
    print(f"{'конфигурация':<12} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    for mode in modes:
        ranks = rows[mode]
        total = len(ranks)
        hit1 = sum(1 for rank in ranks if rank == 1)
        hit3 = sum(1 for rank in ranks if rank)
        mrr = sum(1 / rank for rank in ranks if rank) / (total or 1)
        print(f"{mode:<12} {hit1:>3}/{total} {hit3:>3}/{total} {mrr:>6.3f}")


def main() -> None:
    queries = load_queries()
    labelled = [q for q in queries if q.get("evidence")]
    modes = ["semantic", "bm25", "fused", "reranked"]
    by_document: dict[str, list] = {mode: [] for mode in modes}
    by_passage: dict[str, list] = {mode: [] for mode in modes}
    misses: list[tuple] = []

    print(f"{len(queries)} запросов, индекс {settings.index_dir}")
    print(f"из них с пассажной разметкой: {len(labelled)}\n")
    print("     док / пассаж                              запрос")
    for item in queries:
        marks = []
        for mode in modes:
            order = ranked_ids(item["query"], mode)
            document = rank_of_hit(sources_for(order)[:TOP_N], item["expect"])
            by_document[mode].append(document)
            passage = None
            if item.get("evidence"):
                passage = passage_rank(order, item)
                by_passage[mode].append(passage)
            if mode == "reranked" and not document:
                misses.append((item, sources_for(order)[:TOP_N]))
            marks.append(f"{mode[:4]}={document or '—'}/{passage or '—'}"
                         if item.get("evidence") else
                         f"{mode[:4]}={document or '—'}")
        print(f"  {' '.join(marks)}  | {item['query'][:44]}")

    report(f"документная метрика ({len(queries)} запросов): нашли ли нужный ФАЙЛ",
           modes, by_document)
    report(f"пассажная метрика ({len(labelled)} запросов): несёт ли пассаж ОТВЕТ",
           modes, by_passage)

    if misses:
        print("\nне найдено полным конвейером:")
        for item, found in misses:
            print(f"  «{item['query']}»")
            print(f"     ждали: {item['expect']}")
            print(f"     дали : {found[:TOP_N]}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)
