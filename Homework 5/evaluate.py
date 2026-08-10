"""Замер поиска на размеченном наборе: `python evaluate.py`.

Отвечает на вопрос, который иначе решается на глаз: **какая стадия что даёт**.
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


def sources_for(query: str, mode: str) -> list[str]:
    """Имена файлов в порядке, который даёт указанная конфигурация."""
    chunks = retriever._load()["chunks"]
    top_k = settings.retrieval_top_k

    semantic = retriever.semantic_search(query, top_k)
    lexical = retriever.bm25_search(query, top_k)

    if mode == "semantic":
        order = semantic
    elif mode == "bm25":
        order = lexical
    else:
        order = retriever.reciprocal_rank_fusion([semantic, lexical])
        if mode == "reranked":
            ranked, _ = retriever.rerank(query, order[:max(top_k, TOP_N)], TOP_N)
            order = [i for i, _ in ranked]

    # Дедупликация по файлу: три чанка одного документа — это один ответ, а не
    # три. Иначе hit@3 мерил бы длину документа, а не качество поиска.
    out: list[str] = []
    for i in order:
        name = chunks[i]["source"]
        if name not in out:
            out.append(name)
    return out


def main() -> None:
    queries = load_queries()
    modes = ["semantic", "bm25", "fused", "reranked"]
    results: dict[str, list] = {mode: [] for mode in modes}

    print(f"{len(queries)} запросов, индекс {settings.index_dir}\n")
    for item in queries:
        ranks = {}
        for mode in modes:
            found = sources_for(item["query"], mode)[:TOP_N]
            rank = rank_of_hit(found, item["expect"])
            ranks[mode] = rank
            results[mode].append((item, rank, found))
        flags = " ".join(
            f"{mode[:4]}={ranks[mode] if ranks[mode] else '—'}" for mode in modes)
        mark = "ok " if ranks["reranked"] == 1 else ("~  " if ranks["reranked"] else "MISS")
        print(f"{mark} {flags}  | {item['query'][:52]}")

    print(f"\n{'конфигурация':<12} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    for mode in modes:
        rows = results[mode]
        hit1 = sum(1 for _, rank, _ in rows if rank == 1)
        hit3 = sum(1 for _, rank, _ in rows if rank)
        mrr = sum(1 / rank for _, rank, _ in rows if rank) / len(rows)
        print(f"{mode:<12} {hit1:>3}/{len(rows)} {hit3:>3}/{len(rows)} {mrr:>6.3f}")

    misses = [(item, found) for item, rank, found in results["reranked"] if not rank]
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
