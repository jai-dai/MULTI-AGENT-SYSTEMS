"""Hybrid retrieval: semantic + BM25, fused by RRF, reranked by a cross-encoder.

Four stages, each answering a failure of the one before it:

  1. SEMANTIC (FAISS, cosine over text-embedding-3-small) finds paraphrases —
     "how do I split documents" matches a passage about chunking. It misses
     exact tokens: product names, error codes, API symbols.
  2. BM25 (rank_bm25) finds those exact tokens and knows nothing of meaning.
  3. RECIPROCAL RANK FUSION merges the two rankings. RRF fuses RANKS, not
     scores, and that is the point: a cosine of 0.83 and a BM25 of 11.2 are not
     comparable quantities, while "1st" and "3rd" are.
  4. CROSS-ENCODER RERANK reads (query, passage) together instead of comparing
     two independently produced vectors, and judges relevance far better.

The reranker is a filter, NOT the only gate. Measured on this stack:
"what is RAG?" against a passage saying only "retrieval-augmented generation"
scores 0.0000 — the model does not expand acronyms. So when nothing clears the
threshold, the fusion order is kept and the answer is flagged weak instead of
being dropped: a silent empty result is worse than a hedged one.
"""

from __future__ import annotations

import json
from pathlib import Path

# IMPORT ORDER IS LOAD-BEARING on macOS. faiss and torch each ship their own
# OpenMP runtime; initialising faiss first makes the interpreter die with
# SIGSEGV at shutdown (measured: exit code 139, output lost with it). Importing
# torch first fixes it. The widely-cited KMP_DUPLICATE_LIB_OK=TRUE does NOT —
# also measured. Torch is imported for this side effect alone; the reranker
# model itself is still loaded lazily, on first use.
import torch  # noqa: F401  # isort: skip  (must precede faiss)
import embeddings as emb  # isort: skip
import sparse  # isort: skip
import vectorstore  # isort: skip
import numpy as np

from config import settings

# Токенизатор ОДИН на проект и живёт в sparse.py. Разошедшиеся токенизаторы —
# не гипотетическая опасность: ASCII-класс `[a-z0-9]+` уже оставлял от «Статут
# ТОВ АТОН-ГРУП нова редакція 2020» единственный токен `2020` и выключал стадию
# BM25 целиком, молча. История в README.
_tokenize = sparse.tokenize

# Loaded once per process: the index and the reranker are expensive to build
# and the agent calls knowledge_search many times per session.
_states: dict[str, dict] = {}


def index_names() -> list[str]:
    """Индексы, по которым идёт поиск, в порядке из настроек.

    SEARCH_INDEX_DIRS перечисляет их через запятую; пусто — значит один
    INDEX_DIR. Разделение намеренное: писать `ingest.py` может только в ОДИН
    индекс, и слить эти две настройки означало бы сделать запись двусмысленной.
    """
    raw = settings.search_index_dirs or settings.index_dir
    return [name.strip() for name in raw.split(",") if name.strip()]


def _index_dir(name: str | None = None) -> Path:
    return Path(__file__).parent / (name or settings.index_dir)


def _load(name: str | None = None) -> dict:
    name = name or index_names()[0]
    if name in _states:
        return _states[name]

    directory = _index_dir(name)
    chunks_path = directory / "chunks.json"
    store = vectorstore.get_store(directory=name)
    if not store.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"no {store.name} knowledge index in {directory} — run "
            "`python ingest.py` first")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    total = store.open()
    if total != len(chunks):
        raise RuntimeError(
            f"index holds {total} vectors but {len(chunks)} chunks are "
            "stored — stale index, re-run `python ingest.py --rebuild`")

    # The query is embedded by whatever is configured NOW; the index was built
    # by whatever was configured THEN. If those differ, every distance computed
    # below is meaningless — and nothing else in the pipeline would notice.
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        emb.check_compatible(manifest.get("embedding"))

    # Словарь на диске означает, что BM25 уже посчитан и лежит в хранилище
    # разреженными векторами; тогда rank_bm25 не строится вовсе и не занимает
    # замеренные 49 МБ. Выигрыш не в скорости старта — она 0.39s и роли не играет,
    # — а в том, что фильтр по payload у разреженного вектора применяется до
    # скоринга, как у плотного. Подробности в sparse.py.
    vocabulary = sparse.load(directory)
    bm25 = None
    if vocabulary is None:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    _states[name] = {
        "name": name,
        "chunks": chunks,
        "store": store,
        "vocabulary": vocabulary,
        "bm25": bm25,
    }
    return _states[name]


# Реранкер и порог общие на процесс, а не на индекс: модель одна, и грузить её
# по разу на каждый индекс значило бы платить 1.1 ГБ дважды.
_shared: dict = {}


def _reranker():
    if "reranker" not in _shared:
        from sentence_transformers import CrossEncoder
        _shared["reranker"] = CrossEncoder(settings.reranker_model, max_length=512)
    return _shared["reranker"]


# One pair the model must score high, one it must score low. The distance
# between them is the model's usable range on THIS machine, which is the only
# thing a threshold can honestly be expressed in.
_PROBE_PAIRS = [
    ("what is a cat",
     "A cat is a small domesticated carnivorous mammal often kept as a pet."),
    ("what is a cat",
     "Quarterly revenue increased twelve percent year over year."),
]
# Where to sit above the measured noise floor: low enough to keep marginal but
# real passages, high enough that pure noise does not pass.
_NOISE_MARGIN = 0.15


def rerank_threshold() -> float:
    """The relevance floor, measured for the configured reranker.

    A fixed number does not survive a change of model. RERANK_MIN_SCORE=0.02
    was calibrated for bge-reranker-base; ms-marco-MiniLM answers on a
    different scale, and the same 0.02 flagged perfectly good passages as
    "below the threshold" (observed in the example_output_3 run). So the floor
    is derived from what the model itself returns for a known-relevant and a
    known-irrelevant pair, once per process.

    An explicit RERANK_MIN_SCORE in the environment always wins — measurement
    is the default, not a policy.
    """
    if "threshold" in _shared:
        return _shared["threshold"]

    if "rerank_min_score" in settings.model_fields_set:
        _shared["threshold"] = settings.rerank_min_score
        return _shared["threshold"]

    try:
        relevant, irrelevant = (float(s) for s in _reranker().predict(_PROBE_PAIRS))
    except Exception:                       # never let calibration break search
        _shared["threshold"] = settings.rerank_min_score
        return _shared["threshold"]

    if relevant <= irrelevant:              # model cannot tell them apart
        _shared["threshold"] = settings.rerank_min_score
    else:
        _shared["threshold"] = irrelevant + (relevant - irrelevant) * _NOISE_MARGIN
    return _shared["threshold"]


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def resolve_sources(pattern: str, name: str | None = None) -> list[str]:
    """Filename substring -> the exact filenames it matches.

    The substring is resolved HERE, where the chunk list already lives, so the
    store only ever receives exact values. That keeps the Qdrant filter free of
    a payload text index and keeps both backends fed by the same input.
    """
    needle = pattern.strip().lower()
    if not needle:
        return []
    return sorted({c["source"] for c in _load(name)["chunks"]
                   if needle in c["source"].lower()})


def resolve_correspondent(pattern: str, name: str | None = None) -> dict[str, list[str]]:
    """Фрагмент адреса/домена -> точные значения полей участников.

    Метаданные не ищутся сходством векторов: реранкер обучен на «отвечает ли
    пассаж на вопрос», а не «совпал ли адресат». Замер: «що я відправляв у
    24print» находит нужное письмо, но получает score 0.028 и confident=False.
    Поэтому вопрос про участника — это ФИЛЬТР, а не запрос.
    """
    needle = pattern.strip().lower()
    if not needle:
        return {}
    chunks = _load(name)["chunks"]
    out: dict[str, set] = {"sender_email": set(), "to_emails": set(),
                           "cc_emails": set()}
    for chunk in chunks:
        for field in out:
            value = chunk.get(field)
            for address in ([value] if isinstance(value, str) else (value or [])):
                if address and needle in address.lower():
                    out[field].add(address)
    return {field: sorted(values) for field, values in out.items() if values}


def semantic_search(query: str, top_k: int,
                    sources: list[str] | None = None,
                    ids: list[int] | None = None,
                    name: str | None = None) -> list[int]:
    state = _load(name)
    # is_query=True: prefix-trained models score their own "query:" side
    # differently from the "passage:" side used at ingest time.
    vector = emb.embed_texts([query], is_query=True)
    store = state["store"]
    if ids is not None and not store.supports_filter:
        allowed = set(ids)
        wide = store.search(vector, min(top_k * 20, len(state["chunks"])))
        return [i for i in wide if i in allowed][:top_k]
    if ids is not None:
        return store.search(vector, top_k, ids=ids)
    if sources and not store.supports_filter:
        # Post-filtering is a fallback, not an equivalent: the window is
        # widened and then narrowed, so anything that ranked below the widened
        # window is simply gone. Qdrant applies the same filter before scoring.
        allowed = set(sources)
        wide = store.search(vector, min(top_k * 20, len(state["chunks"])))
        return [i for i in wide if state["chunks"][i]["source"] in allowed][:top_k]
    return store.search(vector, top_k, where={"source": sources} if sources else None)


def bm25_search(query: str, top_k: int,
                sources: list[str] | None = None,
                ids: list[int] | None = None,
                name: str | None = None) -> list[int]:
    """Лексическая стадия. Считает её либо хранилище, либо rank_bm25 в памяти.

    Разница не только в том, где живёт индекс. У разреженных векторов в Qdrant
    фильтр по файлу или участнику переписки применяется ДО скоринга, как у
    плотных; rank_bm25 ничего не знает ни о payload, ни о фильтрах, поэтому
    ниже приходится ранжировать весь корпус и отбрасывать лишнее после.
    """
    state = _load(name)
    if state["vocabulary"] is not None:
        return state["store"].search_sparse(
            sparse.query_vector(query, state["vocabulary"]), top_k,
            where={"source": sources} if sources else None, ids=ids)

    scores = state["bm25"].get_scores(_tokenize(query))
    ranked = np.argsort(scores)[::-1]
    if sources:
        # BM25 scores the whole corpus in one pass anyway, so restricting it is
        # free and exact — no widened window, nothing lost.
        allowed = set(sources)
        ranked = [i for i in ranked if state["chunks"][i]["source"] in allowed]
    if ids is not None:
        allowed_ids = set(ids)
        ranked = [i for i in ranked if int(i) in allowed_ids]
    return [int(i) for i in ranked[:top_k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge rankings by sum of 1/(k + rank). k dampens the head of each list so
    one retriever cannot win on its own confidence alone."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def chunk_at(key: tuple[str, int]) -> dict:
    """Кандидат -> сам чанк. Кандидат это пара (индекс, позиция в нём).

    Пара, а не число, потому что поиск идёт по нескольким индексам сразу: номер
    7 существует в каждом из них и означает разное. Всё, что ниже слияния,
    работает с этой парой и ни разу не спрашивает, из какого источника она.
    """
    name, position = key
    return _load(name)["chunks"][position]


def rerank(query: str, candidates: list[tuple[str, int]],
           top_n: int) -> tuple[list[tuple[tuple[str, int], float]], bool]:
    """Score (query, passage) pairs with the cross-encoder.

    Порог ПОМЕЧАЕТ слабые пассажи, но не выбрасывает их. Раньше выбрасывал: при
    наличии хоть одного пассажа выше порога возвращались только такие. Замер на
    размеченном наборе показал, чего это стоит — запрос про ставку
    дисконтирования, xlsx с числами: шумовой чанк из голых чисел получил 0.2375
    и прошёл порог 0.15, а пассаж с самим ответом получил 0.1298, второе место
    по скору, и был отрезан. Ответ терялся не из-за ранжирования (оно было
    верным), а из-за отсечения. После правки hit@3 по пассажам 12/12 вместо
    11/12.

    Это возвращает модуль к его собственному принципу, записанному в шапке:
    молчаливая потеря хуже, чем оговорённый ответ. Пассаж ниже порога уходит
    вызывающей стороне помеченным, и агент знает, что это слабое свидетельство.

    Returns (ranked, confident); `confident` is False когда порог не взял никто.
    """
    pairs = [(query, chunk_at(key)["text"]) for key in candidates]
    scores = [float(s) for s in _reranker().predict(pairs)]
    ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
    confident = any(score >= rerank_threshold() for _, score in ranked)
    return ranked[:top_n], confident


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def positions_for(pattern: str, name: str | None = None) -> list[int]:
    """Номера чанков, где участник переписки совпал с фрагментом адреса."""
    needle = pattern.strip().lower()
    out = []
    for position, chunk in enumerate(_load(name)["chunks"]):
        haystack = [chunk.get("sender_email") or ""]
        haystack += chunk.get("to_emails") or []
        haystack += chunk.get("cc_emails") or []
        if any(needle in str(a).lower() for a in haystack):
            out.append(position)
    return out


def positions_in_period(since: str | None, until: str | None,
                        name: str | None = None) -> list[int]:
    """Номера чанков, чья дата попадает в период.

    Отбор идёт по списку чанков, а не фильтром Qdrant, и это не лень. Даты в
    двух индексах записаны по-разному: у документов `2020-09-22`, у писем полная
    ISO-метка `2026-08-06T11:47:20`. Диапазонный фильтр хранилища требует одного
    типа на поле, а сравнение первых десяти символов работает для обоих: в ISO
    8601 лексикографический порядок совпадает с хронологическим. Дальше список
    уходит в `ids`, который оба бэкенда уже умеют.
    """
    low = (since or "")[:10]
    high = (until or "")[:10]
    out = []
    for position, chunk in enumerate(_load(name)["chunks"]):
        stamp = (chunk.get("date") or "")[:10]
        if not stamp:
            # Без даты чанк не может подтвердить, что попадает в период. Тихо
            # включить его значило бы ответить «за вчера» документом без даты.
            continue
        if low and stamp < low:
            continue
        if high and stamp > high:
            continue
        out.append(position)
    return out


def retrieve(query: str, top_k: int = None, top_n: int = None,
             source: str | None = None, correspondent: str | None = None,
             since: str | None = None, until: str | None = None) -> dict:
    """Run the whole pipeline; returns results plus per-stage counts.

    `source` narrows the search to files whose NAME contains that substring,
    case-insensitively. It is applied before scoring, so it does not merely
    hide results — it gives the wanted documents the whole top-K to themselves.
    That is what stops a growing corpus from diluting every query: TOP_K stays
    10 while the corpus does not.
    """
    top_k = top_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    # Каждый индекс проходит СВОИ стадии, и только потом два ранжирования
    # сливаются ещё одним RRF. Складывать скоры разных индексов было бы
    # натяжкой: у письма и у договора разная природа релевантности, косинус
    # 0.7 в одном не равен 0.7 в другом. Ранги сравнимы, величины — нет.
    semantic: list[tuple[str, int]] = []
    lexical: list[tuple[str, int]] = []
    per_index: list[list[tuple[str, int]]] = []
    matched_files = 0
    empty_filter = True

    for name in index_names():
        ids = positions_for(correspondent, name) if correspondent else None
        if correspondent and not ids:
            continue
        if since or until:
            dated = positions_in_period(since, until, name)
            # Два фильтра — это И, а не ИЛИ: «что присылал Хетцнер в августе»
            # означает пересечение, и объединение вернуло бы весь август плюс
            # всю переписку с Хетцнером, то есть ответ на другой вопрос.
            ids = sorted(set(ids) & set(dated)) if ids is not None else dated
            if not ids:
                continue
        sources = resolve_sources(source, name) if source else None
        if source and not sources:
            continue
        empty_filter = False
        matched_files += len(sources or [])

        near = [(name, i) for i in semantic_search(query, top_k, sources, ids, name)]
        words = [(name, i) for i in bm25_search(query, top_k, sources, ids, name)]
        semantic += near
        lexical += words
        fused_here = reciprocal_rank_fusion([near, words])
        if fused_here:
            per_index.append(fused_here)

    if empty_filter:
        # Silence here would look identical to "the documents say nothing",
        # which is a different and much more misleading answer.
        return {"results": [], "confident": True,
                "filter": source or correspondent, "matched_files": 0,
                "stages": {"semantic": 0, "bm25": 0, "fused": 0}}

    fused = reciprocal_rank_fusion(per_index)
    if not fused:
        return {"results": [], "confident": True,
                "filter": source, "matched_files": matched_files,
                "stages": {"semantic": 0, "bm25": 0, "fused": 0}}

    floor = rerank_threshold() if settings.rerank_enabled else 0.0
    if settings.rerank_enabled:
        # Пул растёт с числом индексов, и это не щедрость. При фиксированных
        # десяти кандидатах второй индекс занимает половину мест независимо от
        # своего размера: почта на 137 чанков вытесняла документы из 7347.
        # Замерено на разметке — фиксированный пул стоил 0.819 -> 0.778 MRR по
        # пассажам; после правки глубина по каждому индексу прежняя.
        pool = max(top_k * len(per_index), top_n)
        ranked, confident = rerank(query, fused[:pool], top_n)
    else:
        # No cross-encoder: keep the fusion order and score each result by its
        # RRF position, so downstream code sees the same shape either way.
        ranked = [(doc_id, round(1.0 / (rank + 1), 4))
                  for rank, doc_id in enumerate(fused[:top_n])]
        confident = True

    semantic_set, lexical_set = set(semantic), set(lexical)
    results = [{
        "text": (chunk := chunk_at(key))["text"],
        "source": chunk["source"],
        "page": chunk["page"],
        "score": round(score, 4),
        # Из какого индекса пришёл пассаж и что он такое. Письмо и документ
        # цитируются по-разному — «[тема, лист 2]» против «[файл.pdf p.3]», —
        # и агент обязан видеть разницу, а не догадываться по виду источника.
        # Признак — `date_source`: его проставляет только ingest.py, читая файл
        # с диска. У тела письма файла нет, значит нет и его.
        "index": key[0],
        "kind": ("mail" if not chunk.get("date_source")
                 else "attachment" if chunk.get("message_id") else "document"),
        # Слабый пассаж теперь доезжает до агента, а не выбрасывается, — значит
        # он обязан приехать с меткой. Иначе разница между «сильное
        # свидетельство» и «лучшее из плохого» исчезает по дороге.
        "weak": bool(settings.rerank_enabled and score < floor),
        "in_semantic": key in semantic_set,
        "in_bm25": key in lexical_set,
        # Without a date the model cannot tell a charter from 2020 apart from
        # this year's, and quotes the stale one with the same confidence.
        "date": (chunk.get("date") or "")[:10],
        # У тела письма своей даты документа нет — там дата отправки, и
        # называть её иначе значило бы соврать в шапке результата.
        "date_source": chunk.get("date_source") or ("mail" if chunk.get("date") else ""),
        # Дата письма показывается отдельно и только когда отличается от даты
        # документа: «договір 2020 року, надіслали в червні 2026» — это две
        # разные величины, и склеивать их в одну значит терять одну из них.
        "mail_date": chunk.get("mail_date", ""),
        # Recognised from an image, so the wording may carry OCR errors —
        # the reader has to know that before quoting it verbatim.
        "ocr": bool(chunk.get("ocr")),
    } for key, score in ranked]

    return {"results": results, "confident": confident,
            "filter": source, "matched_files": len(sources or []),
            "stages": {"semantic": len(semantic), "bm25": len(lexical),
                       "fused": len(fused)}}


def index_stats() -> dict:
    """Сводка по всем индексам, по которым идёт поиск."""
    out = {"indexes": {}, "chunks": 0, "sources": []}
    for name in index_names():
        chunks = _load(name)["chunks"]
        sources = sorted({c["source"] for c in chunks})
        out["indexes"][name] = {"chunks": len(chunks), "sources": len(sources)}
        out["chunks"] += len(chunks)
        out["sources"] += sources
    out["sources"] = sorted(set(out["sources"]))
    return out


if __name__ == "__main__":                          # manual check
    import sys

    q = " ".join(sys.argv[1:]) or "what is retrieval augmented generation"
    out = retrieve(q)
    print(f"query: {q!r} | stages: {out['stages']} | confident: {out['confident']}")
    for r in out["results"]:
        origin = "+".join(name for name, on in
                          (("semantic", r["in_semantic"]), ("bm25", r["in_bm25"])) if on)
        print(f"\n[{r['source']} p.{r['page']}] score={r['score']} via {origin}")
        print("   ", r["text"][:220].replace("\n", " "))
