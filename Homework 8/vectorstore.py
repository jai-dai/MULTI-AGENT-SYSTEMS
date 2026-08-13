"""Where vectors live — the one place that knows which engine holds them.

Same split as `embeddings.py`, `ocr.py` and `llm.py`: one interface, swappable
implementations, and nothing above this line knows which one is in use.

    faiss    IndexFlatIP in a single file. Exact brute force, no metadata, the
             whole thing rebuilt on every write.
    qdrant   qdrant-client in LOCAL mode — embedded in this process, no server
             and no Docker. Carries a payload per vector, which is what makes
             filtered search possible later.

# The identity of a chunk is its POSITION

`retriever.semantic_search()` returns integer ids that index into the chunk
list, and RRF and the reranker pass those integers around. So a store must
return positions, not ids of its own. Qdrant point ids are therefore the same
positions — which is why swapping the engine changes nothing downstream.

# Why not "because FAISS keeps vectors in RAM"

That was the reason this migration was first proposed, and the numbers do not
support it: 6409 x 1024 float32 is 25 MB, while the cross-encoder next to it is
1.1 GB. Qdrant also keeps vectors in RAM by default — on-disk storage is an
opt-in flag there, just as memory-mapping is in FAISS. The engine is not what
decides that.

The real reasons are the ones FAISS structurally cannot do:

  * filter by metadata BEFORE the search — "only contracts", "only 2024". As a
    corpus grows this matters more than the index type: RETRIEVAL_TOP_K stays
    10 while the corpus does not, so an unfiltered top-10 covers an ever
    smaller slice of it.
  * add and delete points without rewriting everything. `save_state()` still
    rewrites everything, but `append()` no longer does: почта дописывается
    новыми точками, и «пришло пять писем» стоит пяти эмбеддингов, а не тысячи.
  * sparse vectors alongside dense ones, which is the path to retiring
    rank_bm25 — the component measured to break first at scale.

Quantisation, if RAM ever does become the constraint, comes with the same
engine. It is a later argument, not this one.
"""

from __future__ import annotations

import atexit
import json
import shutil
from pathlib import Path

import numpy as np

from config import settings

COLLECTION = "knowledge"


def index_dir(name: str | None = None) -> Path:
    """Каталог индекса. Без аргумента — тот, что настроен для записи.

    Имя появилось, когда поиск научился идти по нескольким индексам сразу:
    писать в один и тот же процесс может только в один индекс, а читать —
    из всех сразу.
    """
    return Path(__file__).parent / (name or settings.index_dir)


# --------------------------------------------------------------------------- #
# FAISS
# --------------------------------------------------------------------------- #


class FaissStore:
    name = "faiss"
    filename = "index.faiss"
    # A flat index stores vectors and nothing else, so a filter cannot be
    # applied before the search. The retriever falls back to over-fetching and
    # filtering afterwards, which is NOT the same thing: if the wanted document
    # is not already inside the over-fetched window, no amount of filtering
    # brings it back. That gap is the reason this abstraction exists.
    supports_filter = False

    def __init__(self, directory: str | None = None) -> None:
        import faiss

        self._faiss = faiss
        self._index = None
        self._dir = directory

    @property
    def path(self) -> Path:
        return index_dir(self._dir) / self.filename

    def exists(self) -> bool:
        return self.path.exists()

    # FAISS хранит только плотные векторы, поэтому лексический поиск при этом
    # бэкенде остаётся за rank_bm25 в памяти процесса. Аргумент принимается и
    # игнорируется молча — так вызывающей стороне не нужно знать, какой движок
    # под ней, а retriever выбирает стадию по наличию словаря на диске.
    def write(self, vectors: np.ndarray, chunks: list[dict],
              sparse: list[dict[int, float]] | None = None) -> None:
        index = self._faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._faiss.write_index(index, str(self.path))

    def append(self, vectors: np.ndarray, chunks: list[dict], offset: int,
               sparse: list[dict[int, float]] | None = None) -> int:
        """Дописать векторы в конец, не трогая уже лежащие.

        FAISS нумерует векторы порядком добавления, поэтому «дописать» здесь
        буквально: позиция нового вектора и есть его номер, и она совпадёт с
        позицией чанка в chunks.json ровно потому, что дописываются они парой.
        """
        index = self._faiss.read_index(str(self.path))
        if index.ntotal != offset:
            raise RuntimeError(
                f"в индексе {index.ntotal} векторов, а чанков {offset} — "
                "рассинхрон, дозапись отменена (нужна полная пересборка)")
        index.add(vectors)
        self._faiss.write_index(index, str(self.path))
        return index.ntotal

    def open(self) -> int:
        self._index = self._faiss.read_index(str(self.path))
        return self._index.ntotal

    def search(self, vector: np.ndarray, top_k: int,
               where: dict[str, list] | None = None,
               ids: list[int] | None = None) -> list[int]:
        _, ids_out = self._index.search(vector, min(top_k, self._index.ntotal))
        ids = ids_out
        return [int(i) for i in ids[0] if i >= 0]

    def all_vectors(self) -> np.ndarray | None:
        index = self._faiss.read_index(str(self.path))
        if index.ntotal == 0:
            return None
        return index.reconstruct_n(0, index.ntotal)


# --------------------------------------------------------------------------- #
# Qdrant (local mode)
# --------------------------------------------------------------------------- #


class QdrantStore:
    name = "qdrant"
    dirname = "qdrant"
    supports_filter = True

    # Everything a chunk carries EXCEPT its text and id becomes payload. Listing
    # the keys instead would mean editing this file every time a new source
    # brings its own metadata — mail arrived with sender, domain, date and
    # language, and none of them are the store's business to know about.
    # The text is excluded because it already lives in chunks.json and would
    # otherwise double the storage for nothing.
    PAYLOAD_SKIP = ("text", "id")
    # Имя разреженного вектора внутри точки. Плотный остаётся безымянным.
    SPARSE_NAME = "bm25"

    def __init__(self, directory: str | None = None) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self._QdrantClient = QdrantClient
        self._client = None
        self._dir = directory

    @property
    def path(self) -> Path:
        return index_dir(self._dir) / self.dirname

    def exists(self) -> bool:
        return self.path.exists() and any(self.path.iterdir())

    def _connect(self):
        # Local mode locks the directory: ingestion and the agent cannot hold it
        # at the same time. They never do in this project — but a second process
        # will fail loudly rather than corrupt anything.
        if self._client is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = self._QdrantClient(path=str(self.path))
            # Closed here rather than left to __del__: the client's finaliser
            # runs while the interpreter is already tearing down its import
            # machinery, and raises "sys.meta_path is None" on the way out. The
            # database is fine either way, but an error printed at exit is
            # indistinguishable from a real one until you go and read it.
            atexit.register(self.close)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def write(self, vectors: np.ndarray, chunks: list[dict],
              sparse: list[dict[int, float]] | None = None) -> None:
        models = self._models
        # A write replaces the collection outright, matching how save_state()
        # behaves today. Incremental upsert is the next step, not this one.
        if self.path.exists():
            self.close()
            shutil.rmtree(self.path)
        client = self._connect()
        client.create_collection(
            COLLECTION,
            # Our vectors are already L2-normalised, so cosine and inner product
            # rank identically; cosine is stated explicitly so the collection
            # stays correct even if that ever stops being true.
            vectors_config=models.VectorParams(size=int(vectors.shape[1]),
                                               distance=models.Distance.COSINE),
            # Разреженный вектор живёт рядом с плотным в той же точке, поэтому
            # фильтр по payload и ограничение по id одинаково действуют на обе
            # стадии поиска. Ради этого он тут и нужен: в rank_bm25 фильтровать
            # нечем, он ничего не знает ни о payload, ни о Qdrant.
            sparse_vectors_config=(
                {self.SPARSE_NAME: models.SparseVectorParams()}
                if sparse is not None else None),
        )
        batch = 512
        for start in range(0, len(chunks), batch):
            window = range(start, min(start + batch, len(chunks)))
            client.upsert(COLLECTION, points=[
                models.PointStruct(
                    id=i,                       # position == id, see module docstring
                    vector=self._vector_of(vectors[i], sparse, i),
                    payload={k: v for k, v in chunks[i].items()
                             if k not in self.PAYLOAD_SKIP},
                ) for i in window
            ])

    def append(self, vectors: np.ndarray, chunks: list[dict], offset: int,
               sparse: list[dict[int, float]] | None = None) -> int:
        """Дописать точки с номерами `offset…`, не перезаписывая коллекцию.

        Ради этого Qdrant и брали: `write()` сносит каталог целиком, и «пришло
        пять писем» стоило бы пересчёта всех эмбеддингов. Здесь пересчитывается
        ровно новое.

        Номер точки — это позиция чанка в chunks.json (см. шапку модуля), и
        сходимость этих двух нумераций проверяется ЯВНО: если в коллекции
        столько точек, сколько чанков уже записано, дозапись безопасна. Иначе
        мы бы молча положили новый вектор поверх чужого, и поиск начал бы
        возвращать не тот текст — дефект, который не проявляется до первого
        неверного ответа.

        Разреженные векторы: коллекция должна быть создана с ними изначально.
        Дописать sparse в коллекцию без sparse нельзя, и это не наша проверка —
        Qdrant откажет сам. А вот словарь BM25 при дозаписи НЕ пересчитывается:
        idf стареет, новые токены в словарь не попадают. Для почтового индекса
        это неважно (он живёт без sparse), для документов дозапись с sparse
        осознанно не используется — там `ingest.py` собирает словарь заново.
        """
        client = self._connect()
        total = client.count(COLLECTION, exact=True).count
        if total != offset:
            raise RuntimeError(
                f"в коллекции {total} точек, а чанков {offset} — рассинхрон, "
                "дозапись отменена (нужна полная пересборка)")
        models = self._models
        batch = 512
        for start in range(0, len(chunks), batch):
            window = range(start, min(start + batch, len(chunks)))
            client.upsert(COLLECTION, points=[
                models.PointStruct(
                    id=offset + i,
                    vector=self._vector_of(vectors[i], sparse, i),
                    payload={k: v for k, v in chunks[i].items()
                             if k not in self.PAYLOAD_SKIP},
                ) for i in window
            ])
        return offset + len(chunks)

    def _vector_of(self, dense: np.ndarray,
                   sparse: list[dict[int, float]] | None, i: int):
        """Плотный вектор, а рядом разреженный — если он есть.

        Пустая строка — это имя безымянного вектора по умолчанию: коллекция
        создавалась с одним плотным вектором без имени, и оно остаётся таким же
        после добавления именованного разреженного. Так старые индексы читаются
        тем же кодом, что и новые.
        """
        if sparse is None:
            return dense.tolist()
        terms = sparse[i]
        return {
            "": dense.tolist(),
            self.SPARSE_NAME: self._models.SparseVector(
                indices=list(terms), values=list(terms.values())),
        }

    def open(self) -> int:
        client = self._connect()
        return client.count(COLLECTION, exact=True).count

    def search(self, vector: np.ndarray, top_k: int,
               where: dict[str, list] | None = None,
               ids: list[int] | None = None) -> list[int]:
        """`ids` — прямое ограничение по номерам точек; `where` — поле payload -> список ТОЧНЫХ значений, до скоринга.

        Точные значения намеренно: MatchAny не требует текстового индекса и
        одинаково работает в локальном режиме и на сервере. Подстроку («invoice»,
        «2*****t») разворачивает в список значений вызывающая сторона — у неё
        уже есть список чанков, так что хранилищу не нужен текстовый поиск.

        Поле произвольное: у документов это `source`, у почты — `to_emails`
        или `sender_email`. Списочные поля Qdrant сопоставляет поэлементно, так
        что MatchAny по `to_emails` находит письмо, где адресат один из многих.
        """
        client = self._connect()
        must = []
        if where:
            must += [self._models.FieldCondition(
                key=field, match=self._models.MatchAny(any=list(values)))
                for field, values in where.items() if values]
        if ids is not None:
            # Участник переписки может совпасть в sender, to ИЛИ cc — это ИЛИ,
            # а условия в `must` соединяются через И. Разворачивать в вложенный
            # should можно, но список номеров точнее и работает одинаково на
            # обоих бэкендах: подходящие чанки уже вычислены вызывающей стороной.
            must.append(self._models.HasIdCondition(has_id=list(ids)))
        query_filter = self._models.Filter(must=must) if must else None
        hits = client.query_points(COLLECTION, query=vector[0].tolist(),
                                   limit=top_k, query_filter=query_filter,
                                   with_payload=False).points
        return [int(h.id) for h in hits]

    def has_sparse(self) -> bool:
        client = self._connect()
        config = client.get_collection(COLLECTION).config.params
        return bool(getattr(config, "sparse_vectors", None))

    def search_sparse(self, terms: dict[int, float], top_k: int,
                      where: dict[str, list] | None = None,
                      ids: list[int] | None = None) -> list[int]:
        """Лексический поиск по BM25-вектору — те же фильтры, что и у плотного.

        Пустой запрос отдаётся как пустой результат, а не как поиск по всей
        базе: если ни один термин не попал в словарь, честный ответ — «ничего»,
        а не произвольные top_k точек.
        """
        if not terms:
            return []
        client = self._connect()
        must = []
        if where:
            must += [self._models.FieldCondition(
                key=field, match=self._models.MatchAny(any=list(values)))
                for field, values in where.items() if values]
        if ids is not None:
            must.append(self._models.HasIdCondition(has_id=list(ids)))
        query_filter = self._models.Filter(must=must) if must else None
        hits = client.query_points(
            COLLECTION,
            query=self._models.SparseVector(indices=list(terms),
                                            values=list(terms.values())),
            using=self.SPARSE_NAME, limit=top_k,
            query_filter=query_filter, with_payload=False).points
        return [int(h.id) for h in hits]

    def all_vectors(self) -> np.ndarray | None:
        client = self._connect()
        total = client.count(COLLECTION, exact=True).count
        if not total:
            return None
        out: dict[int, list[float]] = {}
        offset = None
        while True:
            points, offset = client.scroll(COLLECTION, limit=1024, offset=offset,
                                           with_vectors=True, with_payload=False)
            for point in points:
                # С разреженным вектором рядом точка отдаёт словарь векторов, а
                # плотный лежит под пустым именем — тем самым, под которым он
                # писался. Без разреженного это по-прежнему просто список.
                vector = point.vector
                out[int(point.id)] = (vector[""] if isinstance(vector, dict)
                                      else vector)
            if offset is None:
                break
        return np.stack([out[i] for i in sorted(out)]).astype("float32")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

_BACKENDS = {"faiss": FaissStore, "qdrant": QdrantStore}
_instances: dict[tuple[str, str | None], object] = {}


def get_store(backend: str | None = None, directory: str | None = None):
    """The configured store, built once per (engine, directory).

    Кэш стал по паре, а не одиночкой, когда поиск пошёл по нескольким индексам.
    Для Qdrant это не оптимизация, а необходимость: локальный режим держит
    блокировку каталога, и второй клиент на тот же каталог упал бы.
    """
    choice = ((backend if backend is not None else settings.vector_backend)
              or "faiss").strip().lower()
    if choice not in _BACKENDS:
        raise RuntimeError(f"unknown VECTOR_BACKEND={choice!r}; "
                           f"expected one of: {', '.join(_BACKENDS)}")
    if backend is not None:                    # explicit: used by migration
        return _BACKENDS[choice](directory)
    key = (choice, directory)
    if key not in _instances:
        _instances[key] = _BACKENDS[choice](directory)
    return _instances[key]


def reset() -> None:
    """Forget the cached stores — for tests and for migration, which needs both."""
    for store in _instances.values():
        if hasattr(store, "close"):
            store.close()
    _instances.clear()


def migrate(source: str, target: str) -> int:
    """Move vectors between engines WITHOUT re-embedding anything.

    Every vector already exists; only its container changes. On this corpus
    that is the difference between a second and four hours.
    """
    src, dst = get_store(source), get_store(target)
    if not src.exists():
        raise RuntimeError(f"no {source} index in {index_dir()}")
    vectors = src.all_vectors()
    if vectors is None:
        raise RuntimeError(f"{source} index is empty")
    chunks = json.loads((index_dir() / "chunks.json").read_text(encoding="utf-8"))
    if len(chunks) != len(vectors):
        raise RuntimeError(f"{len(vectors)} vectors but {len(chunks)} chunks — "
                           "stale index, rebuild instead of migrating")
    dst.write(vectors, chunks)
    return len(vectors)
