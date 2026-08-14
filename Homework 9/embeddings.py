"""The single place where text becomes vectors.

Both `ingest.py` (indexing) and `retriever.py` (querying) call this module and
nothing else. That is deliberate: an index built by one model and queried by
another returns confident nonsense — the vectors live in different spaces, and
cosine similarity between them is a meaningless number, not an error. One code
path, plus the signature check below, makes that mismatch impossible to reach
by accident.

Three backends, selected by EMBEDDING_BACKEND:

  openai  — OpenAI's embedding API (default). The corpus text leaves the machine.
  local   — a sentence-transformers model on this machine. Nothing leaves.
  compat  — any OpenAI-compatible /v1/embeddings endpoint (Ollama, vLLM,
            LM Studio, a self-hosted gateway) via EMBEDDING_BASE_URL.

Note that the CHAT model is unrelated to all of this: retrieved passages reach
it as plain text, never as vectors. Embedding backend and chat provider are
chosen independently.
"""

from __future__ import annotations

import numpy as np

from config import settings

# torch (pulled in by sentence-transformers) must be initialised BEFORE faiss —
# see the note in retriever.py. Importing it here keeps that true for ingest.py
# too, which imports this module before faiss.
if settings.embedding_backend == "local":
    import torch  # noqa: F401  # isort: skip

_local_model = None
_client = None


def _batches(texts: list[str]) -> list[list[str]]:
    """Батчи, ограниченные СУММОЙ СИМВОЛОВ, а не количеством элементов.

    Счётчик штук — неверная мера нагрузки. На документах со средним чанком в
    486 символов батч из 64 проходил годами; на почте, где чанк это целое
    письмо (в среднем 1163 символа, максимум 4194), тот же батч из 64 вешал
    сервер: llama.cpp обрабатывает батч как один длинный промпт, и суммарная
    длина упиралась в его пределы. Замер: 8 писем — 200 OK, 32 — таймаут.

    Отказ при этом выглядел не как ошибка размера, а как HTTP 400 ровно через
    пять минут, что читается как «сломался эмбеддер», а не «батч великоват».

    Ограничение по символам подстраивается само: короткие чанки собираются в
    большие батчи, длинные — в маленькие, и ни один источник данных не требует
    отдельной настройки.
    """
    budget = settings.embed_batch_chars
    limit = settings.embed_batch_size
    out: list[list[str]] = []
    window: list[str] = []
    size = 0
    for text in texts:
        length = len(text)
        if window and (size + length > budget or len(window) >= limit):
            out.append(window)
            window, size = [], 0
        window.append(text)
        size += length
    if window:
        out.append(window)
    return out


def _openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        # Falls back to the chat key only because a single OpenAI key serves
        # both endpoints. With a Claude chat model there is no such key, hence
        # the explicit error rather than an AttributeError on None.
        secret = settings.embedding_api_key or settings.api_key
        if secret is None:
            raise RuntimeError(
                f"EMBEDDING_BACKEND={settings.embedding_backend} needs a key: "
                "set EMBEDDING_API_KEY (or API_KEY) in .env"
            )
        key = secret.get_secret_value()
        _client = OpenAI(api_key=key,
                         base_url=settings.embedding_base_url or None,
                         timeout=settings.model_timeout, max_retries=2)
    return _client


def _local():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(settings.embedding_model)
    return _local_model


def _prefixed(texts: list[str], is_query: bool) -> list[str]:
    """Some models are trained WITH instruction prefixes and lose accuracy
    without them (e5: 'query: ' / 'passage: '). Silently omitting them costs
    quality without raising anything, so they are configuration, not folklore."""
    prefix = (settings.embedding_query_prefix if is_query
              else settings.embedding_passage_prefix)
    return [prefix + t for t in texts] if prefix else texts


def embed_texts(texts: list[str], *, is_query: bool = False,
                progress: bool = False) -> np.ndarray:
    """Embed texts and L2-normalise, so a FAISS inner product IS cosine."""
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")

    prepared = _prefixed(texts, is_query)

    if settings.embedding_backend == "local":
        vectors = _local().encode(prepared, batch_size=settings.embed_batch_size,
                                  show_progress_bar=progress,
                                  convert_to_numpy=True,
                                  normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    client = _openai_client()
    collected: list[list[float]] = []
    done = 0
    for window in _batches(prepared):
        response = client.embeddings.create(
            model=settings.embedding_model, input=window)
        collected.extend(item.embedding for item in response.data)
        done += len(window)
        if progress:
            print(f"   embedded {done}/{len(prepared)}")

    array = np.asarray(collected, dtype="float32")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def embedding_dim() -> int:
    """Vector width of the configured model, asked of the model itself."""
    if settings.embedding_backend == "local":
        return int(_local().get_sentence_embedding_dimension())
    probe = _openai_client().embeddings.create(
        model=settings.embedding_model, input=["dimension probe"])
    return len(probe.data[0].embedding)


def signature() -> dict:
    """What an index must have been built with to be queryable by this config.

    Stored in manifest.json at ingest time and compared on every load. The
    dimension alone is not enough: text-embedding-3-small and bge-m3 differ,
    but two unrelated 1024-dim models would pass a width check while returning
    neighbours that mean nothing.

    EMBEDDING_IDENTITY overrides the whole comparison when set. The case it
    exists for: the SAME model reached through different runtimes. Ollama
    serves a quantised bge-m3 as "bge-m3"; sentence-transformers loads the
    fp32 weights as "BAAI/bge-m3". Same architecture, same training, vectors
    that agree to about three decimal places — but nothing in the code can
    prove that, so by default the mismatch is refused. Setting
    EMBEDDING_IDENTITY=bge-m3 in both configurations is you asserting the
    equivalence, and it lets an index built on one runtime be queried from the
    other without re-embedding the corpus.

    Note this is an assertion, not a check: point it at two genuinely different
    models and retrieval degrades silently, which is exactly what the default
    behaviour prevents.
    """
    if settings.embedding_identity:
        return {"identity": settings.embedding_identity}
    return {
        "backend": settings.embedding_backend,
        "model": settings.embedding_model,
        "base_url": settings.embedding_base_url or "",
        "query_prefix": settings.embedding_query_prefix,
        "passage_prefix": settings.embedding_passage_prefix,
    }


def describe(sig: dict) -> str:
    """Человекочитаемо: чем считаем.

    С EMBEDDING_IDENTITY подпись состоит из одного поля, и прежняя версия
    печатала «? (?)» — то есть молчала ровно в том месте, которое должно
    отвечать на вопрос «какой моделью». Имя модели в этом случае берётся из
    текущей конфигурации, а identity показывается как то, чем она себя объявила.
    """
    if "identity" in sig:
        return (f"{settings.embedding_model} "
                f"({settings.embedding_base_url or settings.embedding_backend}, "
                f"identity={sig['identity']})")
    where = sig.get("base_url") or sig.get("backend", "?")
    return f"{sig.get('model', '?')} ({where})"


def check_compatible(stored: dict | None) -> None:
    """Raise unless the index was built with the current embedding config."""
    if not stored:
        return                                    # index predates the check
    current = signature()
    differences = [k for k in current if stored.get(k) != current[k]]
    if not differences:
        return
    raise RuntimeError(
        "embedding mismatch: the index was built with "
        f"{describe(stored)} but the current configuration is "
        f"{describe(current)} (differs in: {', '.join(differences)}). "
        "Vectors from two models are not comparable — searching would return "
        "confident nonsense. Run `python ingest.py --rebuild` to re-embed the "
        "corpus, or restore the previous embedding settings in .env.")
