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


def _openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = (settings.embedding_api_key.get_secret_value()
               if settings.embedding_api_key else
               settings.api_key.get_secret_value())
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
    batch = settings.embed_batch_size
    for start in range(0, len(prepared), batch):
        window = prepared[start:start + batch]
        response = client.embeddings.create(
            model=settings.embedding_model, input=window)
        collected.extend(item.embedding for item in response.data)
        if progress:
            print(f"   embedded {min(start + batch, len(prepared))}/{len(prepared)}")

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
