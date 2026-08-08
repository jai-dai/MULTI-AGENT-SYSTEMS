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
import re
from pathlib import Path

# IMPORT ORDER IS LOAD-BEARING on macOS. faiss and torch each ship their own
# OpenMP runtime; initialising faiss first makes the interpreter die with
# SIGSEGV at shutdown (measured: exit code 139, output lost with it). Importing
# torch first fixes it. The widely-cited KMP_DUPLICATE_LIB_OK=TRUE does NOT —
# also measured. Torch is imported for this side effect alone; the reranker
# model itself is still loaded lazily, on first use.
import torch  # noqa: F401  # isort: skip  (must precede faiss)
import embeddings as emb  # isort: skip
import faiss
import numpy as np

from config import settings

_TOKEN = re.compile(r"[a-z0-9]+")

# Loaded once per process: the index and the reranker are expensive to build
# and the agent calls knowledge_search many times per session.
_state: dict = {}


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _index_dir() -> Path:
    return Path(__file__).parent / settings.index_dir


def _load() -> dict:
    if _state:
        return _state

    directory = _index_dir()
    index_path, chunks_path = directory / "index.faiss", directory / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"no knowledge index in {directory} — run `python ingest.py` first")

    from rank_bm25 import BM25Okapi

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"index holds {index.ntotal} vectors but {len(chunks)} chunks are "
            "stored — stale index, re-run `python ingest.py --rebuild`")

    # The query is embedded by whatever is configured NOW; the index was built
    # by whatever was configured THEN. If those differ, every distance computed
    # below is meaningless — and nothing else in the pipeline would notice.
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        emb.check_compatible(manifest.get("embedding"))

    _state.update({
        "chunks": chunks,
        "index": index,
        "bm25": BM25Okapi([_tokenize(c["text"]) for c in chunks]),
        "reranker": None,
    })
    return _state


def _reranker():
    state = _load()
    if state["reranker"] is None:
        from sentence_transformers import CrossEncoder
        state["reranker"] = CrossEncoder(settings.reranker_model, max_length=512)
    return state["reranker"]


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
    state = _load()
    if state.get("threshold") is not None:
        return state["threshold"]

    if "rerank_min_score" in settings.model_fields_set:
        state["threshold"] = settings.rerank_min_score
        return state["threshold"]

    try:
        relevant, irrelevant = (float(s) for s in _reranker().predict(_PROBE_PAIRS))
    except Exception:                       # never let calibration break search
        state["threshold"] = settings.rerank_min_score
        return state["threshold"]

    if relevant <= irrelevant:              # model cannot tell them apart
        state["threshold"] = settings.rerank_min_score
    else:
        state["threshold"] = irrelevant + (relevant - irrelevant) * _NOISE_MARGIN
    return state["threshold"]


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def semantic_search(query: str, top_k: int) -> list[int]:
    state = _load()
    # is_query=True: prefix-trained models score their own "query:" side
    # differently from the "passage:" side used at ingest time.
    vector = emb.embed_texts([query], is_query=True)
    _, ids = state["index"].search(vector, min(top_k, state["index"].ntotal))
    return [int(i) for i in ids[0] if i >= 0]


def bm25_search(query: str, top_k: int) -> list[int]:
    state = _load()
    scores = state["bm25"].get_scores(_tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_k]
    return [int(i) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge rankings by sum of 1/(k + rank). k dampens the head of each list so
    one retriever cannot win on its own confidence alone."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def rerank(query: str, candidates: list[int],
           top_n: int) -> tuple[list[tuple[int, float]], bool]:
    """Score (query, passage) pairs with the cross-encoder.

    Returns (ranked, confident); `confident` is False when nothing cleared the
    threshold, and the caller keeps the passages but marks them weak.
    """
    state = _load()
    pairs = [(query, state["chunks"][i]["text"]) for i in candidates]
    scores = [float(s) for s in _reranker().predict(pairs)]
    ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
    strong = [p for p in ranked if p[1] >= rerank_threshold()]
    if strong:
        return strong[:top_n], True
    return ranked[:top_n], False


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def retrieve(query: str, top_k: int = None, top_n: int = None) -> dict:
    """Run the whole pipeline; returns results plus per-stage counts."""
    top_k = top_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    semantic = semantic_search(query, top_k)
    lexical = bm25_search(query, top_k)
    fused = reciprocal_rank_fusion([semantic, lexical])
    if not fused:
        return {"results": [], "confident": True,
                "stages": {"semantic": 0, "bm25": 0, "fused": 0}}

    if settings.rerank_enabled:
        ranked, confident = rerank(query, fused[:max(top_k, top_n)], top_n)
    else:
        # No cross-encoder: keep the fusion order and score each result by its
        # RRF position, so downstream code sees the same shape either way.
        ranked = [(doc_id, round(1.0 / (rank + 1), 4))
                  for rank, doc_id in enumerate(fused[:top_n])]
        confident = True
    chunks = _load()["chunks"]
    results = [{
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "page": chunks[i]["page"],
        "score": round(score, 4),
        "in_semantic": i in semantic,
        "in_bm25": i in lexical,
    } for i, score in ranked]

    return {"results": results, "confident": confident,
            "stages": {"semantic": len(semantic), "bm25": len(lexical),
                       "fused": len(fused)}}


def index_stats() -> dict:
    state = _load()
    return {"chunks": len(state["chunks"]),
            "sources": sorted({c["source"] for c in state["chunks"]})}


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
