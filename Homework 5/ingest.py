"""Knowledge ingestion pipeline — documents → chunks → embeddings → index.

    python ingest.py                 # add anything new, keep what is indexed
    python ingest.py --rebuild       # discard the index and start over
    python ingest.py --dirs a,b      # ingest other directories (also DATA_DIR)

Written without LangChain, like the agent itself: a PDF/TXT/MD reader, a
recursive splitter, batched OpenAI embeddings, and a FAISS index saved next to
the chunk texts — the same texts BM25 uses at query time.

INCREMENTAL BY CONTENT HASH. Every source file is keyed by the sha256 of its
bytes, so re-running after adding one document embeds that document only. That
is the assignment's "reloads without re-embedding", and the reason a large
private corpus can be grown one folder at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# `embeddings` first: with the local backend it initialises torch, which must
# precede faiss on macOS (see the note in retriever.py).
import embeddings as emb  # isort: skip
import faiss
import numpy as np
from pypdf import PdfReader

from config import settings

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | PDF_SUFFIXES

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_document(path: Path) -> list[tuple[str, int]]:
    """Return [(text, page)] — one entry per PDF page, one for a text file."""
    if path.suffix.lower() in PDF_SUFFIXES:
        pages = []
        reader = PdfReader(str(path))
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:              # one broken page is not fatal
                print(f"   ! {path.name} p.{number}: {type(exc).__name__}: {exc}")
                continue
            if text.strip():
                pages.append((text, number))
        return pages
    return [(path.read_text(encoding="utf-8", errors="replace"), 1)]


def discover(dirs: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in dirs:
        root = Path(raw).expanduser()
        if not root.is_absolute():
            root = Path(__file__).parent / root
        if not root.exists():
            print(f"   ! directory not found, skipped: {root}")
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED:
                found.append(path)
    return found


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Recursive character splitter.

    Same idea as RecursiveCharacterTextSplitter: break on the coarsest
    separator that fits — paragraph, line, sentence, word — so a chunk rarely
    ends mid-thought.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    for sep in SEPARATORS:
        if sep == "":                                  # last resort: hard cut
            step = max(1, size - overlap)
            return [text[i:i + size] for i in range(0, len(text), step)]
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if len(part) <= size:
                current = part
            else:                       # too long for this level: go finer
                chunks.extend(split_text(part, size, overlap))
        if current:
            chunks.append(current)

        # Overlap: carry the tail of each chunk into the next, so a fact
        # sitting on a boundary stays retrievable from either side.
        if overlap > 0 and len(chunks) > 1:
            stitched = [chunks[0]]
            for previous, chunk in zip(chunks, chunks[1:]):
                stitched.append((previous[-overlap:] + " " + chunk).strip())
            chunks = stitched
        return [c for c in chunks if c.strip()]
    return [text]


def chunk_document(path: Path, digest: str) -> list[dict]:
    chunks = []
    for text, page in read_document(path):
        pieces = split_text(text, settings.chunk_size, settings.chunk_overlap)
        for position, body in enumerate(pieces):
            chunks.append({
                "id": hashlib.sha256(
                    f"{digest}:{page}:{position}".encode()).hexdigest()[:16],
                "text": body,
                "source": path.name,
                "path": str(path),
                "page": page,
            })
    return chunks


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


def embed(texts: list[str]) -> np.ndarray:
    """Passages, not queries — the distinction matters for prefix-trained models."""
    return emb.embed_texts(texts, is_query=False, progress=True)


# --------------------------------------------------------------------------- #
# index i/o
# --------------------------------------------------------------------------- #


def index_dir() -> Path:
    path = Path(__file__).parent / settings.index_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_chunks() -> list[dict]:
    path = index_dir() / CHUNKS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_manifest() -> tuple[dict, dict | None]:
    """Return (files: path -> digest, embedding signature or None).

    The manifest records WHICH MODEL built the index, not just which files went
    into it. Without that, changing EMBEDDING_MODEL in .env and forgetting to
    rebuild produces an index that still answers — with neighbours computed in
    a different vector space. No exception, no warning, just quietly wrong
    retrieval. With it, the next query fails loudly and says what to do.
    """
    path = index_dir() / MANIFEST_FILE
    if not path.exists():
        return {}, None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "files" not in data:                       # pre-signature manifest
        return data, None
    return data.get("files", {}), data.get("embedding")


def load_vectors() -> np.ndarray | None:
    path = index_dir() / INDEX_FILE
    if not path.exists():
        return None
    index = faiss.read_index(str(path))
    if index.ntotal == 0:
        return None
    return index.reconstruct_n(0, index.ntotal)


def save_state(chunks: list[dict], files: dict, vectors: np.ndarray) -> None:
    directory = index_dir()
    manifest = {
        "embedding": {**emb.signature(), "dim": int(vectors.shape[1])},
        "files": files,
    }
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(directory / INDEX_FILE))
    (directory / CHUNKS_FILE).write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def ingest(dirs: list[str] | None = None, rebuild: bool = False) -> dict:
    # Embedding a corpus with a local model is the memory-hungriest thing this
    # project does; check before spending minutes on it, not after.
    import preflight

    preflight.guard()

    dirs = dirs or settings.ingest_dirs
    print(f"scanning: {', '.join(dirs)}")
    files = discover(dirs)
    if not files:
        print("nothing to ingest.")
        return {"files": 0, "chunks": 0, "embedded": 0}

    if rebuild:
        chunks, manifest, vectors = [], {}, None
    else:
        chunks, vectors = load_chunks(), load_vectors()
        manifest, stored_embedding = load_manifest()
        if vectors is not None and len(vectors) != len(chunks):
            print("   ! index and chunk list disagree — rebuilding from scratch")
            chunks, manifest, vectors = [], {}, None
        elif vectors is not None:
            # Appending vectors from a second model into one index would make
            # the two halves incomparable, and nothing downstream could tell.
            try:
                emb.check_compatible(stored_embedding)
            except RuntimeError as exc:
                print(f"\n{exc}\n")
                raise SystemExit(2)

    # chunk id -> vector, so surviving chunks keep their embedding
    known: dict[str, np.ndarray] = {}
    if vectors is not None:
        known = {c["id"]: v for c, v in zip(chunks, vectors)}

    kept = list(chunks)
    fresh: list[dict] = []
    for path in files:
        digest = file_digest(path)
        if manifest.get(str(path)) == digest:
            print(f" = {path.name} (unchanged)")
            continue
        print(f" + {path.name}")
        kept = [c for c in kept if c["path"] != str(path)]   # replace, never duplicate
        fresh.extend(chunk_document(path, digest))
        manifest[str(path)] = digest

    for known_path in list(manifest):                        # deleted on disk
        if not Path(known_path).exists():
            print(f" - {Path(known_path).name} (removed)")
            kept = [c for c in kept if c["path"] != known_path]
            manifest.pop(known_path)

    if not fresh and vectors is not None and len(kept) == len(chunks):
        print(f"index up to date: {len(chunks)} chunks from {len(manifest)} files")
        return {"files": len(manifest), "chunks": len(chunks), "embedded": 0}

    fresh_vectors = None
    if fresh:
        print(f"embedding {len(fresh)} new chunk(s) with "
              f"{emb.describe(emb.signature())}")
        fresh_vectors = embed([c["text"] for c in fresh])

    # Chunks and vectors are rebuilt in ONE order; a mismatch here would make
    # the index answer with somebody else's text.
    final_chunks = kept + fresh
    stack = [known[c["id"]] for c in kept if c["id"] in known]
    if fresh_vectors is not None:
        stack.extend(fresh_vectors)
    if not stack:
        print("nothing indexed.")
        return {"files": len(manifest), "chunks": 0, "embedded": 0}

    final_vectors = np.stack(stack).astype("float32")
    save_state(final_chunks, manifest, final_vectors)
    print(f"\nindex saved to {index_dir()}")
    print(f"  files: {len(manifest)} | chunks: {len(final_chunks)} | "
          f"newly embedded: {len(fresh)} | dim: {final_vectors.shape[1]}")
    return {"files": len(manifest), "chunks": len(final_chunks),
            "embedded": len(fresh)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge index.")
    parser.add_argument("--dirs", help="comma-separated directories to ingest")
    parser.add_argument("--rebuild", action="store_true",
                        help="discard the existing index and embed everything")
    args = parser.parse_args()
    dirs = [d.strip() for d in args.dirs.split(",")] if args.dirs else None
    try:
        ingest(dirs, rebuild=args.rebuild)
    except KeyboardInterrupt:
        print("\ninterrupted; the previous index is untouched.")
        sys.exit(130)


if __name__ == "__main__":
    main()
