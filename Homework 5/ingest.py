"""Knowledge ingestion pipeline — documents → chunks → embeddings → index.

    python ingest.py                 # add anything new, keep what is indexed
    python ingest.py --rebuild       # discard the index and start over
    python ingest.py --dirs a,b      # ingest other directories (also DATA_DIR)

On a large corpus this runs for hours, nearly all of it in the embedding phase,
which reports `embedded X/Y` per batch. Keep that output:

    python -u ingest.py 2>&1 | tee ingest.log

`-u` unbuffers Python, `tee` writes the file and still shows the run. Do NOT
pipe through `tail`/`head` — they hold everything in memory and print only once
the process ends, so a long run looks silent and an interrupted one leaves
nothing at all. The log matters afterwards too: `save_state` writes the manifest
ONCE, at the very end, so an interrupted run leaves no other trace of which
files gave no text and which failed to read.

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
import os
import sys
from pathlib import Path

# `embeddings` first: with the local backend it initialises torch, which must
# precede faiss on macOS (see the note in retriever.py).
import embeddings as emb  # isort: skip
import faiss
import numpy as np
from pypdf import PdfReader

import ocr
from config import settings

# Pages recovered by OCR, as (path, page). Recognition makes mistakes, so a
# chunk that came from an image is flagged and the agent is told to treat its
# wording with care.
ocr_pages: set[tuple[str, int]] = set()

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
XLSX_SUFFIXES = {".xlsx", ".xlsm"}
PPTX_SUFFIXES = {".pptx"}
SUPPORTED = (TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES
             | XLSX_SUFFIXES | PPTX_SUFFIXES)

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


def read_docx(path: Path) -> str:
    """Paragraphs and table cells of a Word document, in reading order.

    Tables matter more than they look: in the kind of documents people actually
    keep — contracts, invoices, specifications — the numbers live in tables,
    and a reader that takes only paragraphs silently drops exactly the part
    someone will search for. Word has no page concept in the file, so every
    chunk is recorded as page 1.
    """
    from docx import Document

    document = Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_xlsx(path: Path) -> list[tuple[str, int]]:
    """One entry per sheet; rows serialised as "header: value" pairs.

    A spreadsheet row dumped raw — "1200 | 15.03.2024 | Acme Ltd" — embeds
    badly, because embedding models are trained on prose, not on grids. Written
    out as "Amount: 1200; Date: 15.03.2024; Counterparty: Acme Ltd" the same row
    reads like a sentence and lands in a sensible place in vector space. The
    sheet name goes in too, since "Payments" or "2024 Q3" is often the only
    thing that says what the numbers are.

    Exact figures and document numbers are still mostly found by BM25 — that is
    what lexical search is for. This makes them findable semantically as well.
    """
    from openpyxl import load_workbook

    # read_only: never loads the whole grid; data_only: formulas as values.
    book = load_workbook(str(path), read_only=True, data_only=True)
    out: list[tuple[str, int]] = []
    try:
        for index, sheet in enumerate(book.worksheets, start=1):
            # Excel routinely declares a sheet as 1048575 x 16384 while holding
            # a dozen rows, and read_only mode believes the declaration: one
            # real file here took 142 seconds to cross 200k empty rows, and
            # would have needed ~12 minutes per sheet. reset_dimensions()
            # recomputes the range from the cells that exist — same 14 rows of
            # data, 0.35 seconds.
            if hasattr(sheet, "reset_dimensions"):
                sheet.reset_dimensions()
            lines = [f"Sheet: {sheet.title}"]
            headers: list[str] | None = None
            blank_run = 0
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if c is None else str(c).strip() for c in row]
                if not any(cells):
                    # Second guard, for files where even the recomputed range
                    # is wrong: a long silence means the data has ended.
                    blank_run += 1
                    if blank_run > settings.xlsx_blank_run_limit:
                        break
                    continue
                blank_run = 0
                if headers is None:                 # first non-empty row
                    headers = cells
                    lines.append(" | ".join(h for h in cells if h))
                    continue
                pairs = [f"{h}: {v}" for h, v in zip(headers, cells) if v and h]
                lines.append("; ".join(pairs) if pairs
                             else " | ".join(c for c in cells if c))
            if len(lines) > 1:
                out.append(("\n".join(lines), index))
    finally:
        book.close()
    return out


def read_pptx(path: Path) -> list[tuple[str, int]]:
    """One entry per slide: shapes, tables and the speaker notes.

    The notes are the point. A slide often carries three words and a picture,
    while what the presenter actually meant is written underneath — so a reader
    that takes only the visible text indexes the least informative half.
    """
    from pptx import Presentation

    deck = Presentation(str(path))
    out: list[tuple[str, int]] = []
    for index, slide in enumerate(deck.slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Notes: {notes}")
        if parts:
            out.append(("\n".join(parts), index))
    return out


def read_document(path: Path) -> list[tuple[str, int]]:
    """Return [(text, locator)].

    The locator is a page for PDFs, a sheet number for spreadsheets, a slide
    number for decks, and 1 for formats without any structure of their own.
    """
    suffix = path.suffix.lower()

    if suffix in XLSX_SUFFIXES:
        try:
            return read_xlsx(path)
        except Exception as exc:
            print(f"   ! {path.name}: {type(exc).__name__}: {exc}")
            return []

    if suffix in PPTX_SUFFIXES:
        try:
            return read_pptx(path)
        except Exception as exc:
            print(f"   ! {path.name}: {type(exc).__name__}: {exc}")
            return []

    if path.suffix.lower() in DOCX_SUFFIXES:
        try:
            text = read_docx(path)
        except Exception as exc:
            print(f"   ! {path.name}: {type(exc).__name__}: {exc}")
            return []
        return [(text, 1)] if text.strip() else []

    if path.suffix.lower() in PDF_SUFFIXES:
        pages: list[tuple[str, int]] = []
        unreadable: list[int] = []
        try:
            reader = PdfReader(str(path))
        except Exception as exc:          # encrypted, truncated, not really a PDF
            print(f"   ! {path.name}: {type(exc).__name__}: {exc}")
            return []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:              # one broken page is not fatal
                print(f"   ! {path.name} p.{number}: {type(exc).__name__}: {exc}")
                continue
            if len(text.strip()) >= settings.ocr_min_chars:
                pages.append((text, number))
            else:
                unreadable.append(number)

        # Only the pages that gave nothing go to OCR — it costs seconds per
        # page, against milliseconds for a text layer.
        if unreadable and ocr.resolve_backend() != "off":
            print(f"   ↻ {path.name}: {len(unreadable)} page(s) without text → "
                  f"{ocr.describe()}")
            for text, number in ocr.ocr_pdf_pages(path, unreadable):
                pages.append((text, number))
                ocr_pages.add((str(path), number))
        pages.sort(key=lambda item: item[1])
        return pages
    return [(path.read_text(encoding="utf-8", errors="replace"), 1)]


def discover(dirs: list[str]) -> list[Path]:
    """Find ingestible files, pruning directories that are not a corpus.

    Pointing this at a real folder taught the lesson: a documents directory
    that also holds a git clone contributes hundreds of READMEs and package
    docs from `.venv`, and they surface in search as confident noise. Hidden
    directories and the names in EXCLUDE_DIRS are pruned during the walk, so
    the tree is never even descended.
    """
    excluded = {d.strip() for d in settings.exclude_dirs.split(",") if d.strip()}
    found: list[Path] = []
    for raw in dirs:
        root = Path(raw).expanduser()
        if not root.is_absolute():
            root = Path(__file__).parent / root
        if not root.exists():
            print(f"   ! directory not found, skipped: {root}")
            continue
        for current, subdirs, files in os.walk(root):
            subdirs[:] = sorted(d for d in subdirs
                                if d not in excluded and not d.startswith("."))
            for name in sorted(files):
                if Path(name).suffix.lower() in SUPPORTED and not name.startswith("."):
                    found.append(Path(current) / name)
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
            chunk = {
                "id": hashlib.sha256(
                    f"{digest}:{page}:{position}".encode()).hexdigest()[:16],
                "text": body,
                "source": path.name,
                "path": str(path),
                "page": page,
            }
            if (str(path), page) in ocr_pages:
                chunk["ocr"] = True
            chunks.append(chunk)
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


def save_state(chunks: list[dict], files: dict, vectors: np.ndarray,
               barren: list[str] = None) -> None:
    directory = index_dir()
    manifest = {
        "embedding": {**emb.signature(), "dim": int(vectors.shape[1])},
        "files": files,
        # Kept so the list survives the run: these are the documents a later
        # OCR pass has to target.
        "no_text": sorted(barren or []),
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

    preflight.guard(includes_reranker=False)   # ingestion never loads it

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
    barren: list[str] = []          # files that produced no text at all
    for path in files:
        digest = file_digest(path)
        if manifest.get(str(path)) == digest:
            print(f" = {path.name} (unchanged)")
            continue
        produced = chunk_document(path, digest)
        if not produced:
            # A file that yields nothing is the quietest failure in the whole
            # pipeline: it is recorded as processed, contributes no chunks, and
            # the agent later answers "not in the knowledge base" about a
            # document that is sitting right there. Measured on a real corpus:
            # ~15% of PDFs were scans with no text layer at all.
            print(f" ∅ {path.name} — no text extracted")
            barren.append(str(path))
        else:
            print(f" + {path.name}")
        kept = [c for c in kept if c["path"] != str(path)]   # replace, never duplicate
        fresh.extend(produced)
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
    save_state(final_chunks, manifest, final_vectors, barren)
    print(f"\nindex saved to {index_dir()}")
    print(f"  files: {len(manifest)} | chunks: {len(final_chunks)} | "
          f"newly embedded: {len(fresh)} | dim: {final_vectors.shape[1]}")

    if barren:
        print(f"\n⚠️  {len(barren)} file(s) produced NO text and are in the "
              "index in name only:")
        for path_str in barren[:10]:
            print(f"     ∅ {Path(path_str).name}")
        if len(barren) > 10:
            print(f"     … and {len(barren) - 10} more (full list in "
                  f"{MANIFEST_FILE} under \"no_text\")")
        print("   Most often these are scanned PDFs — images with no text "
              "layer. They need OCR; until then the agent will honestly say "
              "it knows nothing about them.")
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
