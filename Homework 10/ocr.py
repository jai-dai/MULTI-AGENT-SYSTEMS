"""Reading scanned PDFs — the ones with no text layer at all.

Measured on a real corpus: about 15% of PDFs are images of paper. `pypdf`
returns an empty string for them, the file is recorded as ingested, and the
agent later says it knows nothing about a document sitting right there. OCR is
the only way to get those pages in.

Two engines behind one interface, chosen by OCR_BACKEND:

  vision     — Apple's Vision framework via pyobjc. Built into macOS: nothing
               to install, no binaries, recognises Ukrainian, Russian and
               English. Unavailable anywhere else.
  tesseract  — the portable option. Needs the `tesseract` binary plus language
               data, so it survives a move to Linux or a machine without
               pyobjc.
  auto       — vision where it exists, else tesseract, else no OCR at all.
  off        — never OCR.

Rasterisation is shared: `pypdfium2` renders pages to images in-process, with
no poppler and no temporary files, so both engines see exactly the same input
and a change of engine changes only the recognition step.

OCR is slow — seconds per page against milliseconds for a text layer — so it
runs ONLY on pages that yielded nothing, never on a whole document that was
readable to begin with.
"""

from __future__ import annotations

import shutil
import sys

from config import settings

BACKEND_VISION = "vision"
BACKEND_TESSERACT = "tesseract"
BACKEND_OFF = "off"

# ISO codes per engine. Vision wants BCP-47, tesseract its own three-letter
# codes; keeping the mapping here means OCR_LANGUAGES stays engine-agnostic.
_VISION_LANGS = {"uk": "uk-UA", "ru": "ru-RU", "en": "en-US",
                 "de": "de-DE", "fr": "fr-FR", "pl": "pl-PL", "ko": "ko-KR"}
_TESSERACT_LANGS = {"uk": "ukr", "ru": "rus", "en": "eng",
                    "de": "deu", "fr": "fra", "pl": "pol", "ko": "kor"}

_state: dict = {}


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #


def vision_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
    except ImportError:
        return False
    return True


def tesseract_available() -> bool:
    if not shutil.which(settings.tesseract_cmd):
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend() -> str:
    """Which engine will actually run, after 'auto' is settled."""
    requested = (settings.ocr_backend or BACKEND_OFF).strip().lower()
    if requested == BACKEND_OFF:
        return BACKEND_OFF
    if requested == "auto":
        if vision_available():
            return BACKEND_VISION
        if tesseract_available():
            return BACKEND_TESSERACT
        return BACKEND_OFF
    if requested == BACKEND_VISION and not vision_available():
        return BACKEND_OFF
    if requested == BACKEND_TESSERACT and not tesseract_available():
        return BACKEND_OFF
    return requested


def describe() -> str:
    backend = resolve_backend()
    if backend == BACKEND_OFF:
        reasons = []
        if not vision_available():
            reasons.append("Vision unavailable (not macOS, or pyobjc missing)")
        if not tesseract_available():
            reasons.append(f"{settings.tesseract_cmd!r} not on PATH")
        return "OCR off" + (f" — {'; '.join(reasons)}" if reasons else "")
    return f"OCR via {backend} ({settings.ocr_languages})"


def _languages() -> list[str]:
    return [c.strip().lower() for c in settings.ocr_languages.split(",") if c.strip()]


# --------------------------------------------------------------------------- #
# rasterisation (shared by both engines)
# --------------------------------------------------------------------------- #


def _render_pages(path, pages: list[int]):
    """Yield (page_number, PIL image) for the requested 1-based page numbers."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        scale = settings.ocr_dpi / 72.0          # PDF user space is 72 dpi
        for number in pages:
            if number < 1 or number > len(document):
                continue
            page = document[number - 1]
            yield number, page.render(scale=scale).to_pil()
    finally:
        document.close()


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #


def _recognise_vision(image) -> str:
    """Apple Vision: VNRecognizeTextRequest over an in-memory image."""
    import Quartz
    import Vision
    from Foundation import NSData

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = NSData.dataWithBytes_length_(buffer.getvalue(), len(buffer.getvalue()))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    if source is None:
        return ""
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        return ""

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    languages = [_VISION_LANGS.get(code, code) for code in _languages()]
    if languages:
        request.setRecognitionLanguages_(languages)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        return ""

    lines = []
    for observation in (request.results() or []):
        candidates = observation.topCandidates_(1)
        if candidates and len(candidates):
            lines.append(str(candidates[0].string()))
    return "\n".join(lines)


def _recognise_tesseract(image) -> str:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    langs = "+".join(_TESSERACT_LANGS.get(code, code) for code in _languages())
    return pytesseract.image_to_string(image, lang=langs or "eng")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def ocr_pdf_pages(path, pages: list[int]) -> list[tuple[str, int]]:
    """OCR the given 1-based pages. Returns [(text, page)] for pages with text.

    Never raises: OCR is an improvement over nothing, and a failure here must
    not cost the pages that read normally.
    """
    backend = resolve_backend()
    if backend == BACKEND_OFF or not pages:
        return []

    recognise = (_recognise_vision if backend == BACKEND_VISION
                 else _recognise_tesseract)
    limited = pages[:settings.ocr_max_pages]
    out: list[tuple[str, int]] = []
    try:
        for number, image in _render_pages(path, limited):
            try:
                text = recognise(image).strip()
            except Exception as exc:
                print(f"   ! OCR failed on p.{number}: {type(exc).__name__}: {exc}")
                continue
            if text:
                out.append((text, number))
    except Exception as exc:
        print(f"   ! OCR could not render {path}: {type(exc).__name__}: {exc}")
        return out

    if len(pages) > settings.ocr_max_pages:
        print(f"   … OCR stopped after {settings.ocr_max_pages} pages "
              f"(OCR_MAX_PAGES); {len(pages) - settings.ocr_max_pages} left unread")
    return out


if __name__ == "__main__":
    print(describe())
    print(f"  vision available:    {vision_available()}")
    print(f"  tesseract available: {tesseract_available()}")
    if len(sys.argv) > 1:
        import time

        target = sys.argv[1]
        start = time.time()
        found = ocr_pdf_pages(target, list(range(1, 4)))
        print(f"\n{target}: {len(found)} page(s) recognised in "
              f"{time.time() - start:.1f}s")
        for text, number in found:
            print(f"\n--- p.{number} ---\n{text[:400]}")
