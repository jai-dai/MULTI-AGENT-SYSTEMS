"""Will this machine survive the configuration it was asked to run?

The heavy parts of this project are not the agent — they are the local models:
a cross-encoder reranker (~1.1 GB for bge-reranker-base) and, with
EMBEDDING_BACKEND=local, an embedding model on top of that. On a laptop with
8 GB of RAM, half of which a Docker VM may already hold, that is the difference
between "slow" and "the machine starts swapping and everything stalls".

So the requirement is estimated from things that can actually be measured —
the size of the model weights on disk, the dimensions recorded in the index
manifest, the memory the OS reports as available right now — and compared
against reality BEFORE anything is loaded. Three verdicts:

    ok            run
    tight         run, but say what will be uncomfortable
    insufficient  do not start silently; offer the API route instead

Nothing here is a hard gate: PREFLIGHT=off skips it entirely. The point is an
informed decision, not a nanny.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import Path

from config import settings

GB = 1024 ** 3

# Weight sizes for models that are not downloaded yet (fp32 on disk ≈ what the
# process will hold). Measured where possible, rounded up.
KNOWN_MODEL_GB = {
    "BAAI/bge-reranker-base": 1.1,
    "BAAI/bge-reranker-large": 2.3,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 0.09,
    "BAAI/bge-m3": 2.3,
    "intfloat/multilingual-e5-large": 2.2,
    "intfloat/multilingual-e5-base": 1.1,
    "sentence-transformers/all-MiniLM-L6-v2": 0.09,
}
UNKNOWN_MODEL_GB = 1.5          # assume the expensive case, not the cheap one
TORCH_RUNTIME_GB = 0.35         # interpreter + torch, measured empirically
BASE_PROCESS_GB = 0.25

# Below this fraction of available memory the run is called "tight".
TIGHT_RATIO = 0.7


# --------------------------------------------------------------------------- #
# measuring the machine
# --------------------------------------------------------------------------- #


def _total_ram() -> int:
    if sys.platform == "darwin":
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                      capture_output=True, text=True,
                                      timeout=5).stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _available_ram() -> int:
    """Memory the OS could hand out now — not 'free', which is always small."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return 0
        page = 4096
        size_match = re.search(r"page size of (\d+) bytes", out)
        if size_match:
            page = int(size_match.group(1))
        pages = {k.strip(): int(v) for k, v in
                 re.findall(r"^(.*?):\s+(\d+)\.$", out, re.MULTILINE)}
        reusable = ("Pages free", "Pages inactive", "Pages speculative",
                    "Pages purgeable")
        return sum(pages.get(name, 0) for name in reusable) * page
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _docker_reserved() -> int:
    """A running Docker VM holds its memory whether or not it is doing work."""
    if not shutil.which("docker"):
        return 0
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _hf_cache_size(model: str) -> int:
    """Size of an already-downloaded model, or 0 if it is not cached."""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    folder = root / "hub" / ("models--" + model.replace("/", "--"))
    if not folder.exists():
        return 0
    # Skip symlinks: the HF cache keeps weights once in blobs/ and links them
    # from snapshots/. Following both counts every byte twice.
    return sum(f.stat().st_size for f in folder.rglob("*")
               if f.is_file() and not f.is_symlink())


def _model_need(model: str) -> tuple[float, bool]:
    """(gigabytes, already_downloaded)"""
    cached = _hf_cache_size(model)
    if cached:
        return cached / GB, True
    return KNOWN_MODEL_GB.get(model, UNKNOWN_MODEL_GB), False


def _index_need() -> float:
    """RAM the index itself will occupy: vectors + chunk texts + BM25 postings."""
    directory = Path(__file__).parent / settings.index_dir
    manifest, chunks_file = directory / "manifest.json", directory / "chunks.json"
    if not chunks_file.exists():
        return 0.0
    import json

    text_bytes = chunks_file.stat().st_size
    vectors = 0
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            dim = int(data.get("embedding", {}).get("dim", 0))
            count = len(json.loads(chunks_file.read_text(encoding="utf-8")))
            vectors = dim * count * 4                      # float32
        except (ValueError, OSError):
            vectors = 0
    # BM25 keeps tokenised copies of every chunk; ~2x the raw text in practice.
    return (vectors + text_bytes * 3) / GB


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    verdict: str                       # ok | tight | insufficient
    needed_gb: float
    available_gb: float
    total_gb: float
    free_disk_gb: float
    download_gb: float
    lines: list[str] = field(default_factory=list)
    remedies: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"


def assess(*, includes_reranker: bool = True) -> Report:
    # Everything below is in GIGABYTES. The probes return bytes; converting at
    # the boundary (rather than at each comparison) is what keeps `needed` and
    # `available` comparable — mixing the two silently passes every check.
    total = _total_ram() / GB
    available = _available_ram() / GB
    docker = _docker_reserved() / GB
    free_disk = shutil.disk_usage(Path(__file__).parent).free / GB

    lines: list[str] = []
    needed = BASE_PROCESS_GB
    download = 0.0
    uses_torch = False

    # Ingestion never loads the cross-encoder — counting it there would refuse
    # a run that fits, which is worse than not checking at all.
    if includes_reranker and settings.rerank_enabled:
        rerank_gb, rerank_cached = _model_need(settings.reranker_model)
        needed += rerank_gb
        uses_torch = True
        if not rerank_cached:
            download += rerank_gb
        lines.append(f"reranker {settings.reranker_model}: {rerank_gb:.1f} GB"
                     + ("" if rerank_cached else "  (not downloaded yet)"))

    if settings.embedding_backend == "local":
        embed_gb, embed_cached = _model_need(settings.embedding_model)
        needed += embed_gb
        if not embed_cached:
            download += embed_gb
        lines.append(f"local embedder {settings.embedding_model}: {embed_gb:.1f} GB"
                     + ("" if embed_cached else "  (not downloaded yet)"))
    else:
        lines.append(f"embeddings via {settings.embedding_backend} API: 0 GB local")

    if uses_torch:
        needed += TORCH_RUNTIME_GB
        lines.append(f"torch runtime: {TORCH_RUNTIME_GB:.2f} GB")

    index_gb = _index_need()
    if index_gb:
        needed += index_gb
        shown = (f"{index_gb * 1024:.0f} MB" if index_gb < 1 else f"{index_gb:.2f} GB")
        lines.append(f"index in memory: {shown}")

    if docker:
        lines.append(f"Docker VM already holding: {docker:.1f} GB "
                     "(already excluded from 'available')")

    verdict = "ok"
    remedies: list[str] = []
    if available and needed > available:
        verdict = "insufficient"
    elif available and needed > available * TIGHT_RATIO:
        verdict = "tight"

    if verdict != "ok":
        if settings.embedding_backend == "local":
            remedies.append(
                "EMBEDDING_BACKEND=openai — embeddings computed by the API, "
                "nothing loaded here (the corpus text then leaves the machine)")
        if settings.reranker_model != "cross-encoder/ms-marco-MiniLM-L-6-v2":
            remedies.append(
                "RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 — 0.09 GB "
                "instead of 1.1 GB, slightly weaker ranking")
        remedies.append(
            "RERANK_ENABLED=false — skip the cross-encoder entirely; hybrid "
            "search still runs and RRF still orders the results")
        if docker:
            remedies.append(
                f"stop the Docker VM (or lower its memory in Settings → "
                f"Resources) to give back {docker:.1f} GB")

    if download and download > free_disk:
        verdict = "insufficient"
        remedies.append(f"free {download:.1f} GB of disk for the model download")

    return Report(verdict=verdict, needed_gb=needed, available_gb=available,
                  total_gb=total, free_disk_gb=free_disk,
                  download_gb=download, lines=lines, remedies=remedies)


def render(report: Report) -> str:
    mark = {"ok": "✅", "tight": "⚠️ ", "insufficient": "⛔"}[report.verdict]
    headline = {
        "ok": "the machine can carry this configuration",
        "tight": "this will fit, but with little room to spare",
        "insufficient": "this configuration does not fit in memory",
    }[report.verdict]

    out = [f"{mark} preflight: {headline}",
           f"   machine: {report.total_gb:.1f} GB RAM total, "
           f"{report.available_gb:.1f} GB available now, "
           f"{report.free_disk_gb:.0f} GB free disk, "
           f"{platform.machine()} / {platform.system()}",
           f"   this run needs about {report.needed_gb:.1f} GB:"]
    out += [f"     · {line}" for line in report.lines]
    if report.download_gb:
        out.append(f"   first run will download {report.download_gb:.1f} GB")
    if report.remedies:
        out.append("   ways out:")
        out += [f"     → {r}" for r in report.remedies]
    return "\n".join(out)


def loads_anything_locally() -> bool:
    """Is there a model to weigh at all?

    With API embeddings AND reranking off, nothing heavy is loaded and the
    check has nothing to say — it should not nag. Note that switching
    embeddings to an API is NOT enough on its own: the cross-encoder runs
    locally in every configuration and is the larger of the two (1.1 GB for
    bge-reranker-base against ~0 for an API embedder).
    """
    return settings.embedding_backend == "local" or settings.rerank_enabled


def check_qdrant_server() -> None:
    """Если настроен сервер Qdrant — убедиться, что он поднят.

    Без этой проверки не запущенный сервер выглядит как поломка поиска: агент
    стартует нормально и падает лишь на первом `knowledge_search`, посреди
    рассуждения, с сетевой ошибкой внутри стека qdrant_client. Причина — «забыл
    запустить процесс», а вид у неё как у сломанного индекса.

    Проверка не мешает работать без сервера: пустой QDRANT_URL означает
    встроенный режим, и тогда проверять нечего.
    """
    url = (settings.qdrant_url or "").strip()
    if not url:
        return
    if _qdrant_alive(url):
        return

    command = (settings.qdrant_start_cmd or "").strip()
    if not command:
        raise SystemExit(
            f"⛔ QDRANT_URL={url}, но сервер не отвечает.\n"
            "   Поднять его вручную, либо прописать QDRANT_START_CMD в .env, "
            "чтобы он поднимался сам.\n"
            "   Либо убрать QDRANT_URL — вернётся встроенный режим со старыми "
            "каталогами index*/qdrant.")

    import shlex
    import subprocess
    import time

    print(f"   Qdrant не отвечает — поднимаю: {command}")
    try:
        # start_new_session отвязывает процесс от нашей группы: он переживёт
        # выход агента, и следующий запуск застанет его уже поднятым. Вывод
        # уходит в файл, иначе он лез бы в диалог поверх ответов агента.
        log = open(Path(command).parent / "qdrant.log", "ab")
        subprocess.Popen(shlex.split(command), stdout=log, stderr=log,
                         start_new_session=True)
    except OSError as exc:
        raise SystemExit(f"⛔ не удалось запустить Qdrant ({exc}): {command}")

    for _ in range(30):                       # серверу нужно несколько секунд
        time.sleep(1)
        if _qdrant_alive(url):
            print("   Qdrant поднят")
            return
    raise SystemExit(
        f"⛔ Qdrant запущен, но за 30 с так и не ответил на {url}. "
        f"Смотрите {Path(command).parent / 'qdrant.log'}")


def _qdrant_alive(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def guard(*, interactive: bool = None, includes_reranker: bool = True) -> Report:
    """Assess, print, and decide whether to continue.

    Exits only on `insufficient`, and only without an explicit override —
    a wrong estimate must never make the project unusable.
    """
    if os.environ.get("PREFLIGHT", "").strip().lower() in ("off", "0", "false"):
        return Report("ok", 0, 0, 0, 0, 0)

    check_qdrant_server()

    if not loads_anything_locally():
        return Report("ok", 0, 0, 0, 0, 0)

    report = assess(includes_reranker=includes_reranker)
    print(render(report))

    if report.verdict == "ok":
        return report

    if os.environ.get("PREFLIGHT_OVERRIDE", "").strip().lower() in ("1", "true", "yes"):
        print("   PREFLIGHT_OVERRIDE is set — continuing anyway.")
        return report

    if interactive is None:
        interactive = sys.stdin.isatty()

    if report.verdict == "insufficient":
        if not interactive:
            print("   refusing to start. Set PREFLIGHT_OVERRIDE=1 to insist, "
                  "or apply one of the fixes above.")
            raise SystemExit(3)
        answer = input("   start anyway? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise SystemExit(3)
    return report


if __name__ == "__main__":
    print(render(assess()))
