"""Наблюдаемость: один trace на запуск, сшитый через границы процессов.

# Задача, которой не было в предыдущих работах

Требование задания: суб-агенты должны быть ВЛОЖЕНЫ под один родительский trace.
Звучит как настройка, но здесь это инженерная задача: агенты живут в РАЗНЫХ
ПРОЦЕССАХ за A2A, а трассировка по умолчанию не переживает сетевой вызов. Каждый
процесс начал бы свой trace, и в интерфейсе вместо дерева лежало бы четыре
несвязанных корня.

Решается пробросом контекста. Langfuse 4.x стоит на OpenTelemetry, а там
принадлежность к дереву задаётся парой чисел: `trace_id` (общий на всё дерево) и
`span_id` родителя. Достаточно довезти их до другого процесса и собрать там
`TraceContext` — и вызов встанет веткой под своим родителем.

# Почему через metadata сообщения, а не через HTTP-заголовки

W3C для этого предлагает заголовок `traceparent`, и через `ClientCallInterceptor`
его можно было бы поставить. Но в A2A у сообщения ЕСТЬ своё поле `metadata`
(`google.protobuf.Struct`), которое сервер отдаёт в `RequestContext.metadata`, —
то есть протокол уже предусмотрел место для «данных о запросе, не являющихся
запросом». Заголовок жил бы уровнем ниже протокола и потерялся бы при смене
транспорта на gRPC; metadata переживёт.

# Что здесь НЕ делается

Ключи Langfuse необязательны. Нет ключей — `enabled()` вернёт False, все хелперы
станут пустышками, и система работает как работала. Наблюдаемость не должна быть
условием работоспособности: телеметрия, роняющая продукт, — худший вид телеметрии.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace as otel_trace

from config import settings

TRACE_ID_KEY = "langfuse_trace_id"
PARENT_SPAN_KEY = "langfuse_parent_span_id"

_client = None


def enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def client():
    """Клиент Langfuse, либо None. Создаётся один раз на процесс."""
    global _client
    if not enabled():
        return None
    if _client is None:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            # См. блок «Маскирование» ниже: в облако не должно уехать то, что
            # не уезжает в git.
            mask=mask,
        )
    return _client


def flush() -> None:
    """Дослать накопленное. Обязательно перед выходом: экспорт асинхронный, и
    процесс, завершившийся сразу после ответа, увозит хвост трассы с собой."""
    if _client is not None:
        _client.flush()


# --------------------------------------------------------------------------- #
# проброс контекста между процессами
# --------------------------------------------------------------------------- #

def root_span(name: str, **kwargs):
    """Собственный корневой span на весь запуск. Без него ничего не работает.

    # Почему нельзя положиться на span, который заводит CallbackHandler

    Первая версия так и делала — брала текущий span из OpenTelemetry и надеялась,
    что он есть. В СИНХРОННОМ инструменте он действительно есть; в `async def` —
    НЕТ. Замерено: тот же `carrier()` из синхронного инструмента возвращает пару
    чисел, из асинхронного — пустой словарь.

    Причина в том, что `CallbackHandler` делает span текущим только на время
    своих собственных синхронных колбэков. Тело асинхронного инструмента
    исполняется позже и вне этого окна, а наши делегирования по A2A — как раз
    `async def`, потому что LangGraph асинхронен.

    Итог был бы незаметным и разрушительным: система работает, трейсы уходят,
    но вместо трёх деревьев в интерфейсе лежит десять несвязанных корней —
    супервизор отдельно, каждый суб-агент отдельно.

    Лечится не подпоркой, а сменой подхода: контекстом надо ВЛАДЕТЬ. Свой span,
    открытый обычным `with`, живёт на весь запуск и переживает и `await`, и
    `asyncio.create_task` — contextvars копируются в задачу штатно. Проверено
    отдельным тестом, а не выведено из документации.
    """
    lf = client()
    if lf is None:
        import contextlib

        return contextlib.nullcontext()
    return lf.start_as_current_observation(name=name, as_type="agent", **kwargs)


def carrier() -> dict[str, str]:
    """Текущий контекст трассировки -> то, что можно положить в metadata.

    Берётся из OpenTelemetry напрямую, а не из Langfuse: формат `trace_id` и
    `span_id` задан стандартом (32 и 16 hex-символов), и не зависит от того,
    какая библиотека их породила. Работает только внутри `root_span()` —
    см. объяснение выше.
    """
    span = otel_trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return {}
    return {
        TRACE_ID_KEY: format(context.trace_id, "032x"),
        PARENT_SPAN_KEY: format(context.span_id, "016x"),
    }


def trace_context(metadata: dict[str, Any] | None):
    """metadata запроса -> `TraceContext` для CallbackHandler, либо None.

    None означает «этот вызов пришёл не из трассируемого дерева» — например,
    агента дёрнули напрямую через curl. Тогда он начнёт свой собственный trace,
    и это правильное поведение, а не ошибка.
    """
    if not enabled() or not metadata:
        return None
    trace_id = metadata.get(TRACE_ID_KEY)
    if not trace_id:
        return None
    from langfuse.types import TraceContext

    parent = metadata.get(PARENT_SPAN_KEY)
    return TraceContext(trace_id=trace_id, parent_span_id=parent) if parent \
        else TraceContext(trace_id=trace_id)


def handler(trace_ctx=None):
    """CallbackHandler для LangChain/LangGraph, либо None.

    Передаётся в `ainvoke(..., config={"callbacks": [handler]})`. Один хендлер
    на вызов, а не на процесс: `trace_context` у каждого свой.
    """
    if not enabled():
        return None
    from langfuse.langchain import CallbackHandler

    if trace_ctx is not None:
        return CallbackHandler(trace_context=trace_ctx)
    return CallbackHandler()


def callbacks(trace_ctx=None) -> list:
    """То же, но сразу списком — как этого ждёт `config={"callbacks": [...]}`."""
    one = handler(trace_ctx)
    return [one] if one is not None else []


# --------------------------------------------------------------------------- #
# промпты из Langfuse
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=32)
def _cached(name: str, label: str):
    return client().get_prompt(name, label=label)


def prompt(name: str, label: str | None = None, **variables) -> str:
    """System prompt по имени. НИ ОДНОГО текста промпта в этом репозитории нет.

    Это прямое требование задания, и оно меняет больше, чем кажется. Промпт
    перестаёт быть частью кода и становится ДАННЫМИ с версией и лейблом: его
    можно поправить, не трогая деплой, и откатить, не трогая git.

    Плата названа честно: система теперь не поднимается без сети до Langfuse.
    Поэтому у каждого вызова есть `fallback` — короткий текст, которого хватит,
    чтобы агент не превратился в пустую оболочку, и по которому сразу видно, что
    настоящий промпт не доехал.
    """
    lf = client()
    if lf is None:
        raise RuntimeError(
            f"промпт '{name}' живёт в Langfuse, а ключи не заданы. "
            "Впиши LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY в .env")
    obtained = _cached(name, label or settings.langfuse_prompt_label)
    return obtained.compile(**variables) if variables else obtained.prompt


def prompt_object(name: str, label: str | None = None):
    """Сам объект промпта — нужен, чтобы связать trace с версией промпта.

    Связь не косметическая: без неё в интерфейсе видно «ответ стал хуже», но не
    видно, что накануне поменяли третью строку системного промпта.
    """
    lf = client()
    if lf is None:
        return None
    return _cached(name, label or settings.langfuse_prompt_label)

# --------------------------------------------------------------------------- #
# Маскирование: тот же список, что охраняет git push
# --------------------------------------------------------------------------- #
#
# В Langfuse уезжают не только промпты и ответы, но и РЕЗУЛЬТАТЫ ВЫЗОВОВ
# ИНСТРУМЕНТОВ — то есть пассажи из корпуса вместе с именами файлов. На публичной
# демонстрации это три PDF про RAG и бояться нечего. Наведи ту же систему на
# личный индекс — и в облако поедет настоящая переписка с именами контрагентов.
#
# В проекте уже есть список того, что не должно покидать машину:
# `.private-names.txt`, по которому pre-push хук блокирует публикацию. Странно
# охранять им git и не охранять телеметрию — утечка через трейсы ничем не лучше
# утечки через коммит, только заметить её нельзя вовсе.
#
# Поэтому маска берёт ТОТ ЖЕ файл. Один источник правды, двое ворот.
#
# Сопоставление повторяет `check_names.py`: по основе с границей слова слева,
# без учёта регистра — так ловятся падежные формы («Петренку», «Петренком»),
# на которых ручное маскирование однажды уже прокололось.

_NAMES_FILE = Path(__file__).resolve().parent / ".private-names.txt"
_patterns: list = []
_patterns_loaded = False


def _load_patterns() -> list:
    global _patterns, _patterns_loaded
    if _patterns_loaded:
        return _patterns
    _patterns_loaded = True
    if not _NAMES_FILE.exists():
        return _patterns
    names = []
    for line in _NAMES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    _patterns = [(n, re.compile(r"\b" + re.escape(n), re.IGNORECASE)) for n in names]
    return _patterns


def _blur(name: str) -> str:
    """Первая и последняя буквы, остальное звёздочками — как в README проекта."""
    if len(name) <= 2:
        return "*" * len(name)
    return name[0] + "*" * (len(name) - 2) + name[-1]


def mask(data: Any) -> Any:
    """Маска для Langfuse: рекурсивно чистит всё, что уезжает наружу.

    Вызывается SDK на КАЖДОМ поле трассы перед отправкой, поэтому дешевизна
    важнее изящества: без совпадений строка возвращается как есть.
    """
    patterns = _load_patterns()
    if not patterns:
        return data
    if isinstance(data, str):
        out = data
        for name, pattern in patterns:
            if pattern.search(out):
                out = pattern.sub(_blur(name), out)
        return out
    if isinstance(data, dict):
        return {k: mask(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(mask(v) for v in data)
    return data
