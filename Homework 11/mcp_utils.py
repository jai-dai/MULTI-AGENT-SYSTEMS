"""Инструменты MCP — в том виде, в каком их принимает наш цикл ReAct.

В лекции этот файл называется `mcp_tools_to_langchain`. Здесь он
`mcp_tools_to_registry`, и разница не в названии: у нас нет LangChain (решение
из hw8 — цикл свой), поэтому переводить инструменты нужно не в
`StructuredTool`, а в ту пару, которой цикл пользуется с hw5:

    registry  {имя: вызываемое, возвращающее строку}
    schemas   [схема в формате OpenAI]

Формат OpenAI взят канонической серединой не из симпатии к нему: `llm.py` уже
переводит его в оба протокола моделей, поэтому инструмент, приехавший по MCP, и
инструмент, написанный руками, доходят до модели одинаковыми. Цикл агента при
этом не знает, что за именем — сеть.

# Что происходит со схемой

MCP отдаёт `input_schema` — обычный JSON Schema. OpenAI ждёт JSON Schema в
`function.parameters`. Перевод сводится к перекладыванию трёх полей, и это
приятная неожиданность: два протокола независимо сошлись на одном языке
описания аргументов.

# Ошибка остаётся текстом

Договор инструментов из hw5: что бы ни случилось внутри, наружу идёт СТРОКА, а
не исключение — цикл обязан пережить кривой аргумент и попробовать иначе. Сеть
добавила новых способов не сработать (сервер лежит, таймаут, протокольная
ошибка), и все они складываются туда же. Модель, получившая
«ERROR: SearchMCP не отвечает», ищет другим инструментом. Модель, получившая
исключение, роняет разговор.

Единственное исключение из этого правила — момент ПОДКЛЮЧЕНИЯ. Если сервера нет
при старте, это не «инструмент не сработал», а «системы нет», и падать надо
громко, у пользователя на глазах.
"""
from __future__ import annotations

import json
import threading
from contextlib import AsyncExitStack
from typing import Callable

from mcp import Client

from bridge import AsyncBridge

# Инструмент может лезть в веб и в кросс-энкодер; минуты — это норма, а не
# признак беды. Потолок нужен, чтобы «сервер молча умер» не выглядело как
# «агент думает».
CALL_TIMEOUT = 180.0
CONNECT_TIMEOUT = 30.0


def mcp_tools_to_registry(tools: list, call: Callable[[str, dict], str],
                          only: list[str] | None = None
                          ) -> tuple[dict[str, Callable], list[dict]]:
    """Инструменты MCP -> (реестр, схемы) для ReactAgent.

    `only` — подмножество для роли. Набор инструментов это границы роли (довод
    из hw8: планировщик с `read_url` начинает читать статьи вместо того, чтобы
    составить план), и теперь, когда сервер один на всех, фильтр — единственное
    место, где эти границы вообще проводятся.
    """
    available = {t.name: t for t in tools}
    if only is not None:
        missing = [n for n in only if n not in available]
        if missing:
            raise KeyError(
                f"MCP-сервер не отдаёт инструменты {missing}; "
                f"есть: {sorted(available)}")
        chosen = [available[n] for n in only]
    else:
        chosen = list(tools)

    registry: dict[str, Callable] = {}
    schemas: list[dict] = []
    for tool in chosen:
        registry[tool.name] = _make_caller(tool.name, call)
        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object",
                                                    "properties": {}},
            },
        })
    return registry, schemas


def _make_caller(name: str, call: Callable[[str, dict], str]) -> Callable:
    """Замыкание на имя инструмента.

    Отдельной функцией, а не лямбдой в цикле: лямбда захватила бы переменную
    цикла, и все инструменты в реестре звали бы последний.
    """
    def invoke(**kwargs) -> str:
        return call(name, kwargs)
    invoke.__name__ = name
    return invoke


class McpToolset:
    """Живое соединение с одним MCP-сервером плюс его инструменты для агента.

    Соединение открывается в конструкторе и держится, пока объект жив: список
    инструментов нужен до первого шага агента (без схем модель не знает, что
    вызывать), а сессия — на всех последующих.
    """

    def __init__(self, url: str, *, label: str = "MCP",
                 only: list[str] | None = None) -> None:
        self.url = url
        self.label = label
        self._bridge = AsyncBridge(name=f"mcp-{label}")
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        # Сессия одна, а звать её могут из разных потоков (ACP-сервер держит пул).
        # Наш цикл строго последователен, но полагаться на это не стоит: замок
        # стоит дёшево, а перепутанные ответы отлаживаются дорого.
        self._lock = threading.Lock()

        self._client, tools = self._bridge.call(self._connect(), CONNECT_TIMEOUT)
        self.tools = tools
        self.registry, self.schemas = mcp_tools_to_registry(tools, self._call, only)

    # -- соединение ------------------------------------------------------- #

    async def _connect(self):
        stack = AsyncExitStack()
        client = await stack.enter_async_context(Client(self.url))
        result = await client.list_tools()
        self._stack = stack
        return client, list(result.tools)

    def _call(self, name: str, arguments: dict) -> str:
        """Один инструмент по сети. Любая беда — читаемой строкой."""
        try:
            with self._lock:
                result = self._bridge.call(
                    self._client.call_tool(name, arguments), CALL_TIMEOUT)
        except TimeoutError:
            return (f"ERROR: {self.label} не ответил за {CALL_TIMEOUT:.0f} с "
                    f"на вызов '{name}'. Попробуй другой инструмент или запрос.")
        except Exception as exc:
            return (f"ERROR: вызов '{name}' на {self.label} не удался "
                    f"({type(exc).__name__}: {exc}).")

        text = _text_of(result)
        if getattr(result, "is_error", False):
            # Сервер сам сказал, что вызов неудачен. Его объяснение полезнее
            # нашего: это оно знает, чего не хватило в аргументах.
            return text if text.startswith("ERROR") else f"ERROR: {text}"
        return text

    # -- ресурсы ---------------------------------------------------------- #

    def read_resource(self, uri: str) -> str:
        """Прочитать ресурс MCP. Ресурс ОПИСЫВАЕТ, в отличие от инструмента.

        Ошибка здесь тоже строка, но по другой причине: ресурс читает не модель,
        а наш код — и обзор базы знаний, которого не случилось, не повод не
        начинать работу.
        """
        try:
            with self._lock:
                result = self._bridge.call(
                    self._client.read_resource(uri), CALL_TIMEOUT)
        except Exception as exc:
            return f"ERROR: ресурс {uri} не прочитан ({type(exc).__name__}: {exc})."
        # `.contents`, а не сам результат: `ReadResourceResult` — модель pydantic,
        # и итерирование по НЕЙ даёт пары (поле, значение), а не части ресурса.
        # Молча возвращает пустоту, поэтому названо явно.
        return "\n".join(str(getattr(c, "text", ""))
                         for c in (result.contents or []))

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._bridge.call(self._stack.aclose(), 10)
            except Exception:
                pass                      # закрываемся — жаловаться уже некому
            self._stack = None
        self._bridge.close()


def _text_of(result) -> str:
    """`CallToolResult` -> строка для модели.

    Модель читает текст, поэтому текстовые части склеиваются, а всё остальное
    (картинки, ссылки на ресурсы) честно называется своим типом, а не молча
    пропадает: инструмент, вернувший картинку, не должен выглядеть как
    инструмент, вернувший пустоту.
    """
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(f"[{getattr(item, 'type', type(item).__name__)}]")
    if parts:
        return "\n".join(parts)

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return ""
