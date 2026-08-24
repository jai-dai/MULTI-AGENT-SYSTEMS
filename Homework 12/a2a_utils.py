"""Вызов агента по A2A — одной строкой, как этого ждёт супервизор.

# Почему здесь нет моста в фоновый поток

В версии на ACP такой мост был: цикл агента был синхронным, а клиент протокола —
асинхронным, и их приходилось сшивать отдельным потоком с собственным циклом
событий. Здесь супервизор — это LangGraph, он асинхронный сам, поэтому мост
исчез целиком. Ровно тот случай, когда смена фреймворка убирает не строчку, а
файл.

# Почему упавший агент — это строка, а не исключение

По сети сосед может не ответить по причинам, к работе отношения не имеющим:
сервер перезапускают, порт занят, процесс убит. Для координатора это обычное
положение дел, и узнать о нём он должен так же, как узнаёт о неудачном
инструменте — текстом, с которым можно что-то решить. Поэтому провал приезжает
строкой «ERROR: ...», и модель вольна позвать ещё раз или пойти дальше.

# Что здесь появилось ради наблюдаемости

Ровно три строки: перед отправкой в `metadata` кладутся `trace_id` и `span_id`
текущего шага. Больше ничего — ни своих спанов, ни обёрток. Вся работа по сшиванию
дерева делается на ПРИЁМНОЙ стороне, где есть подо что вкладываться.

# Клиент на вызов, а не на процесс

`create_client` сначала читает Agent Card по известному адресу и только потом
открывает соединение. Держать его между вызовами можно, но выигрыш в
миллисекундах не стоит висящего сокета к соседу, которого могут перезапустить.
"""
from __future__ import annotations

from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest

import observability

CALL_TIMEOUT = 1800.0


async def ask(url: str, request: str, timeout: float = CALL_TIMEOUT,
              carrier: dict | None = None) -> str:
    """Один агент, один запрос, один ответ."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            client = await create_client(
                url,
                client_config=ClientConfig(streaming=False, httpx_client=http),
            )
            message = new_text_message(request, role=Role.ROLE_USER)
            # Контекст трассировки едет в metadata СООБЩЕНИЯ, а не в заголовке
            # HTTP: metadata — часть протокола и переживёт смену транспорта на
            # gRPC, заголовок живёт уровнем ниже и потеряется. Без этих двух
            # чисел агент на том конце начал бы СВОЙ trace, и вместо дерева в
            # интерфейсе лежали бы четыре несвязанных корня.
            # Контекст берётся ЯВНО из аргумента, а не из окружения. Причина в
            # шапке: внутри асинхронного инструмента LangGraph окружающего
            # контекста нет, и `observability.carrier()` вернул бы пустоту.
            # Fallback на окружение оставлен для прямых вызовов мимо супервизора.
            ctx = carrier if carrier is not None else observability.carrier()

            # ВНИМАНИЕ: metadata в A2A ДВА, и это стоило часа отладки.
            # У `Message` своё поле `metadata`, у `SendMessageRequest` — своё.
            # Сервер отдаёт в `RequestContext.metadata` метаданные ЗАПРОСА:
            #     return json_format.MessageToDict(self._params.metadata)
            # Положи их в сообщение — и на том конце будет пусто, без единой
            # ошибки: запрос уйдёт, агент ответит, трассировка молча развалится
            # на несвязанные корни.
            request_message = SendMessageRequest(message=message)
            if ctx:
                request_message.metadata.update(ctx)

            parts: list[str] = []
            async for event in client.send_message(request_message):
                text = _text_of(event)
                if text:
                    parts.append(text)
            answer = "\n".join(parts).strip()
    except Exception as exc:
        return (f"ERROR: вызов агента по A2A ({url}) не удался "
                f"({type(exc).__name__}: {exc}).")

    if not answer:
        return f"ERROR: агент по адресу {url} вернул пустой ответ."
    return answer


def _text_of(event) -> str:
    """Из потока событий берём только сообщения и артефакты — статусы не нужны."""
    which = event.WhichOneof("payload") if hasattr(event, "WhichOneof") else None
    if which == "message":
        return get_message_text(event.message)
    if which == "task":
        task = event.task
        chunks = [get_message_text(m) for m in getattr(task.status, "update", [])] \
            if hasattr(task, "status") else []
        for artifact in getattr(task, "artifacts", []):
            for part in getattr(artifact, "parts", []):
                if part.HasField("text"):
                    chunks.append(part.text)
        history = getattr(task, "history", [])
        if history:
            chunks.append(get_message_text(history[-1]))
        return "\n".join(c for c in chunks if c)
    return ""


async def card(url: str) -> dict:
    """Прочитать Agent Card, не вызывая агента. В этом и смысл A2A-discovery."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.get(url.rstrip("/") + "/.well-known/agent-card.json")
        response.raise_for_status()
        return response.json()
