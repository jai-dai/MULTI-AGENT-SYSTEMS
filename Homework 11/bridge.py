"""Мост между синхронным циклом агента и асинхронными клиентами протоколов.

# Зачем он вообще нужен

Цикл ReAct синхронный, и это не недосмотр: он написан так, потому что «спросить
человека» — это `input()`, а рассуждение агента — цепочка шагов, где каждый
следующий зависит от предыдущего. Асинхронность там нечего распараллеливать.

А клиенты обоих протоколов асинхронные: и `mcp.Client`, и `acp_sdk.client.Client`
живут только внутри `async with`. Между ними и нужен переводчик.

# Почему собственный цикл в отдельном потоке, а не asyncio.run на каждый вызов

`asyncio.run` создаёт цикл и убивает его вместе с задачей. Сессия MCP пережить
этого не может: она открыта внутри `async with`, привязана к своему циклу и
закрылась бы после первого же инструмента. Каждый следующий вызов платил бы
заново за TCP, инициализацию протокола и список инструментов — на разговоре в
полсотни вызовов это заметно.

Здесь цикл живёт в фоновом потоке столько же, сколько объект-владелец, а
синхронная сторона забрасывает в него корутины через
`run_coroutine_threadsafe`. Соединение открывается один раз.

Второй, менее очевидный довод — переиспользование. ACP-сервер выполняет
синхронные функции агентов в своём пуле потоков (`run_in_executor`), то есть у
вызывающего потока СВОЕГО цикла нет вообще, а сервер крутит свой и трогать его
чужими задачами нельзя. Отдельный цикл снимает вопрос: он ничей.

# Про таймаут

`call` без таймаута — это возможность подвесить REPL навсегда: сервер по ту
сторону может думать, а может и умереть молча. Таймаут превращает второе в
ошибку, а не в тишину.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine


class AsyncBridge:
    """Фоновый цикл событий: синхронный код исполняет корутины и ждёт результат."""

    def __init__(self, name: str = "bridge") -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._spin, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _spin(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def call(self, coro: Coroutine, timeout: float | None = None) -> Any:
        """Исполнить корутину в фоновом цикле и вернуть её результат сюда."""
        if not self._thread.is_alive():
            coro.close()
            raise RuntimeError("мост уже закрыт")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            # Отменяем задачу в её собственном цикле, иначе она продолжит жить
            # и держать соединение уже после того, как ждать её перестали.
            future.cancel()
            raise

    def close(self) -> None:
        """Остановить цикл и дождаться потока. Повторный вызов безвреден."""
        if not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
