"""Вызов агента по ACP — синхронной функцией, как того ждёт цикл супервизора.

Пара к `mcp_utils.py`, и устроена так же: асинхронный клиент живёт в фоновом
цикле (`bridge.py`), наружу торчит обычный вызов. Симметрия не случайна — для
супервизора «позвать инструмент» и «позвать агента» должны выглядеть одинаково,
иначе цикл ReAct пришлось бы учить второму способу общения.

# Заголовок, без которого ничего не работает

`Client` из acp-sdk 1.0.3 шлёт тело запроса через `content=`, не выставляя
`Content-Type: application/json`. Старые версии FastAPI такое прощали и
разбирали тело как JSON по факту; FastAPI 0.141 не прощает и отвечает `422
Unprocessable Entity`, жалуясь на «body: input should be a valid dictionary» —
и показывает при этом совершенно валидный JSON. Ошибка указывает не туда, где
поломка, поэтому заголовок проставлен здесь явно и с объяснением.

# Почему упавший агент — это строка, а не исключение

В hw8 суб-агент, не справившийся со структурой, ронял исключение до самого
REPL: они жили в одном процессе, и его смерть была смертью системы. По сети
это уже не так. Агент на том конце может не ответить по причинам, к работе
отношения не имеющим, — сервер перезапускают, порт занят, процесс убит. Для
координатора это обычное положение дел, а не конец света, и узнать о нём он
должен так же, как узнаёт о неудачном инструменте: текстом, с которым можно
что-то решить. Поэтому провал приезжает строкой «ERROR: ...», и модель вольна
позвать ещё раз или пойти дальше без этого шага.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from acp_sdk.client import Client
from acp_sdk.models import Session

from bridge import AsyncBridge

# Исследователь с двенадцатью шагами по сети — это минуты. Потолок здесь про
# «сервер умер молча», а не про «агент долго думает».
RUN_TIMEOUT = 1800.0


@dataclass
class AcpReply:
    """Ответ агента: полезная нагрузка и цена, которую он за неё заплатил."""

    text: str
    tokens: int = 0
    calls: int = 0
    failed: bool = False


class AcpAgents:
    """Клиент ACP-сервера. Одна сессия — один разговор с пользователем."""

    def __init__(self, base_url: str, *, timeout: float = RUN_TIMEOUT) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._bridge = AsyncBridge(name="acp")
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._open()

    def _open(self) -> None:
        async def connect():
            stack = AsyncExitStack()
            client = await stack.enter_async_context(Client(
                base_url=self.base_url,
                session=Session(),
                # См. шапку модуля: без этого сервер отвечает 422 на валидный JSON.
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            ))
            return stack, client

        self._stack, self._client = self._bridge.call(connect(), 30)

    # -- вызов ------------------------------------------------------------ #

    def names(self) -> list[str]:
        """Кто вообще есть на сервере. Discovery — первый шаг протокола."""
        async def discover():
            return [a.name async for a in self._client.agents()]
        try:
            return self._bridge.call(discover(), 30)
        except Exception:
            return []

    def call(self, agent: str, request: str) -> AcpReply:
        """Один агент, один запрос, один ответ."""
        async def run():
            return await self._client.run_sync(request, agent=agent)

        try:
            run_result = self._bridge.call(run(), self.timeout + 30)
        except TimeoutError:
            return AcpReply(
                f"ERROR: агент '{agent}' не ответил за {self.timeout:.0f} с. "
                "Возможно, ACP-сервер упал — проверь его лог.", failed=True)
        except Exception as exc:
            return AcpReply(
                f"ERROR: вызов агента '{agent}' по ACP не удался "
                f"({type(exc).__name__}: {exc}).", failed=True)

        status = str(getattr(run_result, "status", ""))
        if run_result.error is not None or "COMPLETED" not in status.upper():
            return AcpReply(
                f"ERROR: агент '{agent}' завершился со статусом {status}: "
                f"{run_result.error}", failed=True)
        return _parse(run_result, agent)

    def new_session(self) -> None:
        """Забыть разговор. Новая сессия — новые экземпляры агентов на сервере."""
        self.close(keep_bridge=True)
        self._open()

    def close(self, *, keep_bridge: bool = False) -> None:
        if self._stack is not None:
            try:
                self._bridge.call(self._stack.aclose(), 10)
            except Exception:
                pass
            self._stack = self._client = None
        if not keep_bridge:
            self._bridge.close()


def _parse(run_result, agent: str) -> AcpReply:
    """`Run` -> нагрузка и статистика.

    Части ищутся ПО ИМЕНИ, а не по порядку: порядок — это соглашение между
    двумя нашими файлами, а имя — часть протокола, и оно переживёт добавление
    третьей части.
    """
    import json

    payload: list[str] = []
    reply = AcpReply(text="")
    for message in run_result.output:
        for part in message.parts:
            if part.name == "stats":
                try:
                    stats = json.loads(str(part.content))
                    reply.tokens += int(stats.get("tokens", 0))
                    reply.calls += int(stats.get("calls", 0))
                except (ValueError, TypeError):
                    pass                      # цена — не повод терять результат
            elif part.content:
                payload.append(str(part.content))

    reply.text = "\n".join(payload).strip()
    if not reply.text:
        reply.text = f"ERROR: агент '{agent}' вернул пустой ответ."
        reply.failed = True
    return reply
