"""ACP-сервер: три специалиста как сетевые агенты.

    .venv/bin/python acp_server.py            # http://127.0.0.1:8903

Planner, Researcher и Critic — те же, что в hw8, с теми же промптами и тем же
циклом. Изменился способ до них дозвониться: раньше супервизор звал метод
объекта, теперь шлёт HTTP-запрос по ACP.

# Три агента в ОДНОМ процессе — это не компромисс

Задание просит один ACP-сервер, и на первый взгляд это спор с духом протокола:
раз агенты общаются по сети, пусть и живут врозь. Но выигрыш здесь не в том,
чтобы разнести их подальше, а в том, что общий у них теперь ТОЛЬКО протокол:
ни общих переменных, ни общего реестра инструментов, ни возможности заглянуть
друг другу в состояние. Разнести по процессам можно в любой момент, ничего не
переписывая, — и именно это и означает «переведено на протокол».

# Почему у каждого агента своё соединение с SearchMCP

Задание требует, чтобы каждый агент подключался к SearchMCP сам. Соблазн был
сэкономить и открыть одно соединение на троих, но экономить тут нечего: дорогая
часть — кросс-энкодер на 1.1 ГБ — живёт ВНУТРИ SearchMCP и загружен один раз на
всех, независимо от числа клиентов. Три сессии к одному серверу стоят трёх
TCP-соединений, и за эту цену каждый агент получает свой набор инструментов —
границы роли (планировщику не дан `read_url`) проводятся на подключении, а не
уговорами в промпте.

# Память между раундами: то, что в hw8 досталось даром

В hw8 исследователь помнил первый раунд просто потому, что супервизор держал
один и тот же объект. Здесь объект в другом процессе, а каждый вызов ACP —
самостоятельный запрос, и без специальных мер второй раунд начинался бы с
чистого листа: критик просит доработать, а исследователь не помнит, что уже
искал.

Поэтому экземпляры живут в `_AGENTS` и ключуются идентификатором СЕССИИ ACP.
Сессию заводит супервизор (`Client(session=...)`), и она означает ровно то же,
что означал живой объект в hw8: один разговор с пользователем. `reset` в REPL
берёт новую сессию — и агенты честно начинают заново.

# Что уезжает обратно

Ответ — `Message` из двух частей: полезная нагрузка (текст находок либо JSON
структуры) и `stats` со счётчиком токенов. Вторая часть нужна потому, что в hw8
цена запроса складывалась из полей живых объектов, а теперь суб-агенты считают
свои токены в чужом процессе. Не вернёшь их явно — измерение «мультиагент дороже
одноагентного в 19 раз» станет невоспроизводимым.
"""
from __future__ import annotations

import acp_compat  # noqa: F401  — ДО acp_sdk, чинит импорт под uvicorn 0.52
import json
import threading

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, Server
from pydantic import BaseModel

from agents import critic as critic_agent
from agents import planner as planner_agent
from agents import research as research_agent
from agents.react import ReactAgent
from config import settings
from mcp_utils import McpToolset

server = Server()

SEARCH_MCP_URL = (f"http://{settings.protocol_host}:"
                  f"{settings.search_mcp_port}/mcp")

# Роль -> (модуль агента, набор инструментов). Один источник правды на всё ниже.
_ROLES = {
    "planner": planner_agent,
    "researcher": research_agent,
    "critic": critic_agent,
}

_toolsets: dict[str, McpToolset] = {}
_agents: dict[tuple[str, str], ReactAgent] = {}
_guard = threading.Lock()


def _toolset(role: str) -> McpToolset:
    """Соединение с SearchMCP для роли. Одно на роль, на все её сессии.

    Лениво, а не при старте: сервер, падающий на импорте из-за того, что сосед
    ещё не поднялся, невозможно запустить в произвольном порядке. Первый вызов
    честно скажет, что SearchMCP недоступен, — и это будет ошибка ОДНОГО запроса,
    а не всей системы.
    """
    with _guard:
        if role not in _toolsets:
            module = _ROLES[role]
            _toolsets[role] = McpToolset(
                SEARCH_MCP_URL, label=f"SearchMCP/{role}", only=module.TOOLS)
            print(f"  [{role}] подключился к SearchMCP: "
                  f"{', '.join(module.TOOLS)}", flush=True)
        return _toolsets[role]


def _agent_for(role: str, session_id: str) -> ReactAgent:
    """Экземпляр агента для этой сессии. Он же — память между раундами."""
    key = (session_id, role)
    with _guard:
        existing = _agents.get(key)
    if existing is not None:
        return existing

    toolset = _toolset(role)                  # вне замка: подключение не мгновенно
    built = _ROLES[role].build(toolset, depth=1)
    with _guard:
        # Пока подключались, запрос-сосед мог собрать своего. Тогда берём его:
        # два экземпляра на одну сессию — это два разных воспоминания.
        return _agents.setdefault(key, built)


# -- разговор по протоколу ----------------------------------------------- #

def _request_text(input: list[Message]) -> str:
    """Входящие сообщения -> запрос агенту.

    Берутся все текстовые части: ACP разрешает разбить сообщение на части, и
    склеивать их обратно — работа получателя.
    """
    chunks: list[str] = []
    for message in input:
        for part in message.parts:
            if part.content:
                chunks.append(str(part.content))
    return "\n\n".join(chunks).strip()


def _reply(payload: str | BaseModel, agent: ReactAgent) -> Message:
    """Результат агента -> сообщение ACP: нагрузка + счётчик токенов."""
    if isinstance(payload, BaseModel):
        # Структура едет JSON-ом, а не человекочитаемым текстом: супервизору
        # нужен ВЕРДИКТ как поле. Разбор прозы моделью — ровно та ошибка, из-за
        # которой в hw8 и появились схемы (см. structured.py).
        content, content_type = payload.model_dump_json(), "application/json"
    else:
        content, content_type = str(payload), "text/plain"

    return Message(role="agent", parts=[
        MessagePart(name="payload", content_type=content_type, content=content),
        MessagePart(name="stats", content_type="application/json",
                    content=json.dumps({"agent": agent.name,
                                        "tokens": agent.tokens,
                                        "calls": agent.calls})),
    ])


@server.agent(name="planner",
              description="Decompose a research request into an executable "
                          "ResearchPlan. Returns JSON.")
def planner(input: list[Message], context: Context) -> Message:
    agent = _agent_for("planner", str(context.session.id))
    print(f"\n[ACP] planner ← сессия {str(context.session.id)[:8]}", flush=True)
    return _reply(agent.run(_request_text(input)), agent)


@server.agent(name="researcher",
              description="Execute a research plan and report findings with "
                          "sources. Returns text.")
def researcher(input: list[Message], context: Context) -> Message:
    agent = _agent_for("researcher", str(context.session.id))
    print(f"\n[ACP] researcher ← сессия {str(context.session.id)[:8]}", flush=True)
    return _reply(agent.run(_request_text(input)), agent)


@server.agent(name="critic",
              description="Verify findings against the sources and return a "
                          "CritiqueResult with an APPROVE/REVISE verdict. JSON.")
def critic(input: list[Message], context: Context) -> Message:
    agent = _agent_for("critic", str(context.session.id))
    print(f"\n[ACP] critic ← сессия {str(context.session.id)[:8]}", flush=True)
    return _reply(agent.run(_request_text(input)), agent)


def serve() -> None:
    """Поднять сервер, минуя `Server.run()`.

    Обёртка из acp-sdk 1.0.3 передаёт в `uvicorn.Config` около сорока пяти
    аргументов ПОЗИЦИОННО. Uvicorn 0.52 поменял порядок параметров, и значения
    поехали не в свои: сервер падал на `certfile should be a valid filesystem
    path`, хотя никакого TLS здесь нет и в помине. Диагноз стоил времени именно
    потому, что ошибка указывала не туда, где поломка.

    Чинить это заплаткой, как переименованный `LoopSetupType`, нельзя: там
    разошлось имя, здесь — соответствие аргументов, и подгонять его снаружи
    значит угадывать чужую сигнатуру. Зато чинить и не нужно. `Server.run` —
    обёртка над двумя строчками, а настоящий предмет протокола это ASGI-
    приложение: `create_app` собирает его из тех же агентов, и `uvicorn.run`
    запускает по ИМЕНАМ аргументов, где порядок не значит ничего.

    Заодно отключается `self_registration` — попытка объявить себя платформе
    BeeAI. Локальной домашке регистрироваться негде, а фоновая задача, которая
    ходит в сеть и молча падает, только засоряет лог.
    """
    import uvicorn
    from acp_sdk.server.app import create_app

    app = create_app(*server.agents, lifespan=server.lifespan)
    uvicorn.run(app, host=settings.protocol_host, port=settings.acp_port,
                log_level="warning")


if __name__ == "__main__":
    print(f"ACP → http://{settings.protocol_host}:{settings.acp_port}",
          flush=True)
    print(f"агенты: {', '.join(_ROLES)}; инструменты по MCP: {SEARCH_MCP_URL}",
          flush=True)
    serve()
