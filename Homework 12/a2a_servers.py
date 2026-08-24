"""Три A2A-сервера — по одному на агента.

    .venv/bin/python a2a_servers.py        # 8903 planner, 8904 researcher, 8905 critic

# Почему серверов три, а не один

Это и есть содержание перехода. В A2A **один сервер = один агент**: у агента своя
Agent Card по адресу `/.well-known/agent-card.json`, свой порт, своя жизнь. Один
процесс с тремя агентами внутри протоколу не противоречит, но и не выражается в
нём: карточка описывает АГЕНТА, а не набор агентов.

Практическое следствие важнее теоретического. Карточка — статический документ по
известному адресу, поэтому агента можно найти и описать, НЕ ПОДНИМАЯ его и не
имея клиента: достаточно `curl`. Discovery перестаёт быть вызовом и становится
чтением.

# Что здесь дублируется, а что нет

Три процесса, но кросс-энкодер на 1.1 ГБ по-прежнему один — он живёт в SearchMCP.
Каждый агент подключается к нему по MCP и получает инструменты через
`langchain-mcp-adapters`. Если бы поиск оставался импортом, три процесса означали
бы три копии модели, и на восьми гигабайтах система просто не поехала бы.

# AgentExecutor: где кончается LangChain и начинается протокол

`AgentExecutor` — это шов. Внутри `execute()` живёт обычный LangGraph-агент
(`create_agent(...).ainvoke(...)`), снаружи — контракт A2A: взять запрос из
`RequestContext`, положить ответ в `EventQueue` через `new_text_message`.
Протокол ничего не знает про LangChain, LangChain ничего не знает про протокол,
и это правильно: заменяемая часть — та, что внутри.

# Ленивая сборка агента, и почему без неё нельзя

Инструменты приезжают по сети, то есть SearchMCP должен быть поднят. Собирать
агента в момент старта сервера значит требовать порядка запуска: сначала MCP,
потом A2A. Поэтому сборка отложена до первого запроса — сервер поднимается
всегда, а падает только тот вызов, которому действительно не хватило соседа.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading

import uvicorn
from a2a.helpers.proto_helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (create_agent_card_routes, create_jsonrpc_routes,
                               create_rest_routes)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from starlette.applications import Starlette

import observability
from agents import critic, planner, research
from config import settings

SEARCH_MCP_URL = (f"http://{settings.protocol_host}:{settings.search_mcp_port}/mcp")

ROLES = {
    planner.NAME: (planner, settings.planner_port),
    research.NAME: (research, settings.researcher_port),
    critic.NAME: (critic, settings.critic_port),
}


async def _search_tools():
    """Инструменты из SearchMCP — по сети, как требует задание."""
    client = MultiServerMCPClient({
        "search": {"url": SEARCH_MCP_URL, "transport": "streamable_http"},
    })
    return await client.get_tools()


def _card(role, port: int) -> AgentCard:
    """Agent Card — паспорт агента, который читают до обращения к нему."""
    return AgentCard(
        name=role.NAME,
        description=role.DESCRIPTION,
        version="1.0.0",
        supported_interfaces=[AgentInterface(
            url=f"http://{settings.protocol_host}:{port}/",
            protocol_binding="JSONRPC",
        )],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(
            id=f"{role.NAME}-skill",
            name=role.NAME,
            description=role.DESCRIPTION,
            tags=["research", role.NAME],
            examples=list(role.SKILL_EXAMPLES),
        )],
    )


class LangChainExecutor(AgentExecutor):
    """Шов между протоколом и агентом. Всё, что специфично для LangChain, — здесь."""

    def __init__(self, role) -> None:
        self._role = role
        self._agent = None
        self._lock = asyncio.Lock()

    async def _ensure_agent(self):
        # Двойная проверка под замком: первый же параллельный запрос иначе
        # собрал бы второго агента и второй раз сходил бы в MCP за схемами.
        if self._agent is None:
            async with self._lock:
                if self._agent is None:
                    tools = await _search_tools()
                    self._agent = self._role.build(tools, settings.model_name)
        return self._agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request = context.get_user_input()
        try:
            agent = await self._ensure_agent()
            # Токены считаются ЗДЕСЬ, потому что тратятся здесь. Наружу они не
            # едут: в A2A ответ — это сообщение, а не пара «результат + счёт»,
            # и класть счётчики в `metadata` значит расширять протокол под свои
            # нужды. Поэтому цена агента видна только в ЕГО логе, и сложить её
            # с ценой супервизора можно лишь руками (см. README, «Ціна»).
            # Контекст из metadata запроса -> вызов встаёт ВЕТКОЙ под своим
            # родителем, живущим в другом процессе. Нет контекста (агента дёрнули
            # напрямую через curl) — начнётся свой trace, и это правильно.
            trace_ctx = observability.trace_context(context.metadata)
            with get_usage_metadata_callback() as usage:
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=request)]},
                    config={"callbacks": observability.callbacks(trace_ctx)})
            answer = _payload(result)
            total = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
            print(f"[{self._role.NAME}] {total} токенов", flush=True)
            # Экспорт асинхронный: без flush хвост трассы уедет вместе с
            # процессом, если его погасят сразу после ответа.
            observability.flush()
        except Exception as exc:
            # Упавший агент возвращается ТЕКСТОМ, а не исключением. По сети
            # сосед может не ответить по причинам, к работе отношения не
            # имеющим, и для координатора это обычное дело, а не конец света.
            answer = (f"ERROR: агент '{self._role.NAME}' не справился "
                      f"({type(exc).__name__}: {exc}).")
        await event_queue.enqueue_event(new_text_message(answer))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_text_message(f"ERROR: агент '{self._role.NAME}' не умеет отменять "
                             "начатую работу."))


def _payload(result) -> str:
    """Ответ LangGraph -> строка для протокола.

    У структурированных агентов результат лежит в `structured_response`, у
    текстовых — в последнем сообщении. Структура сериализуется в JSON: по
    проводу едет текст, и супервизор разбирает его обратно САМ. Проверять на
    приёмной стороне обязательно — «это точно ResearchPlan» остаётся
    предположением, пока его не проверили.
    """
    structured = result.get("structured_response") if isinstance(result, dict) else None
    if structured is not None and hasattr(structured, "model_dump_json"):
        return structured.model_dump_json()
    messages = result.get("messages", []) if isinstance(result, dict) else []
    return messages[-1].content if messages else ""


def _app(role, port: int) -> Starlette:
    """Starlette + три набора маршрутов A2A.

    Класса `A2AStarletteApplication` из примеров больше нет: в a2a-sdk 1.1.2
    приложение собирают из маршрутов. `fastapi` при этом не нужен —
    `add_a2a_routes_to_fastapi` существует ради `/docs`, а не ради работы.
    """
    card = _card(role, port)
    handler = DefaultRequestHandler(
        agent_executor=LangChainExecutor(role),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return Starlette(routes=[
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, rpc_url="/"),
        *create_rest_routes(handler),
    ])


def serve() -> None:
    """Три сервера в одном процессе, каждый на своём порту.

    Один процесс — уступка удобству запуска, а не отход от протокола: снаружи
    это три независимых агента с тремя карточками и тремя адресами, и разнести
    их по машинам можно, не поменяв ни строки. Задание требует трёх СЕРВЕРОВ, и
    они здесь три; общий у них только интерпретатор.
    """
    servers = []
    for name, (role, port) in ROLES.items():
        cfg = uvicorn.Config(_app(role, port), host=settings.protocol_host,
                             port=port, log_level="warning")
        servers.append((name, port, uvicorn.Server(cfg)))

    for name, port, server in servers:
        threading.Thread(target=server.run, daemon=True).start()
        print(f"A2A {name:11} → http://{settings.protocol_host}:{port}/"
              f"  (карточка: /.well-known/agent-card.json)", flush=True)

    print(f"инструменты по MCP: {SEARCH_MCP_URL}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        threading.Event().wait()


if __name__ == "__main__":
    serve()
