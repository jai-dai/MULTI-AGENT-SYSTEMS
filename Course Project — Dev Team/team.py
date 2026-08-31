"""Сборка команды: инструменты по MCP, агенты, граф.

Отдельный модуль, потому что сборка асинхронная (инструменты приезжают по сети),
а нужна она и REPL, и тестам. Тест, который поднимает команду иначе, чем это
делает продукт, проверяет свою сборку, а не продукт.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

import graph as graph_module
from agents.roles import ALL, tools_for
from config import settings

DOCS_URL = f"http://{settings.protocol_host}:{settings.docs_mcp_port}/mcp"
WORKSPACE_URL = f"http://{settings.protocol_host}:{settings.workspace_mcp_port}/mcp"


async def mcp_tools() -> list:
    """Инструменты обоих серверов одним списком.

    Оба сервера отдают всё всем; роли режут этот список при сборке
    (`agents/roles.py`). Так границы ролей стоят в одном месте, а не
    размазываются по конфигурации серверов.
    """
    client = MultiServerMCPClient({
        "docs": {"url": DOCS_URL, "transport": "streamable_http"},
        "workspace": {"url": WORKSPACE_URL, "transport": "streamable_http"},
    })
    return await client.get_tools()


async def build(model: str | None = None, *, checkpointer=None):
    """Готовый граф с тремя исполнителями."""
    available = await mcp_tools()
    model = model or settings.model_name

    agents = {
        role.name: create_agent(
            model=model,
            tools=tools_for(role, available),
            system_prompt=role.prompt,
            # ToolStrategy, а не голая модель данных, ради `handle_errors`.
            #
            # Замерено: на расплывчатом запросе QA вернул вырожденную структуру —
            # литералы 'score???' попали ВНУТРЬ списка issues, а поля
            # `suggestions` и `score` не появились вовсе. С обычным
            # `response_format` это исключение, которое кладёт весь прогон:
            # четыре часа работы графа умирают от одного кривого ответа.
            #
            # `handle_errors=True` возвращает модели текст ошибки валидации как
            # результат вызова инструмента, и она переспрашивает. Ровно тот
            # механизм, что был в собственном цикле предыдущих работ: кривой
            # ответ — это повод уточнить, а не повод упасть.
            response_format=ToolStrategy(role.output, handle_errors=True),
        )
        for role in ALL
    }
    return graph_module.build(agents, checkpointer=checkpointer or InMemorySaver())
