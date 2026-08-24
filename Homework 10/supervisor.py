"""Супервизор: три специалиста по A2A, запись отчёта по MCP.

# Почему супервизор НЕ является A2A-агентом

Он держит человека в контуре, а подтверждение — действие интерактивное, которого
у HTTP-запроса нет: сервер может вернуть ответ, но не может остановиться и
спросить. Оставить супервизора локальным — не упрощение, а единственное место,
где `input()` вообще имеет смысл.

# Почему суб-агент — это инструмент

У супервизора уже есть механизм «позвать что-то и получить текст» — вызов
инструмента. `delegate_to_planner`, `delegate_to_researcher` и
`delegate_to_critic` встают в список наравне с `save_report`, и модель не знает,
что за одними именами HTTP до A2A, а за другим — HTTP до MCP. Новый уровень
иерархии снова не потребовал нового механизма.

# Почему лимит доработок в коде, а не в промпте

Задание говорит «итеративный цикл». Промпт такое соблюдает обычно, но не всегда,
а цена нарушения — бесконечный цикл на живых деньгах. Модель решает ЧТО делать,
код гарантирует СКОЛЬКО. Счётчик считает именно вердикты REVISE, а не вызовы:
критик, дважды сказавший APPROVE, ничего не израсходовал.

# Что изменилось против версии на ACP

Исчез мост в фоновый поток. Там цикл агента был синхронным, а клиенты протоколов
асинхронными, и их сшивал отдельный файл с собственным циклом событий. LangGraph
асинхронен сам, поэтому инструменты ниже — обычные `async def`, и никакого моста
не нужно. Это тот редкий случай, когда фреймворк убирает не строчку, а файл.
"""
from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

import a2a_utils
import config
from config import settings
from schemas import CritiqueResult, ResearchPlan

MAX_REVISIONS = 2

REPORT_MCP_URL = f"http://{settings.protocol_host}:{settings.report_mcp_port}/mcp"
AGENT_URLS = {
    "planner": f"http://{settings.protocol_host}:{settings.planner_port}",
    "researcher": f"http://{settings.protocol_host}:{settings.researcher_port}",
    "critic": f"http://{settings.protocol_host}:{settings.critic_port}",
}


def render_plan(plan: ResearchPlan) -> str:
    """План — супервизору текстом: он передаёт его исследователю словами."""
    return (
        f"GOAL: {plan.goal}\n"
        "SEARCH QUERIES:\n" + "\n".join(f"  - {q}" for q in plan.search_queries)
        + f"\nSOURCES: {', '.join(plan.sources_to_check)}"
        + f"\nOUTPUT FORMAT: {plan.output_format}"
    )


def render_critique(result: CritiqueResult) -> str:
    lines = [
        f"VERDICT: {result.verdict}",
        f"fresh={result.is_fresh} complete={result.is_complete} "
        f"well_structured={result.is_well_structured}",
    ]
    if result.strengths:
        lines.append("STRENGTHS:\n" + "\n".join(f"  + {s}" for s in result.strengths))
    if result.gaps:
        lines.append("GAPS:\n" + "\n".join(f"  - {g}" for g in result.gaps))
    if result.revision_requests:
        lines.append("REVISION REQUESTS:\n"
                     + "\n".join(f"  {i}. {r}"
                                 for i, r in enumerate(result.revision_requests, 1)))
    return "\n".join(lines)


def _structure(model, raw: str, agent: str):
    """JSON от суб-агента -> модель данных. Либо строка с объяснением для LLM.

    Валидация повторяется на приёмной стороне не из недоверия к своему же
    серверу, а потому что между ними сеть: по проводу приезжает текст, и «это
    точно ResearchPlan» — предположение, пока его не проверили.
    """
    try:
        return model.model_validate_json(raw)
    except Exception as exc:
        return (f"ERROR: агент '{agent}' вернул не {model.__name__} "
                f"({type(exc).__name__}). Позови его ещё раз. Ответ был: {raw[:300]}")


class Supervisor:
    """Координатор: клиент A2A, клиент ReportMCP и счётчик доработок."""

    def __init__(self) -> None:
        self.request = ""
        self.revisions = 0
        self.plan: ResearchPlan | None = None
        self.last_critique: CritiqueResult | None = None
        self.agent = None
        self._checkpointer = InMemorySaver()

    # -- суб-агенты как инструменты -------------------------------------- #

    def _tools(self):
        supervisor = self

        @tool
        async def delegate_to_planner(request: str) -> str:
            """Decompose the user's request into a research plan. Call this first.

            Args:
                request: the user's request, verbatim
            """
            print("\n[Supervisor → A2A → Planner]", flush=True)
            raw = await a2a_utils.ask(AGENT_URLS["planner"], request)
            if raw.startswith("ERROR"):
                return raw
            plan = _structure(ResearchPlan, raw, "planner")
            if isinstance(plan, str):
                return plan
            supervisor.plan = plan
            print(f"  ← план: {len(plan.search_queries)} запросов, "
                  f"источники: {', '.join(plan.sources_to_check)}", flush=True)
            return render_plan(plan)

        @tool
        async def delegate_to_researcher(instructions: str) -> str:
            """Run research. Pass the plan on the first call, and the critic's
            revision requests on later ones.

            Args:
                instructions: what to research, or what to fix
            """
            round_no = supervisor.revisions + 1
            print(f"\n[Supervisor → A2A → Researcher]  (раунд {round_no})", flush=True)
            # Исходный запрос передаётся каждый раз: за раундами доработок легко
            # уехать в то, что просил критик, и потерять то, что просил человек.
            return await a2a_utils.ask(
                AGENT_URLS["researcher"],
                f"ORIGINAL USER REQUEST: {supervisor.request}\n\n{instructions}")

        @tool
        async def delegate_to_critic(findings: str) -> str:
            """Verify the findings against the sources and get APPROVE or REVISE.

            Args:
                findings: the researcher's findings, in full
            """
            if supervisor.revisions >= MAX_REVISIONS:
                # Лимит исчерпан — критиковать больше нет смысла: ещё один
                # REVISE некому исполнять.
                return (f"Revision limit reached ({MAX_REVISIONS}). No further "
                        "critique will run. Write the report from the findings "
                        "you have and state plainly what stayed unresolved.")
            print("\n[Supervisor → A2A → Critic]", flush=True)
            raw = await a2a_utils.ask(
                AGENT_URLS["critic"],
                f"ORIGINAL USER REQUEST: {supervisor.request}\n\nFINDINGS:\n{findings}")
            if raw.startswith("ERROR"):
                return raw
            result = _structure(CritiqueResult, raw, "critic")
            if isinstance(result, str):
                return result

            supervisor.last_critique = result
            print(f"  ← вердикт: {result.verdict}", flush=True)
            rendered = render_critique(result)
            if result.verdict == "REVISE":
                supervisor.revisions += 1
                left = MAX_REVISIONS - supervisor.revisions
                rendered += (f"\n\n(revision {supervisor.revisions} of "
                             f"{MAX_REVISIONS}; "
                             + (f"{left} left)" if left else
                                "this was the last one — after the next research "
                                "round, write the report)"))
            return rendered

        return [delegate_to_planner, delegate_to_researcher, delegate_to_critic]

    # -- сборка ----------------------------------------------------------- #

    async def build(self):
        """Собрать супервизора. `save_report` приезжает из ReportMCP по сети."""
        client = MultiServerMCPClient({
            "report": {"url": REPORT_MCP_URL, "transport": "streamable_http"},
        })
        report_tools = [t for t in await client.get_tools() if t.name == "save_report"]
        if not report_tools:
            raise RuntimeError("ReportMCP не отдаёт save_report — проверь порт 8902")

        self.agent = create_agent(
            model=settings.model_name,
            tools=[*self._tools(), *report_tools],
            system_prompt=config.SUPERVISOR,
            # Человек в контуре стоит ПЕРЕД исполнением инструмента, а не между
            # инструментом и файлом. Поэтому неважно, что за именем `save_report`
            # сетевой вызов к ReportMCP: до сервера дело не доходит, пока человек
            # не сказал «да».
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"save_report": True})],
            # Checkpointer здесь обязателен, и это не деталь реализации:
            # прерывание LangGraph — это ОСТАНОВКА ГРАФА, и чтобы продолжить его
            # после ответа человека, состояние должно где-то лежать. В версии со
            # своим синхронным циклом ничего этого не требовалось: «прерывание»
            # было вызовом `input()` внутри функции, и стек никуда не девался.
            checkpointer=self._checkpointer,
        )
        return self.agent

    def reset(self) -> None:
        self.revisions = 0
        self.plan = None
        self.last_critique = None


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
