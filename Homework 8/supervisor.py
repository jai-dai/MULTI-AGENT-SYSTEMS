"""Супервизор: три специалиста как инструменты плюс цикл Plan → Research → Critique.

# Почему суб-агент это инструмент

Супервизору не нужен отдельный протокол общения с агентами: у него уже есть
механизм «позвать что-то и получить текст» — вызов инструмента. Planner,
Researcher и Critic подставляются в реестр наравне с `save_report`, и цикл
ReAct не знает, что за именем стоит целый агент. Поэтому и код здесь короткий:
новый уровень иерархии не потребовал нового механизма.

# Почему лимит раундов в коде, а не в промпте

Задание говорит «максимум 2 доработки». Промпт такое соблюдает обычно, но не
всегда, а цена нарушения — бесконечный цикл на живых деньгах. Поэтому модель
решает ЧТО делать, а код гарантирует СКОЛЬКО: после исчерпания лимита
`critique` возвращает вердикт вместе с прямым указанием оформлять отчёт.

Счётчик считает именно REVISE, а не вызовы: критик, дважды сказавший APPROVE,
ничего не израсходовал.
"""
from __future__ import annotations

import json

import prompts
import toolbox
from agents import critic as critic_agent
from agents import planner as planner_agent
from agents import research as research_agent
from agents.react import ReactAgent
from schemas import CritiqueResult, ResearchPlan

MAX_REVISIONS = 2
MAX_STEPS = 16


def render_plan(plan: ResearchPlan) -> str:
    """План — супервизору текстом: он передаёт его исследователю словами."""
    return (
        f"GOAL: {plan.goal}\n"
        f"SEARCH QUERIES:\n" + "\n".join(f"  - {q}" for q in plan.search_queries)
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


class Supervisor:
    """Координатор. Держит суб-агентов, их состояние и счётчик доработок."""

    def __init__(self, interceptor=None) -> None:
        self.planner = planner_agent.build(depth=1)
        # Исследователь ОДИН на весь запрос: во втором раунде он помнит первый.
        self.researcher = research_agent.build(depth=1)
        self.critic = critic_agent.build(depth=1)

        self.revisions = 0
        self.request = ""
        self.plan: ResearchPlan | None = None
        self.last_critique: CritiqueResult | None = None

        extra = {
            "plan": {"func": self._plan, "schema": _schema(
                "plan", "Decompose the user's request into a research plan. "
                "Always call this first.",
                {"request": "The user's request, verbatim"})},
            "research": {"func": self._research, "schema": _schema(
                "research", "Run research. Pass the plan on the first call, and "
                "the critic's revision requests on later ones.",
                {"instructions": "What to research, or what to fix"})},
            "critique": {"func": self._critique, "schema": _schema(
                "critique", "Verify the findings against the sources and get a "
                "verdict of APPROVE or REVISE.",
                {"findings": "The researcher's findings, in full"})},
        }
        registry, schemas = toolbox.for_role("supervisor", extra=extra)
        self.agent = ReactAgent(
            name="supervisor",
            system_prompt=prompts.SUPERVISOR,
            registry=registry,
            schemas=schemas,
            max_steps=MAX_STEPS,
            depth=0,
            interceptor=interceptor,
        )

    # -- суб-агенты как инструменты ------------------------------------- #

    def _plan(self, request: str) -> str:
        print("\n[Supervisor → Planner]")
        self.plan = self.planner.run(request)
        return render_plan(self.plan)

    def _research(self, instructions: str) -> str:
        round_no = self.revisions + 1
        print(f"\n[Supervisor → Researcher]  (раунд {round_no})")
        # Исходный запрос передаётся каждый раз: за раундами доработок легко
        # уехать в то, что просил критик, и потерять то, что просил человек.
        return self.researcher.run(
            f"ORIGINAL USER REQUEST: {self.request}\n\n{instructions}")

    def _critique(self, findings: str) -> str:
        if self.revisions >= MAX_REVISIONS:
            # Лимит уже исчерпан — критиковать больше нечего смысла: ещё один
            # REVISE некому исполнять.
            return (f"Revision limit reached ({MAX_REVISIONS}). No further "
                    "critique will run. Write the report from the findings you "
                    "have and state plainly what stayed unresolved.")
        print("\n[Supervisor → Critic]")
        result = self.critic.run(
            f"ORIGINAL USER REQUEST: {self.request}\n\nFINDINGS:\n{findings}")
        self.last_critique = result
        rendered = render_critique(result)
        if result.verdict == "REVISE":
            self.revisions += 1
            left = MAX_REVISIONS - self.revisions
            rendered += (f"\n\n(revision {self.revisions} of {MAX_REVISIONS}; "
                         + (f"{left} left)" if left else
                            "this was the last one — after the next research "
                            "round, write the report)"))
        return rendered

    # -- вход ------------------------------------------------------------ #

    def run(self, request: str) -> str:
        self.request = request
        self.revisions = 0
        self.plan = None
        self.last_critique = None
        answer = self.agent.run(request)
        print(f"\n📊 всего ~{self.total_tokens} токенов "
              f"(супервизор {self.agent.tokens}, planner {self.planner.tokens}, "
              f"researcher {self.researcher.tokens}, critic {self.critic.tokens}), "
              f"доработок: {self.revisions}")
        return answer

    @property
    def total_tokens(self) -> int:
        return (self.agent.tokens + self.planner.tokens
                + self.researcher.tokens + self.critic.tokens)

    def reset(self) -> None:
        for agent in (self.agent, self.planner, self.researcher, self.critic):
            agent.reset()
        self.revisions = 0


def _schema(name: str, description: str, params: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v}
                               for k, v in params.items()},
                "required": list(params),
            },
        },
    }
