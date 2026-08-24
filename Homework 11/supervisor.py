"""Супервизор: три специалиста по ACP, запись отчёта по MCP.

# Что здесь осталось от hw8 и почему

Цикл Plan → Research → Critique, лимит доработок в коде и разбор вердикта по
полю — всё это переехало дословно. Это и есть содержание урока: протоколы
меняют ДОСТАВКУ, а не поведение. Если бы при переезде на MCP и ACP пришлось
переписать логику координации, значит логика была завязана на то, что все
живут в одном процессе.

Супервизор намеренно НЕ является ACP-агентом. Он держит человека в контуре, а
подтверждение — действие интерактивное, которого у HTTP-запроса нет: сервер
может вернуть ответ, но не может остановиться и спросить. Оставить супервизора
локальным — это не упрощение, а единственное место, где `input()` вообще имеет
смысл.

# Почему суб-агент по-прежнему инструмент

У супервизора уже есть механизм «позвать что-то и получить текст» — вызов
инструмента. `plan`, `research` и `critique` подставляются в реестр наравне с
`save_report`, и цикл ReAct не знает, что за одними именами HTTP до ACP, а за
другим — HTTP до MCP. Новый уровень иерархии снова не потребовал нового
механизма, и это лучший аргумент в пользу того, что механизм выбран верно.

# Почему лимит раундов в коде, а не в промпте

Задание говорит «максимум 2 доработки». Промпт такое соблюдает обычно, но не
всегда, а цена нарушения — бесконечный цикл на живых деньгах. Модель решает
ЧТО делать, код гарантирует СКОЛЬКО. Счётчик считает именно REVISE, а не
вызовы: критик, дважды сказавший APPROVE, ничего не израсходовал.

# Про цену, которую теперь надо собирать руками

В hw8 расход токенов читался из полей живых объектов. Теперь суб-агенты считают
свои токены в чужом процессе и возвращают их частью `stats` в ответе ACP.
Складывается это здесь. Без такой сборки сравнение «мультиагент против
одноагентного» просто перестало бы существовать — а именно оно в hw8 показало
разницу в 19 раз.
"""
from __future__ import annotations

import prompts
from acp_utils import AcpAgents, AcpReply
from agents.react import ReactAgent
from config import settings
from mcp_utils import McpToolset
from schemas import CritiqueResult, ResearchPlan

MAX_REVISIONS = 2
MAX_STEPS = 16

ACP_URL = f"http://{settings.protocol_host}:{settings.acp_port}"
REPORT_MCP_URL = f"http://{settings.protocol_host}:{settings.report_mcp_port}/mcp"


def render_plan(plan: ResearchPlan) -> str:
    """План — супервизору текстом: он передаёт его исследователю словами."""
    if plan.blocked_reason:
        return (f"GOAL: {plan.goal}\n"
                f"BLOCKED — no research will run.\n"
                f"REASON: {plan.blocked_reason}\n"
                # «Answer the user directly» стояло здесь раньше и оказалось
                # двусмысленным: супервизор понял это как «ответь на вопрос» и
                # написал рецепт борща из собственных весов. Блокировка сняла
                # ЦЕНУ (383 366 токенов → 3 696), но не поведение. Причина
                # блокировки — это и ЕСТЬ ответ, а не повод сочинить свой.
                "Relay that reason to the user in their own language, briefly, "
                "and say what they could usefully ask instead. Do NOT answer the "
                "original question yourself from your own knowledge: an answer "
                "with nothing behind it is exactly what this system exists to "
                "avoid. Do not save a report.")
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
    """Координатор: клиент ACP, клиент ReportMCP и счётчик доработок."""

    def __init__(self, interceptor=None) -> None:
        self.acp = AcpAgents(ACP_URL)
        # ReportMCP — единственная операция записи в системе, и она отдельным
        # сервером не ради симметрии: у поиска и у записи разные права и разная
        # цена ошибки. Разные адреса делают эту разницу видимой.
        self.reports = McpToolset(REPORT_MCP_URL, label="ReportMCP",
                                  only=["save_report"])

        self.revisions = 0
        self.request = ""
        self.plan: ResearchPlan | None = None
        self.last_critique: CritiqueResult | None = None
        self.agent_tokens: dict[str, int] = {}

        registry = {
            "plan": self._plan,
            "research": self._research,
            "critique": self._critique,
            **self.reports.registry,
        }
        schemas = [
            _schema("plan", "Decompose the user's request into a research plan. "
                            "Always call this first.",
                    {"request": "The user's request, verbatim"}),
            _schema("research", "Run research. Pass the plan on the first call, "
                                "and the critic's revision requests on later ones.",
                    {"instructions": "What to research, or what to fix"}),
            _schema("critique", "Verify the findings against the sources and get "
                                "a verdict of APPROVE or REVISE.",
                    {"findings": "The researcher's findings, in full"}),
            *_with_approval_note(self.reports.schemas),
        ]

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

    def _delegate(self, agent: str, request: str) -> AcpReply:
        reply = self.acp.call(agent, request)
        self.agent_tokens[agent] = self.agent_tokens.get(agent, 0) + reply.tokens
        if not reply.failed:
            print(f"  ← {agent}: {reply.tokens} токенов, {reply.calls} вызовов",
                  flush=True)
        return reply

    def _plan(self, request: str) -> str:
        print("\n[Supervisor → ACP → Planner]")
        reply = self._delegate("planner", request)
        if reply.failed:
            return reply.text
        plan = _structure(ResearchPlan, reply.text, "planner")
        if isinstance(plan, str):
            return plan
        self.plan = plan
        return render_plan(plan)

    def _blocked(self) -> str | None:
        """Причина, по которой исследовать нечего, либо None.

        # Почему это проверяется кодом, а не остаётся промпту

        Тот же довод, по которому лимит доработок живёт в счётчике: модель решает
        ЧТО делать, код гарантирует СКОЛЬКО. Промпт с «не зови исследователя, план
        заблокирован» соблюдается обычно, но не всегда, а цена нарушения здесь —
        сотни тысяч токенов, ровно те, ради экономии которых ветка и заводилась.

        Заблокировано делегирование, то есть дорогое. `save_report` намеренно НЕ
        запрещён в коде: он стоит копейки, у него уже есть человек в контуре, и
        запрещать запись там, где пользователь может её захотеть, — это чинить
        не ту проблему.
        """
        if self.plan is not None and self.plan.blocked_reason:
            return self.plan.blocked_reason
        return None

    def _research(self, instructions: str) -> str:
        reason = self._blocked()
        if reason:
            return ("The plan is blocked: there is nothing to research. "
                    f"Reason: {reason}\n"
                    "Answer the user directly and briefly — say what cannot be "
                    "done and why, and offer the nearest thing you can do. Do "
                    "not call research or critique again.")
        round_no = self.revisions + 1
        print(f"\n[Supervisor → ACP → Researcher]  (раунд {round_no})")
        # Исходный запрос передаётся каждый раз: за раундами доработок легко
        # уехать в то, что просил критик, и потерять то, что просил человек.
        reply = self._delegate(
            "researcher", f"ORIGINAL USER REQUEST: {self.request}\n\n{instructions}")
        return reply.text

    def _critique(self, findings: str) -> str:
        reason = self._blocked()
        if reason:
            return ("The plan is blocked, so no research happened and there is "
                    f"nothing to critique. Reason: {reason}\n"
                    "Answer the user directly and stop.")
        if self.revisions >= MAX_REVISIONS:
            # Лимит исчерпан — критиковать больше нет смысла: ещё один REVISE
            # некому исполнять.
            return (f"Revision limit reached ({MAX_REVISIONS}). No further "
                    "critique will run. Write the report from the findings you "
                    "have and state plainly what stayed unresolved.")
        print("\n[Supervisor → ACP → Critic]")
        reply = self._delegate(
            "critic", f"ORIGINAL USER REQUEST: {self.request}\n\nFINDINGS:\n{findings}")
        if reply.failed:
            return reply.text
        result = _structure(CritiqueResult, reply.text, "critic")
        if isinstance(result, str):
            return result

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
        self.agent_tokens = {}
        answer = self.agent.run(request)

        breakdown = ", ".join(f"{name} {count}"
                              for name, count in sorted(self.agent_tokens.items()))
        print(f"\n📊 всего ~{self.total_tokens} токенов "
              f"(супервизор {self.agent.tokens}"
              + (f", {breakdown}" if breakdown else "")
              + f"), доработок: {self.revisions}")
        return answer

    @property
    def total_tokens(self) -> int:
        return self.agent.tokens + sum(self.agent_tokens.values())

    def reset(self) -> None:
        """Забыть разговор — и здесь, и на ACP-сервере.

        Новая сессия обязательна: экземпляры суб-агентов живут ТАМ и ключуются
        её идентификатором. Сбросить только супервизора значило бы дать ему
        чистую голову, а исследователю оставить память о прошлом разговоре.
        """
        self.agent.reset()
        self.acp.new_session()
        self.revisions = 0
        self.agent_tokens = {}

    def close(self) -> None:
        self.acp.close()
        self.reports.close()


# -- вспомогательное ------------------------------------------------------ #

def _structure(model, raw: str, agent: str):
    """JSON от суб-агента -> модель данных. Либо строка с объяснением для LLM.

    Валидация повторяется на приёмной стороне не из недоверия к своему же
    серверу, а потому что между ними теперь сеть: по проводу приезжает текст, и
    «это точно ResearchPlan» — предположение, пока его не проверили.
    """
    try:
        return model.model_validate_json(raw)
    except Exception as exc:
        return (f"ERROR: агент '{agent}' вернул не {model.__name__} "
                f"({type(exc).__name__}). Позови его ещё раз. Ответ был: "
                f"{raw[:300]}")


def _with_approval_note(schemas: list[dict]) -> list[dict]:
    """Дописать к описанию save_report, что его увидит человек.

    Сервер об этом не знает и знать не должен (см. шапку `report_mcp.py`):
    подтверждение живёт здесь. Но МОДЕЛЬ знать обязана — иначе она не поймёт,
    почему инструмент вернулся с замечаниями вместо «сохранено», и попробует
    сохранить под другим именем.
    """
    out = []
    for schema in schemas:
        patched = {**schema, "function": dict(schema["function"])}
        if patched["function"]["name"] == "save_report":
            patched["function"]["description"] = (
                patched["function"]["description"].rstrip()
                + " This action is shown to the user for approval before it "
                  "happens; it may come back with the user's feedback instead "
                  "of a confirmation.")
        out.append(patched)
    return out


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
