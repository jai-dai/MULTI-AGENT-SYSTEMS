# Домашнє завдання: MCP + ACP для мультиагентної системи (розширення hw8)

Візьміть мультиагентну систему з `homework-lesson-8` (Supervisor + Planner, Researcher, Critic) і переведіть на архітектуру з протоколами комунікації:

- **MCP** — для інструментів (tools) кожного агента
- **ACP** — для самих агентів (agent-to-agent комунікація)
- **Supervisor** залишається локальним оркестратором, який викликає агентів через ACP

---

### Що змінюється порівняно з homework-8

| Було (homework-lesson-8) | Стає (homework-lesson-9) |
|-|-|
| Tools як Python-функції в одному процесі | Tools виставлені як MCP сервери (FastMCP) |
| Суб-агенти як `@tool`-обгортки для Supervisor | Суб-агенти доступні через ACP сервер (`acp-sdk`) |
| Все працює в одному процесі | Кожен MCP/ACP сервер — окремий HTTP endpoint |
| Прямий виклик функцій | Discovery → Delegate → Collect через протоколи |

---

### Архітектура

```
User (REPL)
  │
  ▼
Supervisor Agent (локальний, create_agent)
  │
  ├── delegate_to_planner(request)      ──► ACP ──► Planner Agent  ──► MCP ──► SearchMCP
  │                                                                             (web_search,
  │                                                                              knowledge_search)
  │
  ├── delegate_to_researcher(plan)      ──► ACP ──► Research Agent ──► MCP ──► SearchMCP
  │                                                                             (web_search,
  │                                                                              read_url,
  │                                                                              knowledge_search)
  │
  ├── delegate_to_critic(findings)      ──► ACP ──► Critic Agent   ──► MCP ──► SearchMCP
  │       │
  │       ├── verdict: "APPROVE" → go to save_report
  │       └── verdict: "REVISE"  → back to researcher with feedback
  │
  └── save_report(...)                  ──► MCP ──► ReportMCP
                                                     (save_report — HITL gated)
```

---

### Що потрібно реалізувати

#### 1. MCP Servers (інструменти)

Створіть MCP сервери для кожного набору інструментів:

| MCP Server | Порт | Tools | Resources |
|:---|:---:|:---|:---|
| **SearchMCP** | 8901 | `web_search`, `read_url`, `knowledge_search` | `resource://knowledge-base-stats` — кількість документів, дата останнього оновлення |
| **ReportMCP** | 8902 | `save_report` | `resource://output-dir` — шлях до директорії та список збережених звітів |

> SearchMCP використовується трьома агентами одночасно — кожен підключається до одного й того ж серверу.

Кожен tool повторює логіку з homework-8 (або homework-5), але тепер обгорнутий як MCP tool через FastMCP. Використовуйте документацію FastMCP та приклади з лекції 9.

#### 2. ACP Server (агенти)

Створіть **один ACP сервер** (порт 8903) з трьома агентами. Кожен агент:

1. Підключається до SearchMCP через `fastmcp.Client`
2. Конвертує MCP tools у LangChain format (`mcp_tools_to_langchain` з лекції 9)
3. Створений через `create_agent` з system prompt з homework-8
4. Повертає `Message(role="agent", ...)`

Planner і Critic використовують `response_format` для структурованого виводу (як у homework-8).

#### 3. Supervisor (оркестратор)

Supervisor **НЕ** є ACP-агентом. Він — локальний `create_agent`, інструменти якого — обгортки над ACP-викликами через `acp_sdk.client.Client`.

`save_report` — окремий MCP-tool (через ReportMCP), захищений HITL як у homework-8.

#### 4. HITL на save_report

Так само як у homework-8 — `HumanInTheLoopMiddleware` на Supervisor.

---

### Структура проєкту

```
homework-lesson-9/
├── main.py              # REPL with HITL interrupt/resume loop
├── supervisor.py        # Supervisor agent + ACP delegation tools
├── acp_server.py        # ACP server with 3 agents (planner, researcher, critic)
├── mcp_servers/
│   ├── search_mcp.py    # SearchMCP: web_search, read_url, knowledge_search
│   └── report_mcp.py    # ReportMCP: save_report
├── agents/
│   ├── __init__.py
│   ├── planner.py       # Planner Agent definition (prompt + response_format)
│   ├── research.py      # Research Agent definition
│   └── critic.py        # Critic Agent definition
├── schemas.py           # Pydantic models: ResearchPlan, CritiqueResult
├── mcp_utils.py         # mcp_tools_to_langchain helper (from lesson 9)
├── config.py            # Prompts + settings + ports
├── retriever.py         # Reused from hw5/hw8
├── ingest.py            # Reused from hw5/hw8
├── requirements.txt
├── data/                # Documents for RAG
└── .env                 # API keys (do not commit!)
```

---

### Порядок запуску

```bash
# 1. Ingest documents for RAG (same as hw5/hw8)
python ingest.py

# 2. Start MCP servers (in separate terminals or as background processes)
python mcp_servers/search_mcp.py   # port 8901
python mcp_servers/report_mcp.py   # port 8902

# 3. Start ACP server
python acp_server.py               # port 8903

# 4. Run supervisor REPL
python main.py
```

---

### Вимоги

- [ ] 2 MCP сервери (SearchMCP, ReportMCP) з tools та resources
- [ ] 1 ACP сервер з 3 агентами (planner, researcher, critic)
- [ ] Кожен ACP агент підключається до SearchMCP через `fastmcp.Client`
- [ ] Кожен ACP агент створений через `create_agent`
- [ ] Supervisor оркеструє агентів через `acp_sdk.client.Client`
- [ ] Ітеративний цикл Plan → Research → Critique працює через ACP
- [ ] HITL на `save_report` через `HumanInTheLoopMiddleware`
- [ ] `save_report` працює через ReportMCP

---

### Очікуваний результат

Така сама поведінка як у homework-8 (Plan → Research → Critique → HITL → Save), але вся комунікація йде через протоколи:

```
You: Compare RAG approaches: naive, sentence-window, and parent-child

[Supervisor → ACP → Planner]
  Planner connects to SearchMCP (MCP) for preliminary search
  Returns: ResearchPlan(goal="...", search_queries=[...], ...)

[Supervisor → ACP → Researcher]  (round 1)
  Researcher connects to SearchMCP (MCP)
  🔧 web_search("naive RAG approach") via MCP
  🔧 knowledge_search("RAG retrieval") via MCP
  Returns findings

[Supervisor → ACP → Critic]
  Critic connects to SearchMCP (MCP) for fact-checking
  🔧 web_search("RAG benchmarks 2026") via MCP
  Returns: CritiqueResult(verdict="REVISE", gaps=["outdated benchmarks", ...])

[Supervisor → ACP → Researcher]  (round 2)
  Researcher re-searches with Critic's feedback via MCP

[Supervisor → ACP → Critic]
  Returns: CritiqueResult(verdict="APPROVE")

[Supervisor → MCP → save_report]
  ⏸️  ACTION REQUIRES APPROVAL
  👉 approve / edit / reject: approve
  ✅ Report saved to output/rag_comparison.md
```
