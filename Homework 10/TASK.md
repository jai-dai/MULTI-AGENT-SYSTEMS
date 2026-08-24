# Домашнє завдання: MCP + A2A для мультиагентної системи (розширення hw9)

Візьміть мультиагентну систему з `homework-lesson-9` (Supervisor + Planner, Researcher, Critic) і переведіть на архітектуру з протоколами комунікації:

- **MCP** — для інструментів (tools) кожного агента
- **A2A** — для самих агентів (agent-to-agent комунікація)
- **Supervisor** залишається локальним оркестратором, який викликає агентів через A2A

---

### Що змінюється порівняно з homework-9

| Було (homework-lesson-9) | Стає (homework-lesson-10) |
|-|-|
| Tools як Python-функції в одному процесі | Tools виставлені як MCP сервери (FastMCP) |
| Суб-агенти як `@tool`-обгортки для Supervisor | Кожен суб-агент — окремий A2A сервер зі своєю Agent Card |
| Все працює в одному процесі | Кожен MCP/A2A сервер — окремий HTTP endpoint |
| Прямий виклик функцій | Discovery → Delegate → Collect через протоколи |

---

### Архітектура

```
User (REPL)
  │
  ▼
Supervisor Agent (локальний, create_agent)
  │
  ├── delegate_to_planner(request)      ──► A2A ──► Planner Agent  ──► MCP ──► SearchMCP
  │                                                                             (web_search,
  │                                                                              knowledge_search)
  │
  ├── delegate_to_researcher(plan)      ──► A2A ──► Research Agent ──► MCP ──► SearchMCP
  │                                                                             (web_search,
  │                                                                              read_url,
  │                                                                              knowledge_search)
  │
  ├── delegate_to_critic(findings)      ──► A2A ──► Critic Agent   ──► MCP ──► SearchMCP
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

Кожен tool повторює логіку з homework-9 (або homework-5), але тепер обгорнутий як MCP tool через FastMCP. Використовуйте документацію FastMCP та приклади з лекції 10.

#### 2. A2A Servers (агенти)

Створіть **три A2A сервери** — по одному на агента (в A2A один сервер = один агент зі своєю Agent Card):

| A2A Agent | Порт |
|:---|:---:|
| Planner | 8903 |
| Researcher | 8904 |
| Critic | 8905 |

Кожен агент:

1. Отримує інструменти з SearchMCP через `langchain-mcp-adapters` (`MultiServerMCPClient.get_tools()`)
2. Створений через `create_agent` з system prompt з homework-9
3. Загорнутий в `AgentExecutor`, який кладе відповідь у чергу подій через `new_text_message` (патерн з лекції 10)
4. Публікує Agent Card з назвою, описом і skills за адресою `/.well-known/agent-card.json`

Planner і Critic використовують `response_format` для структурованого виводу (як у homework-9).

#### 3. Supervisor (оркестратор)

Supervisor **НЕ** є A2A-агентом. Він — локальний `create_agent`, інструменти якого — обгортки над A2A-викликами через `a2a.client.create_client` + `send_message`.

`save_report` — окремий MCP-tool (через ReportMCP), захищений HITL як у homework-9.

#### 4. HITL на save_report

Так само як у homework-9 — `HumanInTheLoopMiddleware` на Supervisor.

---

### Структура проєкту

```
homework-lesson-10/
├── main.py              # REPL with HITL interrupt/resume loop
├── supervisor.py        # Supervisor agent + A2A delegation tools
├── a2a_servers.py       # Three A2A agent servers (planner, researcher, critic)
├── mcp_servers/
│   ├── search_mcp.py    # SearchMCP: web_search, read_url, knowledge_search
│   └── report_mcp.py    # ReportMCP: save_report
├── agents/
│   ├── __init__.py
│   ├── planner.py       # Planner Agent definition (prompt + response_format)
│   ├── research.py      # Research Agent definition
│   └── critic.py        # Critic Agent definition
├── schemas.py           # Pydantic models: ResearchPlan, CritiqueResult
├── config.py            # Prompts + settings + ports
├── retriever.py         # Reused from hw5/hw9
├── ingest.py            # Reused from hw5/hw9
├── requirements.txt
├── data/                # Documents for RAG
└── .env                 # API keys (do not commit!)
```

---

### Порядок запуску

```bash
# 1. Ingest documents for RAG (same as hw5/hw9)
python ingest.py

# 2. Start MCP servers (in separate terminals or as background processes)
python mcp_servers/search_mcp.py   # port 8901
python mcp_servers/report_mcp.py   # port 8902

# 3. Start A2A agent servers
python a2a_servers.py              # ports 8903, 8904, 8905

# 4. Run supervisor REPL
python main.py
```

---

### Вимоги

- [ ] 2 MCP сервери (SearchMCP, ReportMCP) з tools та resources
- [ ] 3 A2A сервери (planner, researcher, critic), кожен зі своєю Agent Card
- [ ] Кожен A2A агент отримує інструменти з SearchMCP через `langchain-mcp-adapters`
- [ ] Кожен A2A агент створений через `create_agent`
- [ ] Supervisor оркеструє агентів через `a2a.client.create_client`
- [ ] Ітеративний цикл Plan → Research → Critique працює через A2A
- [ ] HITL на `save_report` через `HumanInTheLoopMiddleware`
- [ ] `save_report` працює через ReportMCP

---

### Очікуваний результат

Така сама поведінка як у homework-9 (Plan → Research → Critique → HITL → Save), але вся комунікація йде через протоколи:

```
You: Compare RAG approaches: naive, sentence-window, and parent-child

[Supervisor → A2A → Planner]
  Planner connects to SearchMCP (MCP) for preliminary search
  Returns: ResearchPlan(goal="...", search_queries=[...], ...)

[Supervisor → A2A → Researcher]  (round 1)
  Researcher connects to SearchMCP (MCP)
  🔧 web_search("naive RAG approach") via MCP
  🔧 knowledge_search("RAG retrieval") via MCP
  Returns findings

[Supervisor → A2A → Critic]
  Critic connects to SearchMCP (MCP) for fact-checking
  🔧 web_search("RAG benchmarks 2026") via MCP
  Returns: CritiqueResult(verdict="REVISE", gaps=["outdated benchmarks", ...])

[Supervisor → A2A → Researcher]  (round 2)
  Researcher re-searches with Critic's feedback via MCP

[Supervisor → A2A → Critic]
  Returns: CritiqueResult(verdict="APPROVE")

[Supervisor → MCP → save_report]
  ⏸️  ACTION REQUIRES APPROVAL
  👉 approve / edit / reject: approve
  ✅ Report saved to output/rag_comparison.md
```
