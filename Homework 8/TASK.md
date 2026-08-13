# Домашнє завдання: мультиагентна дослідницька система (розширення hw5)

Розширте свого Research Agent з `homework-lesson-5` до **мультиагентної системи** з Supervisor, який координує трьох спеціалізованих суб-агентів за патерном **Plan → Research → Critique**.

---

### Що змінюється порівняно з homework-5

| Було (homework-lesson-5) | Стає (homework-lesson-8) |
|-|-|
| Один Research Agent з 4 інструментами | Supervisor + 3 суб-агенти |
| Агент робить усе одразу | Planner досліджує домен і декомпозує задачу, Researcher виконує, Critic перевіряє |
| Одноразове дослідження | Ітеративне: Critic може повернути Researcher на доопрацювання |
| Без потоку затвердження | HITL: операції запису потребують підтвердження користувача |
| Лише вільний текст | Planner і Critic повертають структурований вивід (Pydantic) |

---

### Архітектура

```
User (REPL)
  │
  ▼
Supervisor Agent
  │
  ├── 1. plan(request)       → Planner Agent      → structured ResearchPlan
  │
  ├── 2. research(plan)      → Research Agent     → [web_search, read_url, knowledge_search]
  │
  ├── 3. critique(findings)  → Critic Agent       → structured CritiqueResult
  │       │
  │       ├── verdict: "APPROVE"  → go to step 4
  │       └── verdict: "REVISE"   → back to step 2 with feedback
  │
  └── 4. write_report(...)   → save_report tool   → HITL gated
```

**Ключовий патерн:** Supervisor оркеструє ітеративний цикл — Critic може відхилити дослідження і повернути його з конкретним зворотним зв'язком. Це патерн **evaluator-optimizer** з Лекції 7.

---

### Що потрібно реалізувати

#### 1. Planner Agent (новий)

Декомпозує запит користувача у структурований план дослідження:

- Використовує параметр `response_format` функції `create_agent` для створення Pydantic-моделі:

```python
from langchain.agents import create_agent

class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web', or both")
    output_format: str = Field(description="What the final report should look like")

planner_agent = create_agent(
    model="...",
    tools=[web_search, knowledge_search],
    system_prompt="...",
    response_format=ResearchPlan,
)
# result["structured_response"] → validated ResearchPlan instance
```

- **Інструменти:** `web_search`, `knowledge_search` — Planner робить попередній пошук, щоб зрозуміти домен перед декомпозицією задачі
- Обгорніть як `@tool`-функцію `plan(request: str)` для Supervisor

#### 2. Research Agent (перевикористання з hw5)

Візьміть свого Research Agent з hw5 і обгорніть як суб-агент:

- **Інструменти:** `web_search`, `read_url`, `knowledge_search` (з hw5)
- Створіть через `create_agent` (з `langchain.agents`), задайте `system_prompt`
- Обгорніть як `@tool`-функцію `research(request: str)` для Supervisor
- RAG-пайплайн (`ingest.py`, `retriever.py`) перевикористовується як є

#### 3. Critic Agent (новий)

Оцінює якість дослідження шляхом **незалежної верифікації** знахідок через ті самі джерела:

- **Інструменти:** `web_search`, `read_url`, `knowledge_search` (ті самі, що й у Research Agent)
- Critic не просто рецензує текст — він може **перевіряти факти**, шукати пропущену інформацію та верифікувати, що джерела підтримують висновки
- Critic оцінює три виміри:
  1. **Freshness** — чи базуються знахідки на актуальних даних? Чи є новіші джерела? Позначає застарілу інформацію
  2. **Completeness** — чи повністю дослідження покриває запит користувача? Чи є непокриті аспекти або пропущені підтеми?
  3. **Structure** — чи добре організовані знахідки, чи логічно структуровані, чи готові стати звітом?
- Використовує параметр `response_format` функції `create_agent` для створення Pydantic-моделі (працює разом з інструментами — агент спочатку викликає інструменти, потім повертає структурований вивід):

```python
class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Is the data up-to-date and based on recent sources?")
    is_complete: bool = Field(description="Does the research fully cover the user's original request?")
    is_well_structured: bool = Field(description="Are findings logically organized and ready for a report?")
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(description="What is missing, outdated, or poorly structured")
    revision_requests: list[str] = Field(description="Specific things to fix if verdict is REVISE")

critic_agent = create_agent(
    model="...",
    tools=[web_search, read_url, knowledge_search],
    system_prompt="...",
    response_format=CritiqueResult,
)
# result["structured_response"] → validated CritiqueResult instance
```

- Обгорніть як `@tool`-функцію `critique(findings: str)` для Supervisor
- System prompt має наголошувати: перевіряти freshness відносно поточної дати, перевіряти покриття відносно оригінального запиту, забезпечити логічну структуру

#### 4. Supervisor Agent

Координатор, що оркеструє цикл Plan → Research → Critique:

- **Інструменти:** `plan`, `research`, `critique`, `save_report` (визначені в `tools.py`, захищені HITL)
- System prompt з правилами координації:
  1. Завжди починати з `plan` для декомпозиції запиту
  2. Викликати `research` з планом
  3. Викликати `critique` для оцінки знахідок
  4. Якщо verdict — `REVISE` — викликати `research` знову зі зворотним зв'язком від Critic (максимум 2 раунди доопрацювання)
  5. Якщо verdict — `APPROVE` — скласти фінальний markdown-звіт і викликати `save_report` для збереження
- Checkpointer: `InMemorySaver` (необхідний для HITL interrupt/resume)

#### 5. HITL на save_report (`main.py`)

`save_report` — це **операція запису** в Supervisor — вона потребує затвердження користувача:

- Використовуйте `HumanInTheLoopMiddleware` (з `langchain.agents.middleware`) для захисту операцій запису:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

supervisor = create_agent(
    model="...",
    tools=[plan, research, critique, save_report],
    system_prompt="...",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"save_report": True}),
    ],
    checkpointer=InMemorySaver(),
)
```

- Стрімте відповіді Supervisor у REPL
- При виникненні interrupt покажіть запропонований звіт (ім'я файлу + превʼю вмісту)
- Приймайте від користувача одну з трьох дій:
  - `approve` — зберегти звіт як є
  - `edit` — користувач вводить свій фідбек (що змінити/доповнити), Supervisor переробляє звіт і знову запитує затвердження
  - `reject` — скасувати збереження повністю
- Відновлюйте граф зі структурованим форматом рішення:

```python
# Approve:
supervisor.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config={"configurable": {"thread_id": thread_id}},
)

# Edit — user provides feedback, Supervisor revises and calls save_report again:
supervisor.invoke(
    Command(resume={"decisions": [{"type": "edit", "edited_action": {"feedback": user_feedback}}]}),
    config={"configurable": {"thread_id": thread_id}},
)

# Reject:
supervisor.invoke(
    Command(resume={"decisions": [{"type": "reject", "message": "reason"}]}),
    config={"configurable": {"thread_id": thread_id}},
)
```

#### 6. Промпти та конфігурація

- System prompts для всіх 4 агентів винесені в `config.py`
- Конфігурація (API-ключі, назва моделі, шляхи, параметри RAG) в `config.py` / `.env`

---

### Структура проєкту

```
homework-lesson-8/
├── main.py              # REPL with HITL interrupt/resume loop
├── supervisor.py        # Supervisor agent + agent-as-tool wrappers
├── agents/
│   ├── __init__.py
│   ├── planner.py       # Planner Agent (uses ResearchPlan from schemas.py)
│   ├── research.py      # Research Agent (reuses hw5 tools)
│   └── critic.py        # Critic Agent (uses CritiqueResult from schemas.py)
├── schemas.py           # Pydantic models: ResearchPlan, CritiqueResult
├── tools.py             # Reused from hw5: web_search, read_url, knowledge_search + save_report
├── retriever.py         # Reused from hw5
├── ingest.py            # Reused from hw5
├── config.py            # Prompts + settings
├── requirements.txt     # Dependencies (add langgraph to hw5 deps)
├── data/                # Documents for RAG (from hw5)
└── .env                 # API keys (do not commit!)
```

---

### Очікуваний результат

1. **Ingestion працює** — `python ingest.py` будує FAISS-індекс (так само як у hw5)
2. **Planner декомпозує** — запит користувача розбивається у структурований `ResearchPlan`
3. **Researcher виконує** — слідує плану, використовує web + knowledge base
4. **Critic оцінює** — повертає структурований `CritiqueResult` з verdict
5. **Ітерація працює** — якщо Critic каже `REVISE`, Researcher повертається з конкретним зворотним зв'язком
6. **HITL працює** — коли Supervisor викликає `save_report`, користувач бачить звіт і затверджує/відхиляє
7. **Звіт збережено** — після затвердження звіт зберігається у `./output/`

Приклад консольного виводу:

```
You: Compare RAG approaches: naive, sentence-window, and parent-child. Write a report.

[Supervisor → Planner]
🔧 plan("Compare RAG approaches: naive, sentence-window, parent-child")
  📎 ResearchPlan(
       goal="Compare three RAG retrieval strategies",
       search_queries=["naive RAG approach", "sentence-window retrieval", "parent-child RAG"],
       sources_to_check=["knowledge_base", "web"],
       output_format="comparison table + pros/cons for each approach"
     )

[Supervisor → Researcher]  (round 1)
🔧 research("Research these topics: 1) naive RAG approach 2) sentence-window ...")
  🔧 knowledge_search("RAG retrieval approaches")
  📎 [3 documents found]
  🔧 web_search("sentence-window vs parent-child RAG retrieval")
  📎 [5 results found]

[Supervisor → Critic]
🔧 critique("Findings: ... [research results] ...")
  🔧 web_search("parent-child chunking RAG 2025 2026")  ← checking freshness
  📎 [3 results — newer approaches exist]
  🔧 web_search("RAG retrieval benchmarks 2026")        ← verifying data is current
  📎 [2 results — research used outdated 2023 benchmarks]
  📎 CritiqueResult(
       verdict="REVISE",
       is_fresh=False,
       is_complete=False,
       is_well_structured=True,
       strengths=["Good coverage of naive and sentence-window", "Well-structured comparison"],
       gaps=["Benchmarks from 2023 — outdated", "Parent-child approach barely covered",
             "Missing recent developments in parent-child chunking"],
       revision_requests=["Find 2025-2026 benchmarks comparing the three approaches",
                          "More detail on parent-child chunking strategy"]
     )

[Supervisor → Researcher]  (round 2)
🔧 research("Find: 1) benchmarks comparing RAG approaches 2) parent-child chunking details")
  🔧 web_search("RAG retrieval benchmarks naive vs sentence-window vs parent-child")
  📎 [4 results found]
  🔧 read_url("https://example.com/rag-benchmarks")
  📎 [3200 chars]

[Supervisor → Critic]
🔧 critique("Updated findings: ... [round 1 + round 2 results] ...")
  🔧 web_search("RAG retrieval accuracy benchmarks 2026")   ← spot-checking updated data
  📎 [2 results — confirms benchmark numbers are current]
  📎 CritiqueResult(
       verdict="APPROVE",
       is_fresh=True,
       is_complete=True,
       is_well_structured=True,
       strengths=["Up-to-date benchmarks", "All three approaches covered in depth",
                  "Clear structure with comparison table"],
       gaps=[],
       revision_requests=[]
     )

[Supervisor → save_report]
🔧 save_report(filename="rag_comparison.md", content="# Comparison of RAG Approaches...")

  ============================================================
  ⏸️  ACTION REQUIRES APPROVAL
  ============================================================
    Tool:  save_report
    Args:  {"filename": "rag_comparison.md", "content": "# Comparison of RAG..."}

  👉 approve / edit / reject: edit
  ✏️  Your feedback: Add a summary table at the top and include latency benchmarks

[Supervisor revises report based on feedback]
🔧 save_report(filename="rag_comparison.md", content="# Comparison of RAG Approaches\n\n| Approach | ...")

  ============================================================
  ⏸️  ACTION REQUIRES APPROVAL
  ============================================================
    Tool:  save_report
    Args:  {"filename": "rag_comparison.md", "content": "# Comparison of RAG..."}

  👉 approve / edit / reject: approve

  ✅ Approved! Report saved to output/rag_comparison.md

Agent: I've completed the research with 2 rounds of investigation. The Critic
       identified gaps in parent-child coverage and benchmarks, which were
       addressed in round 2. After your feedback I added a summary table and
       latency benchmarks. Report saved to output/rag_comparison.md.
```
