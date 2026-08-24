# Домашнє завдання: тестування мультиагентної системи (розширення hw8)

Напишіть автоматизовані тести для вашої мультиагентної системи з `homework-lesson-9`, використовуючи DeepEval та підходи з Лекції 10.

---

### Що змінюється порівняно з homework-8

| Було (homework-lesson-8) | Стає (homework-lesson-10)                    |
|-|----------------------------------------------|
| Мультиагентна система без тестів | Та сама система + покриття тестами           |
| Якість перевіряється вручну (vibe check) | Автоматизовані evals з метриками 0–1         |
| Немає golden dataset | 10–15 golden examples для regression testing |
| Немає CI-ready тестів | `deepeval test run` запускає всі тести       |

---

### Що потрібно реалізувати

#### 1. Golden Dataset (10–15 прикладів)

Створіть golden dataset для тестування вашої системи. Кожен приклад — це пара `input` → `expected_output` з категорією:

| Категорія | Кількість | Приклади |
|---|-----------|---|
| **Happy path** | 3–5       | Типові дослідницькі запити, на які система має дати повну відповідь |
| **Edge cases** | 3–5       | Неоднозначні запити, дуже вузькі або дуже широкі теми, запити кількома мовами |
| **Failure cases** | 3–5       | Запити поза доменом, безглузді запити, запити на заборонені теми |

Збережіть як `tests/golden_dataset.json`:

```json
[
  {
    "input": "Compare naive RAG vs sentence-window retrieval",
    "expected_output": "Naive RAG splits documents into fixed-size chunks...",
    "category": "happy_path"
  }
]
```

Можна використати Ragas `TestsetGenerator` для початкової генерації, але **обов'язково зробіть manual review** — виправте або видаліть неякісні приклади.

#### 2. Тести компонентів (component-level)

Протестуйте кожного суб-агента окремо.

**Planner Agent — структурованість плану:**

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)
```

**Critic Agent — якість критики:**

```python
critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)
```

**Research Agent — groundedness відповіді:**

```python
groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-5.4-mini",
    threshold=0.7,
)
```

#### 3. Тести Tool Correctness

Перевірте, що агенти викликають правильні інструменти:

```python
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

# Planner should use web_search and/or knowledge_search for exploration
# Researcher should use web_search, read_url, knowledge_search
# Critic should verify facts via web_search

tool_metric = ToolCorrectnessMetric(threshold=0.5, model="gpt-5.4-mini")
```

Створіть мінімум 3 тест-кейси для tool correctness:
- Planner отримує запит → має викликати пошукові інструменти
- Researcher отримує план → має використати інструменти згідно з `sources_to_check`
- Supervisor отримує APPROVE від Critic → має викликати `save_report`

#### 4. End-to-end тест

Протестуйте повний pipeline Supervisor → Planner → Researcher → Critic:

```python
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-5.4-mini")

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'",
        "Penalize omission of critical details",
        "Different wording of the same concept is acceptable",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-5.4-mini",
    threshold=0.6,
)
```

Запустіть evaluation на повному golden dataset і збережіть результати.

### Структура проєкту

```
homework-lesson-10/
├── tests/
│   ├── golden_dataset.json       # 15-20 golden examples
│   ├── test_planner.py           # Planner agent tests
│   ├── test_researcher.py        # Research agent tests (groundedness)
│   ├── test_critic.py            # Critic agent tests
│   ├── test_tools.py             # Tool correctness tests
│   └── test_e2e.py               # End-to-end evaluation on golden dataset
├── ... (all files from homework-lesson-8)
└── README.md
```

---

### Як запустити тести

```bash
# Run all tests
deepeval test run tests/

# Run specific test file
deepeval test run tests/test_planner.py

# Run with verbose output
deepeval test run tests/ -v
```

---

### Вимоги

1. **Golden Dataset:** 15–20 прикладів (happy path + edge cases + failure cases), збережений як JSON
2. **Component tests:** мінімум по одному тесту на Planner, Researcher, Critic
3. **Tool correctness:** мінімум 3 тест-кейси
4. **End-to-end:** evaluation на повному golden dataset з мінімум 2 метриками
5. **Custom metric:** мінімум 1 GEval метрика під вашу бізнес-логіку
6. **Thresholds:** обґрунтовані пороги (не 0.95 з першого дня — встановіть baseline, потім підвищуйте)
7. **Тести запускаються:** `deepeval test run tests/` проходить без помилок

---

### Очікуваний результат

```
$ deepeval test run tests/

Running 5 test files...

tests/test_planner.py
  ✅ test_plan_quality (Plan Quality: 0.85, threshold: 0.7)
  ✅ test_plan_has_queries (Plan Quality: 0.90, threshold: 0.7)

tests/test_researcher.py
  ✅ test_research_grounded (Groundedness: 0.78, threshold: 0.7)
  ❌ test_research_edge_case (Groundedness: 0.45, threshold: 0.7)

tests/test_critic.py
  ✅ test_critique_approve (Critique Quality: 0.92, threshold: 0.7)
  ✅ test_critique_revise (Critique Quality: 0.88, threshold: 0.7)

tests/test_tools.py
  ✅ test_planner_tools (Tool Correctness: 1.0, threshold: 0.5)
  ✅ test_researcher_tools (Tool Correctness: 1.0, threshold: 0.5)
  ✅ test_supervisor_save (Tool Correctness: 1.0, threshold: 0.5)

tests/test_e2e.py
  ✅ test_golden_dataset [15/20 passed]
     Correctness: avg 0.74, min 0.42, max 0.95
     Answer Relevancy: avg 0.81, min 0.55, max 0.98
     Citation Presence: avg 0.70, min 0.30, max 1.00

======================================================
Overall: 19/20 passed (95.0% pass rate)
```

> Деякі тести можуть fail — це нормально. Мета не 100% pass rate, а мати **baseline** для подальших покращень. Зафіксуйте поточні scores і поступово покращуйте систему.
