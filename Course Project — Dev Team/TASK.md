# Команда розробки ПЗ

## Опис

Мультиагентна система, що симулює AI-команду розробки за патерном Planner–Coder–Reviewer з Лекції 8. На вхід отримує user story, аналізує вимоги, пише код та перевіряє результат через автоматизований рев'ю та тестування.

## Архітектурний патерн

**Основний патерн:** Evaluator-Optimizer (Anthropic) — цикл генерації та оцінки. Developer генерує код, QA оцінює якість зі structured output. Якщо якість недостатня — повертаємось до Developer з feedback. Лінійна частина (User → BA → Developer) — Prompt Chaining з gate (HITL затвердження специфікації).

**Тип взаємодії:** кооперативна — всі агенти працюють на спільну мету (якісний код). Ключовий виклик: синхронізація контексту між ітераціями QA ↔ Developer.

## Агенти та мінімальні інструменти

**Business Analyst** (Planning)
- Роль: отримує задачу від користувача, досліджує контекст, формує структуровану специфікацію
- Інструменти: DuckDuckGo Search (контекст, документація), RAG (пошук по документації мови/фреймворку)

**Developer** (Execution)
- Роль: отримує специфікацію, пише код, створює файли проєкту на диску
- Інструменти: DuckDuckGo Search (бібліотеки, приклади), Python REPL tool (виконання та тестування коду), File System tool (створення файлів проєкту)

**QA Engineer** (Assurance)
- Роль: рев'ює код, запускає його, перевіряє коректність, edge cases, відповідність специфікації
- Інструменти: Python REPL tool (запуск коду та тестів), File System tool (читання файлів, створених Developer)

### Навіщо кожен інструмент (real-world motivation)

1. **DuckDuckGo Search (BA, Developer)**: BA шукає контекст задачі та документацію — якщо user story вимагає інтеграцію з конкретним API, потрібно знайти його специфікацію. Developer шукає приклади використання бібліотек та патерни реалізації.
2. **RAG (BA)**: BA працює з внутрішньою документацією: стандарти кодування, ADR, опис існуючого API, вимоги до безпеки. RAG дає доступ до цієї бази — замість загальних знань LLM агент шукає по актуальній документації проєкту. Аналог: Cursor `@docs`, Copilot з project context.
3. **Python REPL tool (Developer, QA)**: Developer запускає код перед передачею на рев'ю. QA запускає код з тестовими даними та edge cases, перевіряючи реальну поведінку, а не лише читаючи код.
4. **File System tool (Developer, QA)**: Developer працює як AI coding assistant (Claude Code, Devin, Cursor Agent) — створює реальні файли на диску (`src/main.py`, `tests/test_main.py`, `requirements.txt`), формуючи структуру проєкту. QA читає ці файли, щоб перевірити, що вони існують, імпортуються та відповідають специфікації.
5. **LLM structured output (QA)**: QA повертає `ReviewOutput` з вердиктом, списком issues та числовим score — машинно-зчитуваний результат для автоматичного routing (повернути на доопрацювання чи затвердити).

## Structured Output контракти (Pydantic)

Кожен агент має чіткий input/output контракт:

- **SpecOutput:** `title: str, requirements: list[str], acceptance_criteria: list[str], estimated_complexity: Literal["simple","medium","complex"]`
- **CodeOutput:** `source_code: str, description: str, files_created: list[str]`
- **ReviewOutput:** `verdict: Literal["APPROVED","REVISION_NEEDED"], issues: list[str], suggestions: list[str], score: float (0.0–1.0)`

## Підготовка даних для RAG

Студент самостійно готує невелику базу документів:

- Офіційна документація Python stdlib або фреймворку
- Кілька статей/туторіалів з вебу (зберегти як .txt/.md)
- Coding style guide як внутрішній стандарт команди

*Рекомендований обсяг: 10–30 документів.*

**Де брати документи:**
- [Python Standard Library](https://docs.python.org/3/library/) — зберегти потрібні сторінки як .md
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) — покроковий гайд
- [Real Python](https://realpython.com/tutorials/all/) — статті та туторіали
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) — як coding standard для BA

**Альтернатива RAG — документація через MCP:**
Замість наповнення vector store можна підключити документацію бібліотек через MCP-сервер. Наприклад, [Context7 MCP](https://github.com/upstash/context7) дає агентам доступ до актуальної документації бібліотек за запитом. Обидва підходи валідні — RAG (vector store) або MCP (live docs).

## Взаємодія між агентами

*LangGraph: StateGraph з conditional edge (QA → Dev loop). Використовувати Command API для routing від QA назад до Developer.*

| Від | Кому | Що передається (structured output) |
|-----|------|-----------------------------------|
| User | BA | User story (текст) |
| BA | User | SpecOutput на затвердження (Human-in-the-Loop gate) |
| User | BA | Feedback (якщо специфікацію не затверджено) → BA переробляє spec |
| BA | Developer | Затверджена SpecOutput |
| Developer | QA | CodeOutput |
| QA | Developer | ReviewOutput (verdict=REVISION_NEEDED, issues[], score) — макс. 5 ітерацій |
| QA | User | ReviewOutput (verdict=APPROVED) + фінальний код |

## Workflow (LangGraph)

1. **START → BA:** користувач надсилає user story.
2. **BA** досліджує контекст (DuckDuckGo + RAG), формує SpecOutput.
3. **HITL gate:** користувач затверджує специфікацію або повертає з feedback → BA переробляє (цикл до затвердження). Запобігає розробці за неправильними вимогами.
4. **BA → Developer:** передача затвердженої SpecOutput.
5. **Developer** пише код (Python REPL + file write), повертає CodeOutput. ⚠️ LLM-згенерований код потрібно запускати з обмеженнями: timeout, заборонені модулі (os, subprocess, shutil), обмеження на розмір output.
6. **Developer → QA:** передача CodeOutput.
7. **QA** оцінює код, повертає ReviewOutput (`with_structured_output`).
8. **Conditional edge (Command API):** `verdict=REVISION_NEEDED` і `iteration < 5` → Developer з payload (issues + suggestions). Інакше → END.

## Опціональні функції (бонус)

Реалізація будь-яких з наведених пунктів дає додаткові бали. Список не вичерпний — можна запропонувати власні розширення:

**Додаткові інтеграції (MCP-сервери та API)**
- [GitHub MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/github) — створення issue/PR за результатами, читання existing issues як вхідних задач (замість ручного вводу user story).
- [Filesystem MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) — безпечний доступ до файлової системи через стандартизований MCP-протокол (замість прямого file I/O).
- Будь-які інші існуючі MCP-сервери на власний розсуд.

## Моніторинг та тестування (обов'язково)

### Langfuse або LangSmith

Підключити один з сервісів моніторингу до всього pipeline. Кожен виклик LLM (BA, Developer, QA) має логуватися з:
- Input/output кожного агента
- Latency та кількість токенів
- Metadata: назва агента, номер ітерації QA-циклу, session ID

**Навіщо:** Коли QA відхиляє код тричі поспіль, потрібно зрозуміти причину: Developer ігнорує feedback? QA занадто суворий? Специфікація неповна? Без trace-ів це guesswork. Моніторинг також показує, скільки токенів витрачає кожна ітерація — важливо для cost optimization.

### Тести через LLM-as-a-Judge

Написати автоматизовані тести, які оцінюють якість роботи компонентів системи. Кожен тест — це окремий LLM-виклик з критеріями оцінки та тестовими даними.

**Мінімальні тести:**

| Що тестується | Критерій | Приклад тестового сценарію |
|--------------|----------|--------------------------|
| **BA** | Специфікація повна: acceptance criteria тестовані, requirements чіткі | User story: «як користувач, хочу реєстрацію через email» → Judge перевіряє, що spec містить валідацію, edge cases, error handling |
| **Developer** | Код відповідає специфікації: реалізує всі requirements, файли створені | Spec з 3 requirements → Judge перевіряє, що код покриває кожен requirement |
| **QA** | Рев'ю знаходить реальні проблеми, feedback actionable | Подати свідомо поганий код (без error handling, hardcoded values) → Judge перевіряє, що QA виявив проблеми |
| **End-to-end** | Фінальний код працює та відповідає початковій user story | Повний run від user story до approved коду → Judge оцінює якість, відповідність, edge cases |

**Навіщо:** LLM-as-a-Judge — еквівалент unit/integration тестів для LLM-систем. Дозволяє виявляти регресії при зміні промптів — наприклад, після оновлення system prompt BA раптом перестає генерувати acceptance criteria. Без тестів це виявиться лише коли QA-цикл почне зациклюватися.

## Що здавати

- Вихідний код у Git-репозиторії.
- Записане демо роботи системи (відео або GIF).
- README з діаграмою архітектури, інструкцією запуску та прикладами використання.
- Скріншоти або посилання на dashboard Langfuse/LangSmith з прикладами traces.
- Тести (LLM-as-a-Judge) з результатами запуску.
