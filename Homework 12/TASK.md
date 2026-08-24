# Домашнє завдання: Langfuse observability

Підключіть Langfuse до вашої мультиагентної системи з останньої домашньої роботи, налаштуйте tracing та online evaluation через LLM-as-a-Judge.

---

### Що змінюється порівняно з попередніми homework

| Було | Стає |
|---|---|
| Немає observability — система працює як чорна скринька | Кожен запуск трейситься в Langfuse з повним деревом викликів |
| DeepEval тести запускаються локально вручну (hw10) | Langfuse автоматично оцінює нові трейси через LLM-as-a-Judge |
| Промпти захардкоджені в коді | Усі system prompts агентів винесено в Langfuse Prompt Management |

---

### Що потрібно зробити

#### 0. Налаштування Langfuse Cloud

1. Зареєструйтесь на [us.cloud.langfuse.com](https://us.cloud.langfuse.com) (free tier, без credit card)
2. Створіть Organization → Project (наприклад, `homework-12`)
3. **Settings → API Keys → + Create new API keys** — скопіюйте `Public Key` (`pk-lf-...`) та `Secret Key` (`sk-lf-...`)
4. Збережіть ключі у `.env` файл:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

---

#### 1. Підключення tracing до мультиагентної системи

Інтегруйте Langfuse так, щоб **кожен запуск вашої MAS** створював **trace** у Langfuse з повним деревом (усі LLM-виклики, tool calls, суб-агенти — вкладені під один батьківський trace).

Зверніть увагу на:
- `@observe` декоратор та `CallbackHandler` для LangChain/LangGraph — див. [документацію інтеграції](https://langfuse.com/docs/integrations/langchain)
- `propagate_attributes` для прокидання `session_id`, `user_id`, `tags` на весь trace

**Критерій:** зробіть 3-5 запусків з різними запитами. У Langfuse UI → **Tracing → Traces** має бути 3-5 рядків, кожен розгортається у повне дерево з суб-агентами та tool calls.

---

#### 2. Session та User tracking

Переконайтесь, що ваші traces згруповані у **session** і мають `user_id`:

- Після 3-5 запусків перевірте Langfuse UI:
  - **Sessions** tab — має з'явитися ваша сесія з кількома трейсами всередині
  - **Users** tab — має з'явитися ваш user

---

#### 3. Prompt Management

Винесіть **усі system prompts ваших агентів** з коду в Langfuse Prompt Management. Після цього жоден промпт не повинен бути захардкоджений у Python-файлах — код лише завантажує промпти з Langfuse за іменем та label.

##### 3.1. Створіть промпти у Langfuse UI

Для кожного агента у вашій системі:

1. **Prompts → + New prompt**
2. Задайте ім'я, що відповідає ролі агента
3. Вставте текст промпту. Використовуйте template variables (`{{...}}`) де промпт параметризований
4. Додайте label `production`

##### 3.2. Завантажте промпти з коду

Використовуйте `get_prompt(name, label=...)` з Langfuse Python SDK для завантаження промптів, та `.compile(**variables)` для підстановки template variables. Див. [документацію Prompt Management](https://langfuse.com/docs/prompts).

**Критерій:**
- У коді **жодних захардкоджених system prompts** — усі завантажуються з Langfuse
- У Langfuse UI → **Prompts** — видно промпт для кожного агента

---

#### 4. LLM-as-a-Judge: online evaluation у Langfuse

Налаштуйте **автоматичну оцінку** нових трейсів через Langfuse Evaluators.

##### 4.1. Створіть evaluator'и у Langfuse UI

1. Перейдіть: **LLM-as-a-Judge → Evaluators → + Set up evaluator**
2. Створіть **мінімум 2 evaluator'и** з різними score type (numeric, boolean, або categorical)
3. Самостійно продумайте, які аспекти якості найважливіші для вашої конкретної системи — наприклад: relevance відповіді, groundedness фактів, повнота дослідження, структурованість output'у, тощо
4. Напишіть evaluation prompts, використовуючи template variables `{{input}}`, `{{output}}`

Див. [документацію LLM-as-a-Judge](https://langfuse.com/docs/scores/model-based-evals) для доступних score types, template variables та прикладів.

##### 4.2. Запустіть і перевірте

1. Зробіть 3-5 нових запусків вашої системи
2. Зачекайте 1-2 хвилини — Langfuse виконає evaluation асинхронно
3. Перевірте результати:
   - **Tracing → Traces** → відкрийте trace → вкладка **Scores** — має бути автоматично проставлений score від evaluator'а
   - **LLM-as-a-Judge → Evaluators** → статус evaluator'а показує кількість оброблених трейсів

---

### Вимоги

1. **Tracing працює:** кожен запуск MAS → trace з повним деревом суб-агентів і tool calls
2. **Session/User:** traces згруповані в session, мають user_id
3. **Prompt Management:** усі system prompts агентів завантажуються з Langfuse (жодних захардкоджених)
4. **LLM-as-a-Judge:** мінімум 2 evaluator'и налаштовані, автоматично оцінюють нові traces
5. **Скріншоти:** 4 скріншоти з Langfuse UI (trace tree, session, evaluator scores, prompt management)

---

### Що здавати
- Папка `screenshots/` з 4 скріншотами з Langfuse UI
