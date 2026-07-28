# Research Agent — власний ReAct Loop

Той самий research-агент, що в homework-lesson-3, але **без агентних абстракцій фреймворку**:
цикл ReAct написаний руками, tools описані як JSON Schema для tool calling API, памʼять
діалогу — звичайний список `messages`, яким керує сам агент.

Текст завдання — у [TASK.md](TASK.md).

## Швидкий старт

Потрібен **Python 3.10+** та **OpenAI API-ключ**.

```bash
cd homework-lesson-4

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # і вписати свій OPENAI_API_KEY

python main.py
```

Команди в REPL: звичайний текст — запит до агента, `reset` — очистити памʼять діалогу,
`exit` — вихід.

### Змінні середовища

| Змінна | Обовʼязкова | За замовчуванням | Опис |
|---|---|---|---|
| `OPENAI_API_KEY` | так | — | ключ OpenAI |
| `MODEL_NAME` | ні | `gpt-4.1-mini` | будь-яка модель з tool calling |
| `MAX_SEARCH_RESULTS` | ні | `5` | результатів на один `web_search` |
| `MAX_URL_CONTENT_LENGTH` | ні | `8000` | ліміт символів на сторінку в `read_url` |
| `MAX_ITERATIONS` | ні | `10` | ліміт ітерацій ReAct-циклу на один запит |
| `OUTPUT_DIR` | ні | `output` | куди зберігати звіти |

## Приклад роботи

```
You: Порівняй naive RAG та sentence-window retrieval

[step 1/10]
🔧 Tool call: web_search(query="naive RAG approach")
📎 Result: [1857 chars] Search results for 'naive RAG approach': 1. The Complete Guide to RAG…
🔧 Tool call: web_search(query="sentence-window retrieval method")
📎 Result: [2572 chars] Search results for 'sentence-window retrieval method': 1. Sentence Window…

[step 2/10]
🔧 Tool call: read_url(url="https://medium.com/@drjulija/what-are-naive-rag-advanced-rag-modular-rag…")
📎 Result: [8160 chars] Content of https://medium.com/@drjulija/…: LLM RAG Paradigms: Naive RAG…
🔧 Tool call: read_url(url="https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/")
📎 Result: [8143 chars] Content of https://glaforge.dev/…: Advanced RAG — Sentence Window Retrieval…
🔧 Tool call: read_url(url="https://medium.com/@govindarajpriyanthan/advanced-rag-building-and-evaluating…")
📎 Result: [8211 chars] Content of https://medium.com/@govindarajpriyanthan/…: Advanced RAG…

[step 3/10]
🔧 Tool call: write_report(filename="naive_RAG_vs_sentence_window_retrieval.md", content="# Порівняння…")
📎 Result: [136 chars] Report saved to .../output/naive_RAG_vs_sentence_window_retrieval.md (3729 characters).

📊 4 step(s), 6 tool call(s), ~22405 tokens

Agent: Звіт з порівняння naive RAG та sentence-window retrieval збережено у файлі…
[memory: 11 messages]
```

Позначення: `🔧` — виклик tool з аргументами, `📎` — результат (`⚠️` перед ним означає
`ERROR:`), `💭` — текст-міркування моделі поруч із викликами, `📊` — підсумок запиту,
`[memory: N messages]` — розмір памʼяті діалогу.

Згенерований звіт з цього прогону: [example_output/report.md](example_output/report.md).

## Архітектура

```
main.py     REPL: команди exit / reset, друк фінальної відповіді
agent.py    ResearchAgent — власний ReAct loop, памʼять, логування, ліміти
tools.py    реалізації tools + TOOL_SCHEMAS (JSON Schema) + dispatch()
config.py   Pydantic Settings (.env) + SYSTEM_PROMPT
```

### ReAct loop без фреймворку

Увесь агент — це цикл у `ResearchAgent.run()` ([agent.py](agent.py)):

```
messages += user
repeat up to MAX_ITERATIONS:
    message = API.chat.completions.create(messages, tools=TOOL_SCHEMAS)
    messages += assistant(message)
    if message.tool_calls is empty:      -> це фінальна відповідь, вихід
    for call in message.tool_calls:      -> виконати, залогувати
        messages += tool(result, tool_call_id=call.id)
ліміт вичерпано -> один виклик без tools, щоб модель відповіла тим, що має
```

Ключові деталі протоколу, які у фреймворку сховані:

- **`tool_call_id`.** Кожен результат tool повертається повідомленням
  `{"role": "tool", "tool_call_id": ..., "content": ...}`. Якщо id не збігається з id
  виклику — API поверне помилку. Саме ця пара «виклик ↔ результат» і тримає цикл.
- **Повідомлення асистента будується явно** (`_assistant_message`), а не серіалізацією
  обʼєкта SDK: у список `messages` кладеться рівно те, що описано в протоколі —
  `role`, `content`, `tool_calls[].function.{name, arguments}`.
- **Паралельні виклики.** Модель може повернути кілька `tool_calls` в одному
  повідомленні — цикл виконує всі й додає стільки ж `tool`-повідомлень.
- **`arguments` — це рядок JSON**, згенерований моделлю. Він може бути невалідним,
  тому парситься в `dispatch()` під `try/except`.

### Памʼять діалогу

Памʼять — це поле `self.messages`, яке живе між запитами: system prompt першим,
далі вся історія `user / assistant / tool`. На кожен виклик API надсилається весь список,
тому агент бачить попередні репліки й розуміє «а тепер порівняй **це** з X».
`reset` очищає історію, залишаючи system prompt. Памʼять у RAM: після виходу зникає.

### Ліміти та обробка помилок

- **Ліміт ітерацій** — `for step in range(1, MAX_ITERATIONS + 1)`. Коли бюджет вичерпано,
  агент робить ще один виклик **без** `tools`, тож модель фізично не може почати новий
  раунд досліджень і мусить відповісти тим, що вже зібрала. У це фінальне повідомлення
  цикл підставляє **факт із коду** — список реально збережених файлів. Без цього модель
  у фінальній відповіді впевнено писала «звіт збережено у файл X», хоча інструментів у неї
  вже не було і файл не створювався.
- **Помилки tools** не піднімаються як винятки: `dispatch()` перетворює на текст `ERROR: ...`
  невідомий tool, невалідний JSON в `arguments`, неправильні імена аргументів і будь-який
  виняток усередині tool. Модель читає це як звичайний результат і адаптується.
- **Помилки API** (401, rate limit, таймаут) — SDK робить 2 ретраї, далі `RuntimeError`,
  який ловить REPL: сесія не падає.

### Context engineering

`read_url` обрізає сторінку до `MAX_URL_CONTENT_LENGTH` і дописує маркер
`[TRUNCATED: showed first N of M characters]`, щоб модель знала про неповноту; сніпети
пошуку обрізаються до 400 символів; імена файлів від моделі проходять через `Path(...).name`,
тому запис завжди лишається всередині `output/`.

### System prompt

Промпт у [config.py](config.py) переписаний відносно homework-lesson-3 за техніками з лекції:

| Техніка | Як застосована |
|---|---|
| Чітка роль і межі | «You are a research agent… not a chatbot that answers from memory» |
| Процедура замість побажань | нумерований `# Workflow` з 7 кроків: decompose → search → select → read → assess → write → summarise |
| Правила вибору інструмента | блок «situation → tool», найчастіша причина хибних викликів |
| Few-shot | розібраний приклад вдалої послідовності викликів для іншої теми |
| Негативні обмеження | «never cite a URL you did not open», «never repeat an identical tool call» |
| Поведінка на збоях | окремі правила на `ERROR:` та `[TRUNCATED ...]` |
| Контракт виводу | шаблон звіту з обовʼязковими секціями |
| Самоперевірка | чекліст «Before calling write_report, check» |

## Tools

| Tool | Призначення |
|---|---|
| `web_search(query, max_results)` | пошук DuckDuckGo (`ddgs`) → title / url / snippet |
| `read_url(url)` | основний текст сторінки (`trafilatura`), обрізаний до ліміту |
| `write_report(filename, content)` | зберігає Markdown-звіт у `output/` |
| `list_reports()` | список збережених звітів |
| `read_report(filename)` | читає збережений звіт, щоб доповнити його в наступних репліках |

Схеми — у `TOOL_SCHEMAS` ([tools.py](tools.py)), реалізації — звичайні функції,
звʼязок між ними — словник `TOOL_REGISTRY`.

## Відмінності від homework-lesson-3

| homework-lesson-3 | homework-lesson-4 |
|---|---|
| `create_agent` з LangChain | власний цикл у `ResearchAgent.run()` |
| `@tool` декоратор генерує схему | `TOOL_SCHEMAS` написані руками |
| `MemorySaver` + `thread_id` | список `self.messages` |
| `ModelCallLimitMiddleware` | `for step in range(...)` + фінальний виклик без tools |
| `ToolErrorMiddleware` | `try/except` у `dispatch()` |
| залежність `langchain`, `langgraph` | тільки `openai` SDK |

## Обмеження

- DuckDuckGo іноді ріже частоту запитів — `web_search` поверне `ERROR: search failed`.
- `read_url` не бере JS-heavy сторінки та PDF — повертає зрозумілу помилку.
- Памʼять у RAM: історія діалогу зникає після виходу з `main.py`.
- Контекст не стискається: у дуже довгій сесії список `messages` зростатиме, доки не
  впреться в контекстне вікно моделі. Наступний крок — сумаризація старих повідомлень.
