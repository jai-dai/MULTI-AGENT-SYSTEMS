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

#### Де тут власне «ReAct»

Класичний ReAct — цикл **Thought → Action → Observation**, повторюваний до фінальної
відповіді. У ранніх реалізаціях усе це жило в тексті: модель писала
`Action: web_search[query]`, а фреймворк парсив рядок регуляркою. Сьогодні дві третини
протоколу перенесені в API, тому цикл такий короткий:

| ReAct | Де в коді |
|---|---|
| Thought | `message.content` поруч із викликами — друкується як `💭` |
| Action | `message.tool_calls` — структуроване поле відповіді, парсити нічого |
| Observation | повідомлення `{"role": "tool", ...}`, яке цикл дописує в історію |
| Final Answer | відповідь моделі **без** `tool_calls` |

#### Розбір по кроках

**1. Запит потрапляє в памʼять до будь-якої роботи.**
`self.messages` — не лог, а **єдиний стан агента**. Модель по той бік API stateless:
вона не памʼятає ні попередній запит, ні попередній крок циклу. На кожному виклику
надсилається весь список — саме тому агент взагалі щось «памʼятає».

**2. Цикл `for step in range(1, MAX_ITERATIONS + 1)` — і є агентність.**
Зверни увагу, чого тут немає: немає плану й немає зашитого порядку «спершу пошук,
потім читання, потім звіт». Порядок щоразу обирає модель, а код лише дає їй право на
N спроб. `for` замість `while True` — це вже вбудований запобіжник від зациклення.

**3. `tool_choice="auto"`** означає «вирішуй сама: викликати інструмент чи відповісти
текстом». Саме це рішення моделі й рухає цикл далі.

**4. Відповідь кладеться в історію ДО виконання інструментів.**
Це критичний порядок, а не стилістика. Протокол вимагає: спершу повідомлення асистента
з `tool_calls`, потім відповіді на них. Якщо поміняти місцями або зовсім не зберегти
assistant-повідомлення (спокуса є — там часто `content: None`, «порожньо»), наступний
виклик API впаде з 400: у `tool`-повідомлення не буде виклику, на який воно відповідає.

**5. Умова виходу рівно одна — `if not message.tool_calls`.**
Модель перестала кликати інструменти → вважає роботу завершеною. Шукати «Final Answer:»
в тексті не потрібно: вихід визначається структурою відповіді, а не її змістом.

**6. `for call in message.tool_calls` — множина не випадкова.**
Модель може повернути кілька викликів в одному повідомленні. У прогоні вище так і сталося:
два `web_search` на першому кроці й три `read_url` на другому. Тому 6 tool calls вмістилися
в 3 кроки циклу: крок — це не виклик інструмента, а раунд «модель подумала → інструменти
відпрацювали».

**7. `call.function.arguments` — це рядок JSON, згенерований моделлю**, а не обʼєкт.
Він може бути невалідним, містити зайві або хибні ключі — тому парсинг і виклик живуть
у `dispatch()` під `try/except`, а будь-яка поломка повертається текстом `ERROR: ...`.

**8. `tool_call_id` — зчіпка «виклик ↔ результат».**
Якщо id не збігається з тим, що надіслала модель, або на якийсь із викликів не відповісти
взагалі — наступний запит поверне помилку. Правило: **скільки `tool_calls` у повідомленні,
стільки ж `tool`-повідомлень у відповідь**, не більше й не менше.

**9. `content` результату завжди рядок.** Навіть провал інструмента — це не виняток, що
летить угору, а звичайний текст у розмові. У цьому вся ідея обробки помилок в агентах:
помилка стає контекстом, на який модель реагує, а не аварією, що вбиває сесію.

#### Як росла памʼять на реальному прогоні

Запуск «Порівняй naive RAG та sentence-window retrieval» (лог вище) наростив
`self.messages` так:

```
[0]  system     SYSTEM_PROMPT
[1]  user       "Порівняй naive RAG та..."
─ крок 1 ─
[2]  assistant  tool_calls: [web_search, web_search]      ← два виклики разом
[3]  tool       результат web_search  (id збігається з [2].tool_calls[0])
[4]  tool       результат web_search
─ крок 2 ─
[5]  assistant  tool_calls: [read_url, read_url, read_url]
[6]  tool       8160 символів тексту статті
[7]  tool       8143
[8]  tool       8211
─ крок 3 ─
[9]  assistant  tool_calls: [write_report]
[10] tool       "Report saved to .../naive_RAG_vs_sentence_window_retrieval.md"
─ крок 4 ─
[11] assistant  фінальний текст, tool_calls порожній → вихід
```

Звідси `[memory: 11 messages]` у логу (system не рахується). І звідси ж `~22405 tokens`
за один запит: на четвертому кроці в модель їдуть **усі** попередні повідомлення, включно
з трьома статтями по 8 тисяч символів. Контекст росте квадратично — N-й крок пересилає все,
що було на кроках 1..N-1. Тому обрізання в `read_url` — не косметика, а те, що визначає,
чи доживе агент до кінця дослідження.

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

Два інструменти перевіряють не лише вхід, а й **зміст** результату:
`write_report` відхиляє звіт без обовʼязкових секцій, `read_url` позначає сторінки,
які виявились анотацією замість повного тексту. Чому саме так — нижче.

## Відмінності від homework-lesson-3

| homework-lesson-3 | homework-lesson-4 |
|---|---|
| `create_agent` з LangChain | власний цикл у `ResearchAgent.run()` |
| `@tool` декоратор генерує схему | `TOOL_SCHEMAS` написані руками |
| `MemorySaver` + `thread_id` | список `self.messages` |
| `ModelCallLimitMiddleware` | `for step in range(...)` + фінальний виклик без tools |
| `ToolErrorMiddleware` | `try/except` у `dispatch()` |
| залежність `langchain`, `langgraph` | тільки `openai` SDK |

## Що показали реальні прогони — і що з цього виправлено

Розбір справжнього звіту, згенерованого агентом
([example_output/best_practices_rag.md](example_output/best_practices_rag.md), другий
запит у сесії — доповнити вже збережений звіт новим розділом). Механіка спрацювала:
агент згадав контекст, викликав `read_report` і переписав існуючий файл замість створення
нового. Але у вмісті виявились три дефекти — і всі три показові.

**1. Зламана нумерація: секції йдуть `9 → 11 → 10`.** Новий розділ вставлено перед
останнім і пронумеровано навмання. Класичний артефакт «латання» документа: модель
генерує вставку, не перераховуючи сусідів.

**2. Зникла обовʼязкова секція `## Conclusions`.** Вона є в контракті виводу в
[config.py](config.py), але під час переписування просто загубилась — і ніхто цього
не помітив, бо перевіряти не було кому.

**3. Джерела прочитані неглибоко.** Пʼять із шести посилань — сторінки **анотацій**
(`arxiv.org/abs/...`, `aclanthology.org/...`), а не повні тексти. `read_url` на такій
сторінці віддає абстракт, тож у звіті переважають загальні тези замість конкретики.

### Виправлення — у коді, а не в промпті

Перші два дефекти вже були заборонені промптом («звіт мусить містити Conclusions»,
«перевір нумерацію перед збереженням»), і промпт не допоміг. Тому перевірки перенесені
в інструменти, де умова виконується детерміновано:

- **`write_report` перевіряє контракт.** Немає секції Conclusions або Sources (заголовки
  розпізнаються англійською, українською та російською) — виклик відхиляється з
  `ERROR: report is missing required section(s): ...` і вимогою переписати звіт **цілком**.
  Повне перезаписування заодно вбиває причину дефекту №1: модель більше не латає файл.
- **`read_url` бореться з анотаціями.** Для `arxiv.org/abs/<id>` спершу запитується
  повнотекстова HTML-версія `arxiv.org/html/<id>`, з відкатом на `/abs/`, якщо її немає.
  Коли повного тексту не існує, результат позначається `[ABSTRACT ONLY]`; будь-яка
  сторінка коротша за 1200 символів — `[SHORT PAGE]`. Модель бачить маркер і не може
  видати анотацію за прочитану статтю.

### Перевірка

Живий прогон із запитом «best practices для chunking у RAG»
([example_output/chunking_best_practices.md](example_output/chunking_best_practices.md)):

```
[step 3/10]
🔧 Tool call: write_report(filename="chunking_best_practices.md", content="# Best Practices…")
📎 Result: ⚠️ [187 chars] ERROR: report is missing required section(s): Conclusions. Rewrite the FULL report…

[step 4/10]
🔧 Tool call: write_report(filename="chunking_best_practices.md", content="# Best Practices…")
📎 Result: [121 chars] Report saved to .../output/chunking_best_practices.md (5138 characters).

📊 5 step(s), 8 tool call(s), ~34319 tokens
```

Модель самостійно відреагувала на відмову й переписала звіт — секції `1…5` йдуть підряд,
`## 5. Conclusions` і `## Sources` на місці. Для arXiv: `arxiv.org/abs/2407.01219` тепер
повертає повний текст статті (33 713 символів до обрізання) замість короткої анотації,
а стаття 1997 року без HTML-версії коректно відкочується на `/abs/` і позначається
`[ABSTRACT ONLY]`.

### Що лишається за моделлю

Глибина аналізу — вже не про інструменти. `gpt-4.1-mini` схильний до узагальнень
(«обирайте ембеддинги під задачу»), навіть маючи повний текст статті. Це лікується не
кодом, а вибором моделі: вона міняється одним рядком у `.env` (`MODEL_NAME`), решта коду
не змінюється.

## Обмеження

- DuckDuckGo іноді ріже частоту запитів — `web_search` поверне `ERROR: search failed`.
- `read_url` не бере JS-heavy сторінки та PDF — повертає зрозумілу помилку.
- Сторінки-візитки статей (наприклад, `aclanthology.org`) віддають анотацію + BibTeX;
  повний текст там лише у PDF, який `trafilatura` не парсить.
- Памʼять у RAM: історія діалогу зникає після виходу з `main.py`.
- Контекст не стискається: у дуже довгій сесії список `messages` зростатиме, доки не
  впреться в контекстне вікно моделі. Наступний крок — сумаризація старих повідомлень.
