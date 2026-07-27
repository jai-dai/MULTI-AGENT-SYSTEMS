# Research Agent

ReAct-агент на LangChain 1.x: отримує питання, сам шукає джерела в інтернеті, читає сторінки,
і зберігає структурований Markdown-звіт у `output/`.

Текст завдання — у [TASK.md](TASK.md).

## Швидкий старт

Потрібен **Python 3.10+** (LangChain 1.x не підтримує 3.9) та **OpenAI API-ключ**.

```bash
cd homework-lesson-3

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # і вписати свій OPENAI_API_KEY

python main.py
```

Або через [uv](https://docs.astral.sh/uv/) (швидше, сам поставить потрібний Python):

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
uv run main.py
```

### Змінні середовища

| Змінна | Обовʼязкова | За замовчуванням | Опис |
|---|---|---|---|
| `OPENAI_API_KEY` | так | — | ключ OpenAI |
| `MODEL_NAME` | ні | `gpt-4.1-mini` | будь-яка модель з tool calling |
| `MAX_SEARCH_RESULTS` | ні | `5` | результатів на один `web_search` |
| `MAX_URL_CONTENT_LENGTH` | ні | `8000` | ліміт символів на сторінку в `read_url` |
| `MAX_ITERATIONS` | ні | `10` | ліміт викликів моделі на один запит |
| `OUTPUT_DIR` | ні | `output` | куди зберігати звіти |

`.env` у `.gitignore` — секрети в репозиторій не потрапляють.

## Приклад роботи

```
You: Порівняй три підходи до побудови RAG: naive, sentence-window та parent-child retrieval

  → web_search(query='RAG naive approach', max_results=3)
  → web_search(query='RAG sentence-window approach', max_results=3)
  → web_search(query='RAG parent-child retrieval approach', max_results=3)
    ✓ Search results for 'RAG parent-child retrieval approach': 1. Parent-Child Chunking in LangChain…
  → read_url(url='https://www.promptingguide.ai/research/rag')
  → read_url(url='https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/')
  → read_url(url='https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain…')
    ✓ Content of https://www.promptingguide.ai/research/rag: Retrieval Augmented Generation (RAG)…
  → write_report(filename='rag_approaches_comparison.md', content='# Порівняння трьох підходів…')
    ✓ Report saved to .../output/rag_approaches_comparison.md (5111 characters).

Agent: Я підготував детальний звіт з порівнянням трьох підходів…
```

Згенерований звіт з цього прогону: [example_output/report.md](example_output/report.md).

У трасі: `→` — виклик tool, `✓` / `✗` — результат tool (успіх / помилка).

## Архітектура

```
main.py     REPL: читає ввід, стрімить кроки агента, друкує ReAct-трасу
agent.py    складання агента: ChatOpenAI + tools + MemorySaver + middleware
tools.py    5 tools з обробкою помилок і обрізанням результатів
config.py   Pydantic Settings (.env) + SYSTEM_PROMPT
```

**Agent loop.** `create_agent` з LangChain 1.x будує граф `model ⇄ tools`: модель сама
вирішує, який tool і з якими аргументами викликати, результат повертається їй у контекст,
цикл повторюється, поки модель не відповість без tool calls. Порядок викликів ніде не
захардкоджений — типовий прогон дає 5-8 tool calls на запит.

**Памʼять.** `MemorySaver` (checkpointer) зберігає історію повідомлень за `thread_id`,
який генерується один раз на запуск процесу (`main.py`). Завдяки цьому працює звʼязний
діалог: «а тепер порівняй **це** з X» агент розуміє з контексту попередніх реплік.

**Ліміт кроків.** `ModelCallLimitMiddleware(run_limit=MAX_ITERATIONS, exit_behavior="end")` —
після N викликів моделі агент коректно завершує відповідь замість нескінченного циклу.
Другий запобіжник — `recursion_limit` у конфізі графа.

**Обробка помилок.** Кожен tool ловить винятки й повертає рядок `ERROR: ...` — модель
бачить причину в контексті й реагує (інше джерело / інший запит), замість того щоб впасти.
`ToolErrorMiddleware` підстраховує на випадок неочікуваних винятків усередині tool-node.

**Context engineering.** `read_url` обрізає сторінку до `MAX_URL_CONTENT_LENGTH` символів і
дописує маркер `[TRUNCATED: showed first N of M characters]`, щоб модель знала, що текст
неповний; сніпети пошуку обрізаються до 400 символів; `web_search` повертає компактний
текстовий блок `title / url / snippet` замість сирого JSON.

## Tools

| Tool | Призначення |
|---|---|
| `web_search(query, max_results)` | пошук DuckDuckGo (`ddgs`) → title / url / snippet |
| `read_url(url)` | основний текст сторінки (`trafilatura`), обрізаний до ліміту |
| `write_report(filename, content)` | зберігає Markdown-звіт у `output/` |
| `list_reports()` | список уже збережених звітів |
| `read_report(filename)` | читає збережений звіт, щоб доповнити або переписати його |

Останні два — додаткові: вони дають агенту змогу в наступних репліках діалогу працювати
з тим, що він уже написав, а не переписувати звіт з нуля.

Імена файлів від моделі проходять через `Path(...).name`, тому запис завжди відбувається
всередині `output/` (спроба `../../evil.md` перетворюється на `output/evil.md`).

## Обмеження

- DuckDuckGo іноді ріже частоту запитів — тоді `web_search` поверне `ERROR: search failed`,
  і агент піде іншим шляхом.
- `read_url` не бере JS-heavy сторінки та PDF — повертає зрозумілу помилку.
- Памʼять живе в RAM (`MemorySaver`): після виходу з `main.py` історія діалогу зникає.
