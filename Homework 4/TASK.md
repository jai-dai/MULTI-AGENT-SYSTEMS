# Завдання: Research Agent з власним ReAct Loop

Розширте свого Research Agent з homework-lesson-3 — **замініть `create_react_agent` на власну реалізацію ReAct-циклу** та **покращіть system prompt**, застосувавши техніки промптингу з лекції.

**Мета:** зрозуміти, як працює ReAct loop зсередини — без "магії" фреймворків — та навчитися писати ефективні промпти, які суттєво впливають на поведінку агента.

---

### Що змінюється порівняно з homework-lesson-3

| homework-lesson-3 | homework-lesson-4 |
|---|---|
| `create_react_agent` з LangChain | Власна реалізація ReAct loop |
| LangChain керує циклом tool calling | Ви самі керуєте циклом |
| Фреймворк парсить відповіді LLM | Ви самі обробляєте `tool_calls` з відповіді API |
| `MemorySaver` для памʼяті | Ви самі керуєте списком `messages` |
| `@tool` декоратор LangChain | Tools описані як JSON Schema для API |
| Базовий system prompt | Покращений prompt із застосуванням технік промптингу |

---

### Що потрібно реалізувати

1. **Власний ReAct Loop** — замініть `create_react_agent` на власний цикл, який відправляє повідомлення в LLM API з tool definitions, обробляє відповідь, виконує tool calls, і повторює до фінальної відповіді
2. **Tools як JSON Schema** — опишіть tools (`web_search`, `read_url`, `write_report`) у форматі tool calling API вашого провайдера замість `@tool` декоратора LangChain
3. **Памʼять діалогу** — реалізуйте збереження контексту між запитами без `MemorySaver`
4. **System Prompt** — напишіть осмислений system prompt, що керує поведінкою агента. Експериментуйте з формулюваннями — це і є промпт-інжиніринг на практиці
5. **Логування кроків** — виводьте в консоль, який tool викликається, з якими аргументами, та який результат
6. **Обробка помилок** — tool errors не повинні крашити агента; ліміт ітерацій, щоб агент не зациклився
7. **Покращений System Prompt** — перепишіть system prompt з homework-lesson-3, застосувавши практики промпт-інжинірингу з лекції (чітка роль, структуровані інструкції, приклади, обмеження поведінки тощо).

---

### Очікуваний результат

1. **Працюючий агент** — запускається через `python main.py`, працює в інтерактивному режимі
2. **Власний ReAct loop** — без `create_react_agent`, `AgentExecutor`, або інших агентних абстракцій фреймворків
3. **Tool calling через API** — tools описані як JSON Schema, LLM сам вирішує, коли їх викликати
4. **Логування** — в консолі видно кожен крок: який tool викликано, з якими параметрами, який результат
5. **Multi-step reasoning** — агент робить 3-5+ tool calls на один запит
6. **Памʼять діалогу** — агент памʼятає попередні повідомлення в межах сесії
7. **Звіт** — агент генерує та зберігає Markdown-звіт через `write_report`

Приклад логу в консолі:
```
You: Порівняй naive RAG та sentence-window retrieval

🔧 Tool call: web_search(query="naive RAG approach explained")
📎 Result: Found 5 results...

🔧 Tool call: web_search(query="sentence window retrieval RAG")
📎 Result: Found 5 results...

🔧 Tool call: read_url(url="https://example.com/rag-comparison")
📎 Result: [5000 chars] Article about RAG approaches...

🔧 Tool call: write_report(filename="rag_comparison.md", content="# RAG Comparison...")
📎 Result: Report saved to output/rag_comparison.md

Agent: Звіт збережено у output/rag_comparison.md. Ось основні відмінності: ...
```

---
