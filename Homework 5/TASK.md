# Завдання: Research Agent з RAG-системою

Розширте свого Research Agent з `homework-lesson-3` — додайте **RAG-інструмент** з гібридним пошуком та reranking, щоб агент міг шукати не лише в інтернеті, а й у локальній базі знань.

---

### Що змінюється порівняно з попередніми homework

| Було (homework-lesson-3)                        | Стає (homework-lesson-5) |
|-------------------------------------------------|---|
| Tools: `web_search`, `read_url`, `write_report` | + новий tool: `knowledge_search` |
| Агент шукає лише в інтернеті                    | Агент шукає і в інтернеті, і в локальній базі знань |
|                                                 | Є pipeline для завантаження документів у векторну БД |
|                                                 | Hybrid search (semantic + BM25) з cross-encoder reranking |

---

### Що потрібно реалізувати

#### 1. Knowledge Ingestion Pipeline (`ingest.py`)

Скрипт, який завантажує документи у векторну базу даних:

- **Завантаження документів** — PDF файли з директорії `./data/`
- **Chunking** — розбиття на чанки з `RecursiveCharacterTextSplitter` (або semantic chunking — за бажанням)
- **Embeddings** — використайте `text-embedding-3-small` або `text-embedding-3-large`
- **Векторна БД** — оберіть одну з: FAISS (для простоти), Qdrant, Chroma, або pgvector
- **Збереження індексу** — індекс повинен зберігатися на диск і перезавантажуватися без повторного embedding

Скрипт запускається окремо: `python ingest.py` — і створює/оновлює індекс.

#### 2. Hybrid Retrieval з Reranking (`retriever.py`)

Модуль, що реалізує пошук по базі знань:

- **Semantic search** — пошук за cosine similarity у векторній БД
- **BM25 search** — лексичний пошук за ключовими словами
- **Ensemble** — об'єднання результатів semantic + BM25 (наприклад, через `EnsembleRetriever` або вручну)
- **Reranking** — cross-encoder reranker (наприклад, `BAAI/bge-reranker-base`) для фільтрації шуму

#### 3. RAG Tool для агента (`tools.py`)

Новий tool `knowledge_search`, який агент використовує поряд з `web_search`, `read_url`, `write_report`:

```python
def knowledge_search(query: str) -> str:
    """Search the local knowledge base. Use for questions about ingested documents."""
    ...
```

Агент сам вирішує, коли шукати в інтернеті (`web_search`), а коли — в локальній базі (`knowledge_search`).

#### 4. Тестові дані (`data/`)

У `./data/` вже є декілька документів для тестування. За бажанням, додайте ще для перевірки роботи системи з різними типами.

---

### Структура проекту

```
homework-lesson-5/
├── main.py              # Entry point (з homework-lesson-3/4, адаптований)
├── agent.py             # Agent setup з новим knowledge_search tool
├── tools.py             # web_search, read_url, write_report, knowledge_search
├── retriever.py         # Hybrid retrieval + reranking logic
├── ingest.py            # Ingestion pipeline: docs → chunks → embeddings → vector DB
├── config.py            # Settings
├── requirements.txt     # Залежності
├── data/                # Документи для ingestion
│   └── (ваші PDF/TXT файли)
└── .env                 # API ключі (не комітити!)
```

---

### Очікуваний результат

1. **Ingestion працює** — `python ingest.py` завантажує документи з `./data/` та створює індекс
2. **Hybrid search** — пошук використовує і semantic, і BM25, результати об'єднуються
3. **Reranking** — cross-encoder фільтрує нерелевантні результати
4. **Агент використовує RAG** — агент самостійно вирішує, коли шукати в базі знань, а коли в інтернеті
5. **Multi-step reasoning** — агент комбінує результати з різних джерел (web + knowledge base)
6. **Звіт** — агент генерує Markdown-звіт з посиланнями на джерела

Приклад логу в консолі:
```
You: Що таке RAG і які є підходи до retrieval?

🔧 Tool call: knowledge_search(query="RAG retrieval approaches")
📎 Result: [3 documents found]
   - [Page 2] Retrieval-augmented generation combines...
   - [Page 5] Hybrid search approaches include...
   - [Page 3] Dense retrieval using bi-encoders...

🔧 Tool call: web_search(query="RAG retrieval techniques 2026")
📎 Result: Found 5 results...

🔧 Tool call: read_url(url="https://example.com/advanced-rag")
📎 Result: [5000 chars] Latest RAG techniques...

🔧 Tool call: write_report(filename="rag_approaches.md", content="# RAG Approaches...")
📎 Result: Report saved to output/rag_approaches.md

Agent: RAG — це техніка, де...
```