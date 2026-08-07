# Research Agent + RAG

Подальший розвиток агента: вихідних код взятий з агентів з homework-lesson 3 та 
(власний ReAct-цикл, без агентних абстракцій) плюс локальна база знань на основі 
RAG, що включає змінний ембеддінг, гібридний пошук (semantic + BM25), злиття через 
RRF і cross-encoder reranking. 

Агент сам вирішує, коли шукати в документах, а коли в інтернеті.

Текст завдання — у [TASK.md](TASK.md).

## Швидкий старт

```bash
cd homework-lesson-5

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # і вписати OPENAI_API_KEY

python ingest.py           # PDF з data/ → чанки → ембеддинги → index/
python main.py
```

Команди REPL: текст — запит, `reset` — очистити памʼять, `exit` — вихід.

### Ingestion

```bash
python ingest.py                     # додати нове, не чіпати вже проіндексоване
python ingest.py --rebuild           # перебудувати індекс з нуля
python ingest.py --dirs ~/docs/specs # інший корпус (або DATA_DIR у .env)
```

Індекс **інкрементальний за хешем вмісту**: кожен файл ключується sha256 своїх
байтів, тому повторний запуск після додавання одного документа ембеддить лише
його. Видалений з диска файл прибирається з індексу. Це і є вимога завдання
«перезавантажується без повторного embedding» — і те, що дозволяє нарощувати
великий приватний корпус папку за папкою.

### Змінні середовища

| Змінна | За замовчуванням | Опис |
|---|---|---|
| `OPENAI_API_KEY` | — | обовʼязково |
| `MODEL_NAME` | `gpt-5.2` | будь-яка модель з tool calling |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | модель ембеддингів |
| `DATA_DIR` | `data` | каталоги для ingestion, через кому |
| `INDEX_DIR` | `index` | де лежить індекс |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | розмір чанка і перекриття |
| `RETRIEVAL_TOP_K` | `10` | кандидатів від КОЖНОГО ретривера |
| `RERANK_TOP_N` | `3` | скільки пасажів доживає до моделі |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder |
| `RERANK_MIN_SCORE` | `0.02` | поріг релевантності (див. нижче) |
| `MAX_ITERATIONS` | `10` | ліміт кроків ReAct-циклу |

## Архітектура

```
main.py       REPL
agent.py      власний ReAct-цикл (з homework-lesson-4, без змін)
tools.py      knowledge_search + web_search, read_url, write_report, list/read_report
retriever.py  semantic + BM25 → RRF → cross-encoder rerank
ingest.py     документи → чанки → ембеддинги → FAISS (+ chunks.json для BM25)
config.py     Settings + SYSTEM_PROMPT
index/        index.faiss · chunks.json · manifest.json   (не комітиться)
```

### Чому чотири стадії пошуку, а не одна

Кожна наступна закриває провал попередньої:

1. **Semantic** (FAISS, cosine по `text-embedding-3-small`) знаходить
   перефразування: «how do I split documents» витягує пасаж про chunking.
   Промахується на точних токенах — назвах, кодах помилок, іменах API.
2. **BM25** (`rank_bm25`) знаходить саме ці токени і нічого не знає про сенс.
3. **RRF** зливає два ранжування. Ключове: RRF складає **ранги, а не скори** —
   cosine 0.83 і BM25 11.2 непорівнянні як величини, а «перший» і «третій»
   порівнянні. Формула `Σ 1/(k + rank)`, `k=60` притлумлює голову списку, щоб
   один ретривер не перемагав самою лише впевненістю.
4. **Cross-encoder** читає пару (запит, пасаж) **разом**, а не порівнює два
   незалежно зроблені вектори — і судить релевантність значно краще.

### Реранкер не є єдиним фільтром — і на це є причина

Заміряно на цьому ж стеку:

```
0.9886  «what is retrieval augmented generation?» → «Retrieval-augmented generation…»
0.0000  той самий запит                           → «The weather in Kyiv…»
0.9400  «what is RAG?»                            → «RAG (retrieval-augmented generation)…»
0.0000  «what is RAG?»                            → «Retrieval-augmented generation…»
```

Останній рядок — це не помилка тесту. `bge-reranker-base` **не розкриває
аббревіатуру**: правильний пасаж отримує нуль лише тому, що в ньому немає
літер «RAG». Реранкер, поставлений єдиним воротарем, викинув би вірну
відповідь.

Тому: якщо **жоден** кандидат не перетнув `RERANK_MIN_SCORE`, порядок RRF
зберігається, а результат позначається як слабкий:

```
NOTE: every passage scored BELOW the relevance threshold… Treat these as weak
evidence and verify.
```

Мовчазна порожня відповідь гірша за чесно позначену слабку. Друга половина
відповіді — у system prompt: агенту наказано формулювати запити до бази
словами документів і **розкривати аббревіатури**. У прогоні нижче видно, що
він так і робить.

## Джерело ембеддингів — змінне

Ембеддинги **ніколи не потрапляють у мовну модель**. Вектор потрібен рівно для
одного: вирішити, які абзаци дістати з диска. У модель їде звичайний текст цих
абзаців. Тому модель ембеддингів і модель генерації обираються **незалежно** —
між ними не існує вимоги сумісності.

Три бекенди, перемикаються в `.env`, код не змінюється:

| `EMBEDDING_BACKEND` | Що робить | Що виходить за межі машини |
|---|---|---|
| `openai` | OpenAI Embeddings API (типово) | текст усього корпусу при індексації |
| `local` | sentence-transformers тут же | нічого |
| `compat` | будь-який OpenAI-сумісний `/v1/embeddings` (Ollama, vLLM, LM Studio) | залежить від того, де ендпоінт |

Заміряно на цьому корпусі: `all-MiniLM-L6-v2` локально порахував ті самі 410
чанків за 55 секунд на CPU і знайшов той самий топ-пасаж, що й
`text-embedding-3-small`.

Різниця для приватності конкретна: з `openai` при індексації в хмару їде **весь
корпус**, з `local` — **нічого**, і назовні потрапляють лише ті 2-3 абзаци, які
агент реально процитував у відповіді. Повна ізоляція — це ще й локальна
генерація: `CHAT_BASE_URL` приймає будь-який OpenAI-сумісний адрес, тож
DeepSeek або Ollama підставляються тим самим способом.

### Порядок перемикання

Все живе в `.env`; жоден `.py` чіпати не потрібно.

**1.** Змінити налаштування ембеддингів:

```bash
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=BAAI/bge-m3
```

**2.** Перебудувати індекс — обовʼязково:

```bash
python ingest.py --rebuild
```

**3.** Перевірити, що підпис у `index/manifest.json` оновився:

```json
{"backend": "local", "model": "BAAI/bge-m3", "dim": 1024, …}
```

Для моделей сімейства e5 додатково потрібні префікси, інакше якість тихо падає:

```bash
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_QUERY_PREFIX="query: "
EMBEDDING_PASSAGE_PREFIX="passage: "
```

### Перевірено на живому ендпоінті: Ollama через `compat`

Не «мало б працювати», а працює. Ollama піднята в Docker, модель
`nomic-embed-text`, корпус переіндексований і той самий запит прогнаний
повторно — змінилося **лише джерело ембеддингів**, решта конфігурації
ідентична.

```bash
docker run -d --name jj-ollama -p 11434:11434 -v jj-ollama-models:/root/.ollama ollama/ollama
docker exec jj-ollama ollama pull nomic-embed-text

# .env
EMBEDDING_BACKEND=compat
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_API_KEY=ollama          # фіктивний, Ollama його ігнорує
INDEX_DIR=index_ollama            # окремий індекс: 768 вимірів проти 1536

python ingest.py
```

| | OpenAI | Ollama (`nomic-embed-text`) |
|---|---|---|
| Розмірність | 1536 | **768** |
| Індексація 410 чанків | ~40 с | **342 с** (CPU у контейнері) |
| Прогон агента | 4 кроки, 9 викликів | 4 кроки, 11 викликів, **104 с** |
| Топ-пасаж на контрольний запит | `retrieval-augmented-generation.pdf p.1` | **той самий** |
| Корпус залишає машину | так | **ні** |

Звіт того прогону —
[example_output_2/rag_ollama_embeddings.md](example_output_2/rag_ollama_embeddings.md):
14 КБ, 13 цитат з бази з номерами сторінок, 5 веб-джерел. Порівняйте з
[example_output/rag_kb_vs_web.md](example_output/rag_kb_vs_web.md) — той самий
агент, той самий запит, інша модель ембеддингів.

**Жодного рядка коду не змінено** — тільки `.env` і `ingest.py`.

Швидкість — головний компроміс: 1.2 чанка на секунду на CPU проти ~10 через
API. Для великого приватного корпусу це години, але платяться вони один раз:
індекс інкрементальний, наступні запуски торкаються лише нових файлів.

### Навіщо в manifest.json імʼя моделі й розмірність

Без цього запису зміна `EMBEDDING_MODEL` без `--rebuild` **не викликає жодної
помилки**. Індекс продовжує відповідати — просто сусідів тепер шукають в одному
векторному просторі за координатами з іншого. Пошук стає впевнено неправильним,
і жоден етап конвеєра цього не помітить: релевантність не перевіряє ніхто,
падати нема чому.

Перевірка розмірності теж не рятує: `bge-m3` і `multilingual-e5-large` обидва
дають 1024 виміри, і індекс, побудований одним, «підійде» іншому за формою,
залишаючись беззмістовним за суттю.

Тому `ingest.py` записує в маніфест повний підпис — бекенд, модель, base_url,
префікси, розмірність, — а `retriever.py` звіряє його **на кожному завантаженні
індексу**:

```
RuntimeError: embedding mismatch: the index was built with
text-embedding-3-small (openai) but the current configuration is
text-embedding-3-large (openai) (differs in: model). Vectors from two models
are not comparable — searching would return confident nonsense.
Run `python ingest.py --rebuild` …
```

Той самий запобіжник стоїть і в `ingest.py`: дописати в наявний індекс вектори
іншої моделі не вийде — половини індексу стали б непорівнянними між собою.
Агент бачить цю відмову як звичайний `ERROR:` у результаті інструмента й може
про неї повідомити, а не мовчки цитувати випадкові абзаци.

## Preflight — чи потягне машина цю конфігурацію

Найважчі частини проєкту — не агент, а локальні моделі: cross-encoder-реранкер
(1.1 ГБ) і, з `EMBEDDING_BACKEND=local`, ще й модель ембеддингів. На ноутбуці з
8 ГБ, з яких Docker VM може вже тримати половину, це різниця між «повільно» і
«машина пішла у своп і все стало».

Тому [preflight.py](preflight.py) рахує потребу **до** того, як щось
завантажиться, і зі справжніх вимірів: розміру ваг у кеші HuggingFace,
розмірності з `manifest.json`, того, скільки памʼяті ОС віддає прямо зараз.
Запускається автоматично з `main.py` та `ingest.py`, окремо — `python preflight.py`.

```
✅ preflight: the machine can carry this configuration
   machine: 8.0 GB RAM total, 2.4 GB available now, 111 GB free disk, x86_64 / Darwin
   this run needs about 1.7 GB:
     · reranker BAAI/bge-reranker-base: 1.1 GB
     · embeddings via openai API: 0 GB local
     · torch runtime: 0.35 GB
     · index in memory: 3 MB
     · Docker VM already holding: 3.8 GB (already excluded from 'available')
```

А ось та сама машина з важкою локальною моделлю:

```
⛔ preflight: this configuration does not fit in memory
   this run needs about 4.0 GB:
     · local embedder BAAI/bge-m3: 2.3 GB  (not downloaded yet)
   first run will download 2.3 GB
   ways out:
     → EMBEDDING_BACKEND=openai — embeddings computed by the API, nothing loaded here
     → RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 — 0.09 GB instead of 1.1 GB
     → RERANK_ENABLED=false — skip the cross-encoder; hybrid search and RRF still run
     → stop the Docker VM to give back 3.8 GB
```

Три вердикти: `ok` — запускається; `tight` — запускається з попередженням;
`insufficient` — не стартує мовчки, а показує варіанти. Всі виходи ведуть або
до легшої моделі, або до API (`EMBEDDING_BACKEND=openai`) — тобто відмова
ніколи не означає «нічого не вдієш».

Це **не жорсткий шлагбаум**: `PREFLIGHT_OVERRIDE=1` наполягає на запуску,
`PREFLIGHT=off` вимикає перевірку зовсім. Помилка в оцінці не має робити
проєкт непридатним. У неінтерактивному режимі (пайп, CI) `insufficient` завершує
роботу з кодом 3 замість того, щоб чекати відповіді, якої ніхто не дасть.

> Перша ж версія цієї перевірки була **невірною**: `needed` рахувався в
> гігабайтах, а `available` лишався в байтах, тож `5.0 > 2_684_354_560` завжди
> хибне — і найважча конфігурація отримувала «✅ поміститься». Другий баг:
> розмір моделі подвоювався, бо кеш HuggingFace тримає ваги у `blobs/` і
> симлінкує їх зі `snapshots/`, а обхід рахував і те, і те. Тому перетворення
> одиниць тепер робиться один раз на межі, а симлінки пропускаються.

## Приклад роботи

Запит навмисне вимагає ОБОХ джерел: визначення — з документів, стан справ у
2026 році — з інтернету.

```
You: Що таке RAG і які його компоненти — за документами у базі знань?
     І окремо: які підходи до RAG обговорюються у 2026 році за свіжими
     джерелами в інтернеті? Збережи звіт у rag_kb_vs_web.md

[step 1/10]
🔧 knowledge_search(query="retrieval-augmented generation (RAG) definition components retriever generator")
🔧 knowledge_search(query="RAG architecture components: indexing embedding vector database retriever rerank…")
🔧 knowledge_search(query="LangChain retrieval augmented generation chains retriever document loader split…")
🔧 web_search(query="2026 retrieval-augmented generation approaches agentic RAG graph RAG corrective…")

[step 2/10]
🔧 read_url(url="https://www.turingpost.com/p/ragtypes")
🔧 read_url(url="https://arxiv.org/html/2501.09136v4")
🔧 read_url(url="https://arxiv.org/html/2605.05538v1")
🔧 read_url(url="https://www.edenai.co/post/the-2026-guide-to-retrieval-augmented-generation-rag")

[step 3/10]
🔧 write_report(filename="rag_kb_vs_web.md", content="# RAG: визначення й компоненти…")
📎 Report saved to output/rag_kb_vs_web.md

📊 4 step(s), 9 tool call(s), ~32351 tokens
```

Що тут варто помітити:

- **Порядок джерел заданий не кодом, а промптом.** Агент почав з бази знань і
  пішов у веб лише за тим, чого в документах бути не могло — за 2026 роком.
- **Аббревіатури розкриті самим агентом**: `"retrieval-augmented generation
  (RAG)"`, а не `"RAG"`. Це прямо адресує нульовий скор реранкера, описаний
  вище, — інструкція в промпті спрацювала.
- **Три різні запити до бази** замість одного широкого: визначення, архітектура,
  LangChain-специфіка.

Результат: [example_output/rag_kb_vs_web.md](example_output/rag_kb_vs_web.md) —
13 КБ, 18 цитат з бази з номерами сторінок
(`[retrieval-augmented-generation.pdf p.2]`) і 4 реально прочитані веб-джерела,
розведені по різних секціях звіту.

## Environment notes (Intel Mac)

Версії в `requirements.txt` закріплені **як набір**, і послаблювати їх поодинці
не можна — ланцюг жорсткий:

- `torch 2.2.2` — остання збірка PyTorch для macOS x86_64, новіших немає;
- `transformers >= 4.46` вимагає torch ≥ 2.4 і мовчки вимикає бекенд → 4.44.2;
- `numpy >= 2` ламає torch 2.2.2 → 1.26.4;
- `scipy` і `scikit-learn` свіжих версій зібрані під numpy 2 (`np.long`
  зникло) → 1.13.1 / 1.5.2.

**Порядок імпорту в `retriever.py` — робочий елемент, а не стиль.** `faiss` і
`torch` тягнуть кожен свій OpenMP; якщо першим ініціалізується faiss,
інтерпретатор падає з SIGSEGV на виході (exit code 139 — і разом з ним
губиться весь вивід). Рятує імпорт `torch` перед `faiss`. Загальновідомий
`KMP_DUPLICATE_LIB_OK=TRUE` тут **не допомагає** — перевірено.

На Apple Silicon і на Linux ці обмеження не діють: там доступні свіжі torch і
numpy, і набір можна оновити.

## Обмеження

- Реранкер завантажує ~1.1 ГБ при першому виклику `knowledge_search`.
- `read_url` не бере JS-сторінки та PDF; для arXiv підставляє повнотекстову
  HTML-версію.
- Памʼять діалогу — у RAM, зникає з виходом.
- Чанки ріжуться по символах, а не за семантикою; для довгих таблиць у PDF це
  видно. Наступний крок — semantic chunking.
