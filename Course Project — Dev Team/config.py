from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables / .env file."""

    # ---- the chat model ----
    # Which wire protocol MODEL_NAME speaks: "auto" | "openai" | "anthropic".
    # "auto" reads it off the model name (claude* → anthropic), so changing
    # provider is a MODEL_NAME + key edit and nothing else. See llm.py.
    llm_backend: str = "auto"
    model_name: str = "gpt-5.2"
    # Optional because each backend has its own key and only needs its own:
    # llm.py raises a clear error if the one it needs is missing.
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )
    temperature: float = 0.0
    # Required by Anthropic, optional for OpenAI. Below ~16k a non-streaming
    # request stays clear of SDK HTTP timeouts.
    max_output_tokens: int = 16000
    # Claude's safety classifiers can decline a request; with this on, the API
    # re-runs it on a fallback model inside the same call instead of returning
    # the refusal. Only applies to the models that support it, and the swap is
    # printed rather than made silently.
    anthropic_fallbacks: bool = True
    # Any OpenAI-compatible chat endpoint: DeepSeek (https://api.deepseek.com),
    # Ollama, vLLM, LM Studio. Independent of the embedding backend — retrieved
    # passages arrive here as text, never as vectors.
    chat_base_url: str | None = None

    # Web tools
    max_search_results: int = 5
    max_url_content_length: int = 8000

    # ---- RAG ----
    # Where vectors come from: "openai" | "local" | "compat".
    #   openai — OpenAI's embedding API; the corpus text leaves the machine
    #   local  — a sentence-transformers model here; nothing leaves
    #   compat — any OpenAI-compatible /v1/embeddings (Ollama, vLLM, …) via
    #            EMBEDDING_BASE_URL
    # Unrelated to MODEL_NAME: retrieved passages reach the chat model as text,
    # never as vectors, so the two are chosen independently.
    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    # Instruction prefixes some models are trained with (e5: "query: " /
    # "passage: "). Omitting them costs accuracy without raising anything.
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    # Asserts that two configurations produce compatible vectors, replacing the
    # automatic signature. Meant for one model reached through two runtimes —
    # e.g. bge-m3 quantised in Ollama and fp32 in sentence-transformers. See
    # embeddings.signature().
    embedding_identity: str = ""
    # Comma-separated list of directories to ingest. Another corpus is added
    # here, not by editing code: DATA_DIR="data,/Users/me/docs/specs"
    data_dir: str = "data"
    # Directory names never descended into. Hidden directories (.git, .venv)
    # are skipped regardless. A documents folder that also holds a code
    # checkout otherwise contributes hundreds of READMEs and package docs,
    # which then surface in search as confident noise.
    exclude_dirs: str = ".git,.venv,node_modules,__pycache__,site-packages,dist,build"
    # Filename patterns never ingested — fnmatch against the NAME, case-insensitive.
    # Two different jobs share one list. `~$*` are Office lock files, created
    # while a document is open; they are not documents and hold no text. The
    # rest is a privacy floor: a folder of real documents contains things that
    # must not become searchable, and a chunk in the index is one retrieval away
    # from being pasted into a prompt for a hosted model. Measured on a real
    # corpus: GitHub recovery codes were indexed and reachable by search.
    # This is a default, not a boundary — extend it per corpus in .env.
    exclude_files: str = "~$*,.env,*.key,*.pem,id_rsa*,*recovery*code*"
    # Stop scanning a spreadsheet after this many consecutive empty rows.
    # Excel declares sheets far larger than their contents; see read_xlsx.
    xlsx_blank_run_limit: int = 2000

    # ---- Mail (IMAP → SQLite → mailprep → index) ----
    imap_host: str = "imap.gmail.com"
    imap_user: str = ""
    # An App Password, NOT the account password: Google disabled the latter for
    # IMAP. It grants mail access only and is revoked separately.
    imap_password: SecretStr | None = None
    # A name starting with a backslash is an IMAP special-use ATTRIBUTE, not a
    # folder name — resolved against the server's own LIST. Gmail translates
    # folder names into the account's language ("[Gmail]/Отправленные"), so a
    # literal name only works on an English account; the attribute always does.
    imap_folders: str = "INBOX,\\Sent"
    imap_since: str = ""              # YYYY-MM-DD; empty = the whole mailbox
    mail_db: str = "mail/mail.db"
    # Вложения поддерживаемых форматов сохраняются сюда и индексируются ОБЫЧНЫМ
    # конвейером документов: DATA_DIR="data,mail/attachments". В деловой почте
    # договор приходит файлом, а в теле письма стоит «у вкладенні».
    mail_attachments_dir: str = "mail/attachments"
    mail_attachment_max_mb: int = 25
    # Адреса, чьи письма не попадают в индекс (fnmatch по email отправителя).
    # Список НАМЕРЕННО короткий. Соблазн отсечь всё «no-reply» велик, но замер
    # по реальному ящику его не подтвердил: из 16 автоматических отправителей
    # ценными оказались отчёты B*****l по звонкам компании, поддержка Hetzner и
    # логистика S****y. Мусор тут только уведомления Google о самом аккаунте —
    # они же единственные, что описывают состояние доступа.
    mail_exclude_senders: str = "*@accounts.google.com"

    # ---- OCR (scanned PDFs) ----
    # off | auto | vision | tesseract. "auto" prefers Apple Vision, which is
    # built into macOS, and falls back to tesseract where that exists.
    ocr_backend: str = "auto"
    ocr_languages: str = "uk,ru,en"
    ocr_dpi: int = 200
    # A page yielding fewer characters than this is treated as un-extracted and
    # sent to OCR. Headers and page numbers alone can reach a few dozen.
    ocr_min_chars: int = 50
    # OCR costs seconds per page; a 300-page scan would stall the whole run.
    ocr_max_pages: int = 40
    tesseract_cmd: str = "tesseract"
    index_dir: str = "index"
    # Индексы, по которым ИЩЕТ агент, через запятую. Пусто — ищем в index_dir.
    # Две настройки, а не одна, потому что запись и чтение здесь несимметричны:
    # `ingest.py` пишет ровно в один индекс, а поиск честно идёт по нескольким.
    # Тела писем и документы лежат раздельно не по недосмотру — у них разные
    # конвейеры и разные схемы чанка, — но вопрос «что мне присылали по А**Н»
    # не знает об этом делении, и знать не должен.
    search_index_dirs: str = ""
    # Куда пишет ИНДЕКСАЦИЯ ПОЧТЫ. Отдельно от `index_dir` по той же причине,
    # по которой почта вообще живёт своим индексом: писать в него нужно, не
    # трогая цель записи документов. Пустая строка выключает докачку при старте.
    mail_index_dir: str = "index_mail"
    # Which engine holds the vectors: "faiss" | "qdrant" (see vectorstore.py).
    # Not a RAM decision — 6409 vectors are 25 MB either way, against 1.1 GB for
    # the reranker. Qdrant buys metadata filtering, incremental writes and
    # sparse vectors; it runs embedded here, no server and no Docker.
    vector_backend: str = "faiss"
    # Адреса локального сервера Qdrant, напр. http://localhost:6333.
    # Порожньо — вбудований режим, як було: бібліотека всередині процесу читає
    # каталог індексу. Сервер це ТЕЖ локальний процес, просто окремий: дані
    # лишаються на диску, назовні не йде нічого. Різниця в тому, що вбудований
    # режим шукає повним перебором і не вміє ні HNSW, ні квантування, ні
    # `on_disk` — вони там приймаються мовчки й не працюють (заміряно).
    qdrant_url: str = ""
    # Ключ, якщо сервер його вимагає. Локальному зазвичай не потрібен.
    qdrant_api_key: SecretStr | None = None
    # Команда підняти сервер, якщо він не відповідає. Порожньо — не піднімати,
    # лише сказати про це. Процес запускається ВІДОКРЕМЛЕНИМ і переживає агента:
    # на сховище ходить не тільки він, а й ingest.py та evaluate.py, і робити
    # базу дочірнім процесом одного з клієнтів — значить гасити її при його
    # виході й ділити порт при наступному запуску.
    qdrant_start_cmd: str = ""
    # Порты серверов протоколов. Инструменты и агенты живут отдельными
    # процессами, и порт — это их адрес.
    # Порты своих серверов. Разведены со всеми домашними работами: делить
    # адрес значит делить систему, и однажды это уже стоило дня отладки.
    docs_mcp_port: int = 8931
    workspace_mcp_port: int = 8932
    # A2A: один сервер = один агент, поэтому портов три, а не один. Это и есть
    # главное отличие от ACP, где три агента жили за одним `Server()` на 8903.
    # Где живут файлы, которые пишет Developer, и где их запускает песочница.
    workspace_dir: str = "workspace"
    # Куда ложатся спецификации — отдельно от кода намеренно. Спецификация это
    # ДОКУМЕНТ решения, а не артефакт сборки: её читают, сверяют и хранят, когда
    # код давно переписан. Смешай их — и `workspace/` придётся чистить между
    # задачами вместе с историей того, что вообще просили сделать.
    specs_dir: str = "specs"
    # Предел итераций QA -> Developer. Живёт в КОДЕ, а не в промпте: промпт
    # соблюдает такое обычно, но не всегда, а цена нарушения — бесконечный цикл
    # на живых деньгах.
    #
    # Имя не `max_iterations` НАМЕРЕННО: такое поле уже есть выше — это предел
    # шагов ReAct-цикла, приехавший по цепочке копий, и в .env он выставлен в 10.
    # Совпади имена — предел ревью молча стал бы десяткой, и заметить это можно
    # было бы только по счёту за токены.
    max_review_iterations: int = 5

    # ---- Langfuse ----
    # Ключей может не быть: без них система работает как работала, просто молча.
    # Телеметрия, без которой продукт не поднимается, — худший вид телеметрии.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    # Лейбл, по которому берутся промпты. Отдельной настройкой, чтобы можно было
    # прогнать систему на черновике промпта, не трогая production.
    langfuse_prompt_label: str = "production"
    # Кто и в какой сессии работает. Задание требует группировки трейсов в
    # session и наличия user_id — это они.
    langfuse_user_id: str = "vl"
    langfuse_session_prefix: str = "repl" 
    # Хост намеренно 127.0.0.1, а не 0.0.0.0: серверы отдают наружу поиск по
    # личному корпусу и запись файлов, и слушать они должны только эту машину.
    protocol_host: str = "127.0.0.1"
    chunk_size: int = 500
    chunk_overlap: int = 100
    # How many candidates each retriever contributes before fusion and rerank.
    retrieval_top_k: int = 10
    # How many chunks survive reranking and reach the model.
    rerank_top_n: int = 3
    reranker_model: str = "BAAI/bge-reranker-base"
    # Off = hybrid search still runs and RRF still orders the results; only the
    # cross-encoder stage is skipped. The escape hatch for a memory-starved
    # host (see preflight.py), at some cost in ranking quality.
    rerank_enabled: bool = True
    # A reranked chunk below this score is noise — UNLESS every candidate is
    # below it, in which case the fusion ranking is kept and the result is
    # marked weak. The reranker misses paraphrases and unexpanded acronyms
    # (measured: "what is RAG?" vs a passage saying only "retrieval-augmented
    # generation" scores 0.0000), so it must not be the only gate between the
    # agent and its own knowledge base.
    rerank_min_score: float = 0.02
    # Переводить ли запрос под язык пассажа перед реранкингом. Стоит одного
    # короткого вызова модели на пару (запрос, алфавит), с кэшем на процесс, и
    # ноль памяти — в отличие от мультиязычного реранкера, который весит 2.2 ГБ
    # против нынешних 1.1. Замер: `bge-reranker-base` на украинский запрос
    # против английской переписки выдаёт по всему пулу 0.0000, тот же вопрос
    # по-английски — 0.8976 на правильном пассаже.
    translate_for_rerank: bool = True
    # Потолок батча по КОЛИЧЕСТВУ и по СУММЕ СИМВОЛОВ; срабатывает тот, что
    # раньше. Символьный предел — главный: он не даёт длинным текстам (письмо
    # целиком) собраться в батч, который локальный сервер эмбеддингов не тянет.
    embed_batch_size: int = 64
    embed_batch_chars: int = 12000

    output_dir: str = "output"

    # ReAct loop limits
    max_iterations: int = 10
    request_timeout: int = 60
    model_timeout: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def imap_folder_list(self) -> list[str]:
        return [f.strip() for f in self.imap_folders.split(",") if f.strip()]

    @property
    def ingest_dirs(self) -> list[str]:
        return [d.strip() for d in self.data_dir.split(",") if d.strip()]


settings = Settings()


# --------------------------------------------------------------------------- #
# Ключи — в окружение процесса
#
# LangChain собирает клиента сам и берёт ключ из ПЕРЕМЕННОЙ ОКРУЖЕНИЯ. Наш .env
# читает pydantic-settings — он кладёт значения в объект `settings`, а не в
# `os.environ`, поэтому LangChain их не видит и падает с «Missing credentials»
# при полностью правильном .env.
#
# Экспорт стоит здесь, а не в точках входа: серверов шесть, и забыть про один из
# них — вопрос времени. Существующее окружение не трогаем: переменная,
# выставленная снаружи, должна побеждать файл.
# --------------------------------------------------------------------------- #

def _export_keys() -> None:
    import os

    for name, value in (
        ("OPENAI_API_KEY", settings.api_key),
        ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
    ):
        if not os.environ.get(name) and value is not None:
            os.environ[name] = value.get_secret_value()


_export_keys()


# `SYSTEM_PROMPT` одноагентной версии удалён вместе с остальными: он приезжал по
# цепочке копий из Agent_1, в мультиагентной системе не используется НИКЕМ, но
# оставался последним захардкоженным system prompt в репозитории. Требование
# задания — «жодних захардкоджених» — про все, а не только про используемые.
