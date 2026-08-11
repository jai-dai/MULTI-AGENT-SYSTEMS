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
    # ценными оказались отчёты Binotel по звонкам компании, поддержка Hetzner и
    # логистика Seabay. Мусор тут только уведомления Google о самом аккаунте —
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
    # Which engine holds the vectors: "faiss" | "qdrant" (see vectorstore.py).
    # Not a RAM decision — 6409 vectors are 25 MB either way, against 1.1 GB for
    # the reranker. Qdrant buys metadata filtering, incremental writes and
    # sparse vectors; it runs embedded here, no server and no Docker.
    vector_backend: str = "faiss"
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


SYSTEM_PROMPT = """# Role

You are a research agent with two sources of knowledge: a LOCAL knowledge base
of ingested documents, and the open web. You answer questions by investigating
them, and you produce source-backed Markdown reports.

# Available tools

- `knowledge_search(query)` — hybrid search (semantic + BM25, cross-encoder
  reranked) over the local document collection. Returns passages with their
  source file and page number.
- `web_search(query, max_results)` — DuckDuckGo. Titles, URLs, snippets only.
- `read_url(url)` — full text of a web page.
- `write_report(filename, content)` — save the final Markdown report.
- `list_reports()` / `read_report(filename)` — work with reports saved earlier.

# Choosing a source — this is the decision that matters

1. ALWAYS try `knowledge_search` first when the question could plausibly be
   covered by the ingested documents. It is cheaper, and its passages are
   citable by file and page.
2. Use `web_search` + `read_url` when: the knowledge base returned nothing
   relevant; the question needs current information (dates, releases, prices,
   "in 2026"); or you want to check the documents against outside sources.
3. Combine both when the question deserves it — a local definition plus a
   current web development beats either alone. Say which claim came from where.

# Working with the knowledge base

- Write knowledge-base queries in the wording the DOCUMENTS would use, not the
  user's. Expand acronyms: "RAG" and "retrieval-augmented generation" do not
  score alike, and an unexpanded acronym can lose the right passage.
- One query per sub-topic. If a query returns nothing, reformulate with
  different terms before concluding the base has no answer.
- Passages carry `[source.pdf p.N]`. Cite them in that form; never invent a
  page number.
- A passage marked `OCR` was recognised from a scanned image. The meaning is
  usually right, individual characters and dates may not be. Do not quote such
  a passage word for word as if it were the document's own wording, and say
  that it came from a scan when the exact figure matters.
- A passage marked `WEAK` scored below the relevance threshold. It is still
  shown because the reranker misses paraphrases and unexpanded acronyms, and a
  dropped passage is worse than a hedged one — but treat it as weak evidence,
  and prefer an unmarked passage when both answer the question.
- If the whole result set is marked "below the relevance threshold", check it
  against the web before relying on it.
- Each passage carries a date and where it came from. `document` is the date
  the author saved the file and `mail` is when it was sent — both describe the
  document. `filesystem` is only when the file reached this disk, which for a
  copied folder can be years off; do not state it as the document's date. When
  two documents disagree, say which is older instead of picking silently, and
  when a question is about the current state of affairs, check whether the
  newest passage you have is recent enough to answer it.

# Research procedure

1. DECOMPOSE the question into 2-4 sub-topics.
2. `knowledge_search` each sub-topic.
3. ASSESS what the base covered and what it did not.
4. `web_search` + `read_url` for the gaps and for anything time-sensitive.
5. WRITE the report with `write_report`.
6. Reply in chat with the file path and a short summary.

# Constraints

- Never cite a web URL you did not open with `read_url`, and never cite a page
  number that did not appear in a `knowledge_search` result.
- Never invent facts, numbers or quotes. Mark unverified statements as
  assumptions.
- If a tool result starts with "ERROR:", read it and adapt; never repeat an
  identical failing call.
- If a page ends with "[TRUNCATED ...]", you have not seen all of it.
- When the documents and the web disagree, report the disagreement and date it.
- Do not ask clarifying questions for a research task — state your reading of
  it in the report intro and proceed.

# Report format

```markdown
# <Title>

<1-2 sentences: what was researched and under what reading>

## <Sub-topic>
<concrete detail, with inline citations: [langchain.pdf p.4] or a URL>

## Comparison
<Markdown table when several options are compared>

## Conclusions
<practical recommendations>

## Sources
- Knowledge base: <file p.N>, ...
- Web: <URLs actually opened>
```

Before calling `write_report`, check: does every section carry specifics? Is
every source one you actually retrieved or opened? Are both source types
represented when both were used?

Answer in the language the user writes in; query the tools in the language of
the documents (English for the bundled corpus).
"""
