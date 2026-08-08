from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables / .env file."""

    api_key: SecretStr = Field(
        validation_alias=AliasChoices("API_KEY", "OPENAI_API_KEY"),
    )
    model_name: str = "gpt-5.2"
    temperature: float = 0.0
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
    # Comma-separated list of directories to ingest. Another corpus is added
    # here, not by editing code: DATA_DIR="data,/Users/me/docs/specs"
    data_dir: str = "data"
    # Directory names never descended into. Hidden directories (.git, .venv)
    # are skipped regardless. A documents folder that also holds a code
    # checkout otherwise contributes hundreds of READMEs and package docs,
    # which then surface in search as confident noise.
    exclude_dirs: str = ".git,.venv,node_modules,__pycache__,site-packages,dist,build"
    index_dir: str = "index"
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
    embed_batch_size: int = 64

    output_dir: str = "output"

    # ReAct loop limits
    max_iterations: int = 10
    request_timeout: int = 60
    model_timeout: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
- If a result is marked "below the relevance threshold", treat it as weak
  evidence and check it against the web before relying on it.

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
