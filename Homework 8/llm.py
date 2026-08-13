"""The single place where a conversation becomes a provider request.

`agent.py` owns the ReAct loop; this module owns the wire format. The split is
the same one already made for embeddings (`embeddings.py`) and OCR (`ocr.py`):
the vendor-specific part lives behind one interface, and everything above it
stays provider-neutral.

Two backends:

    openai      /v1/chat/completions — OpenAI itself, and every endpoint that
                speaks its protocol: DeepSeek, Ollama, vLLM, LM Studio, Groq,
                Together, OpenRouter, Mistral, xAI (set CHAT_BASE_URL)
    anthropic   /v1/messages via the official `anthropic` SDK — a genuinely
                different protocol, not a compatibility shim

Which one is used is derived from MODEL_NAME unless LLM_BACKEND says otherwise,
so switching provider really is a .env edit:

    MODEL_NAME=claude-opus-5
    ANTHROPIC_API_KEY=sk-ant-...

# The canonical message format is OpenAI's

`agent.messages` is a list of plain dicts in OpenAI's chat shape (`role`,
`content`, `tool_calls`, `tool_call_id`). That is a deliberate choice, not an
oversight: it is the format the agent was written against, the one most
endpoints already accept, and it keeps the OpenAI backend a pass-through. The
cost is that the abstraction leans toward one vendor — so the rule for anything
added later is that a field only enters the canonical format if BOTH backends
can carry it. Anything provider-native travels under a `_`-prefixed key that
only its own backend reads (see `_blocks` below).

# What Anthropic needs that OpenAI does not

- `system` is a top-level parameter, not a message. All system messages are
  hoisted out and joined, which moves a mid-conversation instruction to the
  front. Claude Opus 5 / 4.8 would accept `role: "system"` inline, but Sonnet 5
  rejects it — hoisting works on every model, and for instructions the position
  matters less than the presence.
- A tool result is a content block inside a USER message, and all results
  answering one assistant turn must sit in the SAME message. Consecutive `tool`
  entries are therefore merged.
- `max_tokens` is required (MAX_OUTPUT_TOKENS).
- `temperature` is rejected outright by Opus 5 / 4.8 / Sonnet 5, so this
  backend never sends it.
- `tool_choice` для «обязательно вызови вот этот инструмент» пишется иначе:
  `{"type": "tool", "name": …}` против `{"type": "function", "function": {…}}`
  у OpenAI. Параметр `require_tool` — это и есть структурированный вывод:
  инструмент, чья схема равна нужной модели данных, и запрет отвечать текстом.
  Отдельного `response_format` не нужно, и он всё равно есть не у всех
  провайдеров, а инструменты есть у обоих.
- Thinking is on by default on Opus 5. The blocks come back with empty text
  (raw reasoning is never returned) but must be replayed VERBATIM on the next
  call — dropping them can trigger ordering/signature errors. So the assistant
  turn keeps the provider's own content blocks under `_blocks` and replays
  those instead of reconstructing them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from config import settings

# Server-side refusal fallbacks exist only for these models (Claude API only).
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5", "claude-mythos-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Any failure to obtain an answer, whatever the provider called it."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON text; tools.dispatch() parses it


@dataclass
class Reply:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0
    blocks: Any = None      # provider-native content, kept for replay


def _secret(*candidates, hint: str) -> str:
    for candidate in candidates:
        if candidate:
            return candidate.get_secret_value()
    raise LLMError(f"no API key for this backend — set {hint} in .env")


# --------------------------------------------------------------------------- #
# OpenAI protocol
# --------------------------------------------------------------------------- #


class OpenAIBackend:
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=_secret(settings.api_key, hint="API_KEY (or OPENAI_API_KEY)"),
            base_url=settings.chat_base_url or None,
            timeout=settings.model_timeout,
            max_retries=2,
        )

    def complete(self, messages: list[dict], tools: list[dict] | None,
                 require_tool: str | None = None) -> Reply:
        kwargs: dict[str, Any] = {"model": settings.model_name, "messages": messages}
        # Reasoning models (gpt-5*, o*) accept only the default temperature and
        # reject temperature=0 with a 400. This is a heuristic over OpenAI's own
        # naming, which is exactly why it belongs here and not in the agent:
        # a model called `deepseek-chat` or `llama3` matches nothing and gets
        # the temperature, which is correct.
        if not settings.model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["temperature"] = settings.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = (
                {"type": "function", "function": {"name": require_tool}}
                if require_tool else "auto")

        from openai import OpenAIError

        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        message = response.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
            for c in (message.tool_calls or [])
        ]
        return Reply(
            text=message.content,
            tool_calls=calls,
            tokens=response.usage.total_tokens if response.usage else 0,
        )

    @staticmethod
    def assistant_entry(reply: Reply) -> dict:
        entry: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in reply.tool_calls
            ]
        return entry


# --------------------------------------------------------------------------- #
# Anthropic protocol
# --------------------------------------------------------------------------- #


class AnthropicBackend:
    name = "anthropic"

    def __init__(self) -> None:
        # Imported here, not at module level: staying on OpenAI must not require
        # installing an SDK you never call.
        try:
            import anthropic
        except ModuleNotFoundError as exc:
            raise LLMError(
                "MODEL_NAME points at a Claude model but the `anthropic` package "
                "is not installed — run: .venv/bin/pip install anthropic"
            ) from exc

        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=_secret(settings.anthropic_api_key, settings.api_key,
                            hint="ANTHROPIC_API_KEY"),
            base_url=settings.chat_base_url or None,
            timeout=settings.model_timeout,
            max_retries=2,
        )
        # Turned off permanently if the server rejects the beta header, so a
        # future change on Anthropic's side degrades instead of breaking.
        self._fallbacks = (settings.anthropic_fallbacks
                           and settings.model_name in _FALLBACK_MODELS)

    # -- translation ------------------------------------------------------- #

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        system = [m["content"] for m in messages
                  if m["role"] == "system" and m.get("content")]
        turns = [m for m in messages if m["role"] != "system"]
        return "\n\n".join(system), turns

    @staticmethod
    def _to_blocks(entry: dict) -> list[dict]:
        """One canonical assistant entry -> Anthropic content blocks."""
        if entry.get("_blocks"):
            return entry["_blocks"]          # replay verbatim (thinking, signatures)
        blocks: list[dict] = []
        if entry.get("content"):
            blocks.append({"type": "text", "text": entry["content"]})
        for call in entry.get("tool_calls", []):
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            blocks.append({
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": arguments,
            })
        return blocks

    @classmethod
    def _to_messages(cls, turns: list[dict]) -> list[dict]:
        out: list[dict] = []
        pending: list[dict] = []       # tool results awaiting one user message

        def flush() -> None:
            if pending:
                out.append({"role": "user", "content": pending.copy()})
                pending.clear()

        for entry in turns:
            role = entry["role"]
            if role == "tool":
                # Every result answering the same assistant turn goes into a
                # single user message — splitting them is a protocol error.
                pending.append({
                    "type": "tool_result",
                    "tool_use_id": entry["tool_call_id"],
                    "content": entry["content"] or "",
                })
                continue
            flush()
            if role == "user":
                if entry.get("content"):
                    out.append({"role": "user", "content": entry["content"]})
            elif role == "assistant":
                blocks = cls._to_blocks(entry)
                if blocks:                    # an empty message is rejected
                    out.append({"role": "assistant", "content": blocks})
        flush()
        return out

    @staticmethod
    def _to_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"]["parameters"],
            }
            for tool in tools
        ]

    # -- the call ---------------------------------------------------------- #

    def complete(self, messages: list[dict], tools: list[dict] | None,
                 require_tool: str | None = None) -> Reply:
        system, turns = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": settings.model_name,
            "max_tokens": settings.max_output_tokens,
            "messages": self._to_messages(turns),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_tools(tools)
            if require_tool:
                kwargs["tool_choice"] = {"type": "tool", "name": require_tool}
        # No temperature: Opus 5 / 4.8 / Sonnet 5 reject it with a 400.

        response = self._send(kwargs)

        # A safety classifier can decline the request. That arrives as HTTP 200
        # with an empty or partial content list — reading content[0] blindly is
        # how this turns into a confusing crash instead of a clear message.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMError(f"the model declined this request (category: {category})")

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input, ensure_ascii=False),
                ))
            elif block.type == "fallback":
                # A different model finished this turn; say so rather than let
                # the substitution pass silently.
                print(f"↩️  {block.from_.model} declined — continued on {block.to.model}")

        usage = response.usage
        return Reply(
            text="\n".join(text_parts) or None,
            tool_calls=calls,
            tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            blocks=response.content,
        )

    def _send(self, kwargs: dict):
        try:
            if self._fallbacks:
                try:
                    return self._client.beta.messages.create(
                        betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
                except self._sdk.BadRequestError as exc:
                    if "fallback" not in str(exc).lower():
                        raise
                    print("ℹ️  server-side fallbacks unavailable — continuing without them")
                    self._fallbacks = False
            return self._client.messages.create(**kwargs)
        except self._sdk.AnthropicError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    @staticmethod
    def assistant_entry(reply: Reply) -> dict:
        entry: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in reply.tool_calls
            ]
        if reply.blocks is not None:
            entry["_blocks"] = reply.blocks
        return entry


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

_BACKENDS = {"openai": OpenAIBackend, "anthropic": AnthropicBackend}
_instance = None


def resolve_backend() -> str:
    """Which protocol MODEL_NAME speaks, unless LLM_BACKEND overrides it."""
    choice = (settings.llm_backend or "auto").strip().lower()
    if choice == "auto":
        model = settings.model_name.lower()
        return "anthropic" if model.startswith(("claude", "anthropic")) else "openai"
    if choice not in _BACKENDS:
        raise LLMError(f"unknown LLM_BACKEND={choice!r}; "
                       f"expected one of: auto, {', '.join(_BACKENDS)}")
    return choice


def get_backend():
    global _instance
    if _instance is None:
        _instance = _BACKENDS[resolve_backend()]()
    return _instance


def describe() -> str:
    """One line for startup output — which model is answering, over which wire."""
    backend = resolve_backend()
    where = settings.chat_base_url or (
        "api.anthropic.com" if backend == "anthropic" else "api.openai.com")
    return f"{settings.model_name} via {backend} → {where}"
