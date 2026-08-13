"""Hand-written ReAct loop — no LangChain, no AgentExecutor, no create_react_agent.

The whole agent is the loop in `ResearchAgent.run()`:

    user message -> model call (with tool schemas)
                 -> model returns tool_calls? -> execute them, append results -> repeat
                 -> model returns plain text?  -> that is the final answer

Conversation memory is just `self.messages`: the list we keep between requests and
resend on every API call. That list *is* the agent's state.

Nothing here knows which provider answers. The message list is kept in one
canonical format and `llm.py` translates it on the way out — so the loop below
reads the same whether the model is GPT, Claude, or a local one behind Ollama.
"""

import json

from config import SYSTEM_PROMPT, settings
from llm import LLMError, get_backend
from tools import TOOL_SCHEMAS, dispatch

LOG_PREVIEW = 160


def _preview(text: str, limit: int = LOG_PREVIEW) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _format_args(arguments: str) -> str:
    """Render tool arguments for the console log, keeping long values short."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return _preview(arguments, 120)
    return ", ".join(f"{k}={_preview(json.dumps(v, ensure_ascii=False), 80)}" for k, v in parsed.items())


class ResearchAgent:
    def __init__(self) -> None:
        self.llm = get_backend()
        # Dialogue memory: the system prompt stays first, everything else is appended.
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ------------------------------------------------------------------ #
    # the ReAct loop
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> str:
        """Run one user request to completion. Returns the final answer text."""
        self.messages.append({"role": "user", "content": user_input})

        tool_calls_made = 0
        tokens_used = 0
        # Facts about this run that the model must not be trusted to remember.
        saved_reports: list[str] = []

        for step in range(1, settings.max_iterations + 1):
            reply = self._call_model(with_tools=True)
            tokens_used += reply.tokens

            # Persist the turn in whatever shape this backend needs to replay it.
            self.messages.append(self.llm.assistant_entry(reply))

            # Some models emit reasoning text alongside tool calls — that is the
            # "Thought" part of ReAct, worth showing.
            if reply.text and reply.tool_calls:
                print(f"\n💭 {_preview(reply.text, 300)}")

            if not reply.tool_calls:
                print(f"\n📊 {step} step(s), {tool_calls_made} tool call(s), ~{tokens_used} tokens")
                return reply.text or ""

            print(f"\n[step {step}/{settings.max_iterations}]")
            for call in reply.tool_calls:
                tool_calls_made += 1
                name = call.name
                print(f"🔧 Tool call: {name}({_format_args(call.arguments)})")

                result = dispatch(name, call.arguments)

                if name == "write_report" and result.startswith("Report saved to"):
                    saved_reports.append(result.split("Report saved to ", 1)[1].split(" (")[0])

                marker = "⚠️ " if result.startswith("ERROR") else ""
                print(f"📎 Result: {marker}[{len(result)} chars] {_preview(result)}")

                # A tool result MUST reference the id of the call it answers.
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        # Step budget exhausted: ask for a final answer once, without tools, so the
        # model cannot start another round of research.
        print(f"\n⏹  Step limit reached ({settings.max_iterations}) — asking for a final answer.")
        # Without this the model happily says "report saved" while holding no tools —
        # so the saved-files fact is injected from the loop, not left to its memory.
        files = ", ".join(saved_reports) if saved_reports else "NONE"
        self.messages.append(
            {
                "role": "system",
                "content": (
                    "Step budget exhausted. Do not call any more tools — you no longer "
                    f"have any. Reports actually saved during this run: {files}. "
                    "Never claim that a file was saved unless it is in that list; if the "
                    "list is NONE, say plainly that no report was saved. Answer now with "
                    "what you already have and state what remained unverified."
                ),
            }
        )
        reply = self._call_model(with_tools=False)
        tokens_used += reply.tokens
        self.messages.append({"role": "assistant", "content": reply.text})
        print(f"\n📊 {settings.max_iterations} step(s), {tool_calls_made} tool call(s), ~{tokens_used} tokens")
        return reply.text or ""

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _call_model(self, *, with_tools: bool):
        """One model call, whichever provider is configured. Returns a Reply."""
        try:
            return self.llm.complete(self.messages, TOOL_SCHEMAS if with_tools else None)
        except LLMError as exc:
            # Surfaced as one exception type regardless of provider, so main.py
            # needs no per-vendor error handling.
            raise RuntimeError(f"model call failed: {exc}") from exc

    def reset(self) -> None:
        """Forget the dialogue, keep the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def history_size(self) -> int:
        return len(self.messages) - 1  # system prompt is not part of the dialogue
