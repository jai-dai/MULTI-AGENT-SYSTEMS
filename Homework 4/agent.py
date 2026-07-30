"""Hand-written ReAct loop — no LangChain, no AgentExecutor, no create_react_agent.

The whole agent is the loop in `ResearchAgent.run()`:

    user message -> model call (with tool schemas)
                 -> model returns tool_calls? -> execute them, append results -> repeat
                 -> model returns plain text?  -> that is the final answer

Conversation memory is just `self.messages`: the list we keep between requests and
resend on every API call. That list *is* the agent's state.
"""

import json

from openai import OpenAI, OpenAIError

from config import SYSTEM_PROMPT, settings
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
        self.client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.model_timeout,
            max_retries=2,
        )
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
            message, usage = self._call_model(with_tools=True)
            tokens_used += usage

            # Persist exactly what the protocol expects: assistant message with the
            # tool_calls it requested (built explicitly, not dumped from the SDK object).
            self.messages.append(self._assistant_message(message))

            # Some models emit reasoning text alongside tool calls — that is the
            # "Thought" part of ReAct, worth showing.
            if message.content and message.tool_calls:
                print(f"\n💭 {_preview(message.content, 300)}")

            if not message.tool_calls:
                print(f"\n📊 {step} step(s), {tool_calls_made} tool call(s), ~{tokens_used} tokens")
                return message.content or ""

            print(f"\n[step {step}/{settings.max_iterations}]")
            for call in message.tool_calls:
                tool_calls_made += 1
                name = call.function.name
                print(f"🔧 Tool call: {name}({_format_args(call.function.arguments)})")

                result = dispatch(name, call.function.arguments)

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
        message, usage = self._call_model(with_tools=False)
        tokens_used += usage
        self.messages.append({"role": "assistant", "content": message.content})
        print(f"\n📊 {settings.max_iterations} step(s), {tool_calls_made} tool call(s), ~{tokens_used} tokens")
        return message.content or ""

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _call_model(self, *, with_tools: bool):
        """One API call. Returns (message, tokens_used)."""
        kwargs = {
            "model": settings.model_name,
            "messages": self.messages,
        }
        # Reasoning models (gpt-5*, o*) accept only the default temperature and
        # reject temperature=0 with a 400.
        if not settings.model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["temperature"] = settings.temperature
        if with_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            # Roll the conversation back to a clean state so the next user request
            # does not start from a half-finished exchange.
            raise RuntimeError(f"model call failed: {type(exc).__name__}: {exc}") from exc

        usage = response.usage.total_tokens if response.usage else 0
        return response.choices[0].message, usage

    @staticmethod
    def _assistant_message(message) -> dict:
        """Convert an SDK response message into a plain protocol dict."""
        result: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return result

    def reset(self) -> None:
        """Forget the dialogue, keep the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def history_size(self) -> int:
        return len(self.messages) - 1  # system prompt is not part of the dialogue
