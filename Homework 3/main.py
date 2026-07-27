"""Interactive REPL for the research agent."""

import uuid

from langchain_core.messages import AIMessage, ToolMessage

from agent import agent
from config import settings

# One conversation thread per process run: the checkpointer keys the history by it.
THREAD_ID = str(uuid.uuid4())

CONFIG = {
    "configurable": {"thread_id": THREAD_ID},
    # Last-resort stop for the graph itself. ModelCallLimitMiddleware is the real
    # limiter, so this needs enough headroom (one model call spans several graph
    # supersteps) to let the middleware finish the run gracefully.
    "recursion_limit": 4 * settings.max_iterations + 10,
}

TOOL_RESULT_PREVIEW = 160


def _preview(text: str, limit: int = TOOL_RESULT_PREVIEW) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _print_step(message) -> None:
    """Render one message from the agent loop as a ReAct-style trace line."""
    if isinstance(message, AIMessage):
        if message.tool_calls:
            for call in message.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                print(f"  → {call['name']}({_preview(args, 120)})")
        elif message.content:
            print(f"\nAgent: {message.content}")
    elif isinstance(message, ToolMessage):
        marker = "✗" if str(message.content).startswith("ERROR") else "✓"
        print(f"    {marker} {_preview(message.content)}")


def main():
    print("Research Agent (type 'exit' to quit)")
    print(f"model: {settings.model_name} | max model calls per request: {settings.max_iterations}")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config=CONFIG,
            ):
                for node_state in chunk.values():
                    if not isinstance(node_state, dict):
                        continue
                    for message in node_state.get("messages", []):
                        _print_step(message)
        except KeyboardInterrupt:
            print("\n[interrupted — the partial conversation is kept in memory]")
        except Exception as exc:
            print(f"\n[agent error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
