"""Interactive REPL for the research agent with a hand-written ReAct loop."""

import llm
import preflight
from agent import ResearchAgent
from config import settings


def main():
    # Before anything heavy is imported or downloaded: can this machine carry
    # the configured models at all? Refuses only when they plainly do not fit,
    # and always says what to change instead.
    preflight.guard()

    agent = ResearchAgent()

    print("Research Agent — custom ReAct loop (type 'exit' to quit, 'reset' to clear memory)")
    # Say which provider is actually answering — with LLM_BACKEND=auto that is
    # derived from the model name, so it should be visible rather than guessed.
    print(f"model: {llm.describe()} | step limit: {settings.max_iterations}")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in ("exit", "quit"):
            print("Goodbye!")
            break
        if command == "reset":
            agent.reset()
            print("Memory cleared.")
            continue

        try:
            answer = agent.run(user_input)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except RuntimeError as exc:
            print(f"\n[agent error] {exc}")
            continue

        print(f"\nAgent: {answer}")
        print(f"[memory: {agent.history_size} messages]")


if __name__ == "__main__":
    main()
