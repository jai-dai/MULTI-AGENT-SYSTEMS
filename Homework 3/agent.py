"""Agent assembly: model + tools + memory + loop limits."""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolErrorMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from config import SYSTEM_PROMPT, settings
from tools import TOOLS

llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
    temperature=0,
    timeout=settings.request_timeout,
    max_retries=2,
)

tools = TOOLS

# Checkpointer: keeps the message history per thread_id, so the agent
# remembers earlier turns of the conversation ("now compare it with X").
memory = MemorySaver()

middleware = [
    # Step limit: stop after N model calls per user request instead of looping forever.
    ModelCallLimitMiddleware(run_limit=settings.max_iterations, exit_behavior="end"),
    # Unexpected tool exceptions come back to the model as an error message
    # instead of killing the run.
    ToolErrorMiddleware(
        on_error=lambda exc, request: (
            f"ERROR: tool '{request.tool_call['name']}' failed "
            f"({type(exc).__name__}: {exc}). Try different arguments or another source."
        )
    ),
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
    middleware=middleware,
)
