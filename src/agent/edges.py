"""
Edge functions for the research agent graph.

These functions handle conditional routing between nodes in the agent graph.
"""

from typing import Literal

from src.state.state import AgentState
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def route_to_tool_or_end(state: AgentState) -> Literal["web_search", "final_answer"]:
    """Route to a tool or finish.

    Args:
        state: The current agent state

    Returns:
        The next node to route to
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check if message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call.get("name")
        if tool_name == "web_search":
            logger.info("Routing to web_search tool")
            return "web_search"

    # If no tool calls or unrecognized tool, go to final answer
    logger.info("Routing to final_answer")
    return "final_answer"
