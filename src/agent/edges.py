"""
Edge functions for the research agent graph.

These functions handle conditional routing between nodes in the agent graph.
"""

from typing import Literal, Optional

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

    # Check if we have search queries in the state
    if state.get("search_queries"):
        logger.info("Routing to web_search tool based on search queries in state")
        return "web_search"

    # Check if we need to move to the next section
    if _should_continue_research(state):
        logger.info("Routing to final_answer to write section content")
        return "final_answer"

    # If no tool calls or unrecognized tool, go to final answer
    logger.info("Routing to final_answer for final report")
    return "final_answer"


def _should_continue_research(state: AgentState) -> bool:
    """Determine if we should continue with research.

    Args:
        state: The current agent state

    Returns:
        True if we should continue research, False otherwise
    """
    # Check if we have a current section
    current_section_index = state.get("current_section_index")
    if current_section_index is None:
        return False

    # Check if we've reached the maximum search iterations
    search_iterations = state.get("search_iterations", 0)
    max_iterations = state.get("max_search_iterations", 3)

    if search_iterations >= max_iterations:
        logger.info(f"Reached maximum search iterations ({max_iterations})")
        return True

    # Check if we have search results
    search_results = state.get("search_results", {})
    if search_results:
        logger.info("Have search results, moving to section writing")
        return True

    return False
