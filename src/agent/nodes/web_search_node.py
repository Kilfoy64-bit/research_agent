"""
WebSearchNode for the research agent.
"""

import logging
from typing import Any, Optional

from langchain_core.messages import ToolMessage

from src.state.state import AgentState
from src.tools.web_search import WebSearchTool
from src.utils.logging import get_logger

logger = logging.getLogger(__name__)


class WebSearchNode:
    """Node that runs web searches."""

    def __init__(self, web_search_tool: Optional[Any] = None):
        self.web_search_tool = web_search_tool or WebSearchTool()

    def __call__(self, state: AgentState) -> AgentState:
        """Run the web search tool.

        Args:
            state: The current agent state

        Returns:
            Updated agent state with the search results
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Extract tool call
        tool_call = last_message.tool_calls[0]
        tool_args = tool_call.get("args", {})
        query = tool_args.get("query", "")

        logger.info("Running web search for: %s", query)

        # Run the tool
        search_results = self.web_search_tool._run(query)

        # Get existing steps and sources or initialize empty lists
        research_steps = state.get("research_steps", [])
        sources = state.get("sources", [])

        # Add results to research steps and sources
        research_steps.append(f"Searched for: {query}")
        sources.append("Web search results")

        logger.debug("Updated research steps (%d total)", len(research_steps))
        logger.debug("Updated sources (%d total)", len(sources))

        # Create a tool response message
        tool_response = ToolMessage(
            content=search_results, tool_call_id=tool_call.get("id"), name="web_search"
        )

        logger.debug("Created tool response message")

        # Return updated state
        return {
            "messages": messages + [tool_response],
            "research_steps": research_steps,
            "sources": sources,
        }
