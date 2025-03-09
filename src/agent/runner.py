"""
Execution runner for the research agent.

This module contains the code to execute the research agent
with a given query and return the results.
"""

from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from src.agent.graph import build_research_agent
from src.tools.web_search import WebSearchTool
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_agent(query: str) -> str:
    """Run the research agent to answer a query.

    Args:
        query: The research question to answer

    Returns:
        A formatted research report
    """
    logger.info("Running research agent with query: %s", query)

    # Build the agent with our tools
    logger.debug("Building research agent")
    agent = build_research_agent()

    # Initial state
    logger.debug("Initializing agent state")
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "research_steps": [],
        "sources": [],
    }

    # Run the agent
    logger.info("Invoking research agent")
    result = agent.invoke(initial_state)
    logger.info("Research agent execution completed")

    # Return the formatted report from the state
    return result.get(
        "formatted_report", "No formatted report found in the result state."
    )
