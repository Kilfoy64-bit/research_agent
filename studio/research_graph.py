"""
Wrapper module to expose the research agent graph to LangGraph Studio.
This module decouples the research agent from the main application logic.
"""

from typing import Any, Dict, List

from src.agent.graph import build_research_agent
from src.state.state import AgentState
from src.utils.logging import get_logger, configure_logging


# Configure logging
configure_logging()
logger = get_logger(__name__)


def get_research_graph() -> Any:
    """
    Returns the research agent graph for visualization in LangGraph Studio.

    This function creates the graph with mock implementations of LLM and tools,
    allowing it to be visualized without executing any business logic.

    Returns:
        The research agent graph instance
    """
    logger.info("Building research agent graph for LangGraph Studio")

    # Build the research agent graph
    graph = build_research_agent()

    return graph


# Expose the graph for LangGraph Studio
graph = get_research_graph()
