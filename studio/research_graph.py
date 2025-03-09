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


class MockTool:
    """A mock tool for graph visualization purposes."""

    def __init__(self, name: str = "mock_tool"):
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Mock implementation that returns an empty result."""
        return {"result": "This is a mock tool response for visualization purposes."}


class MockLLM:
    """A mock LLM for graph visualization purposes."""

    def invoke(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Mock invocation that returns an empty result."""
        return {"content": "This is a mock LLM response for visualization purposes."}

    def bind_tools(self, tools: List[Any]) -> Any:
        """Mock tool binding that returns self."""
        return self


def get_research_graph() -> Any:
    """
    Returns the research agent graph for visualization in LangGraph Studio.

    This function creates the graph with mock implementations of LLM and tools,
    allowing it to be visualized without executing any business logic.

    Returns:
        The research agent graph instance
    """
    logger.info("Building research agent graph for LangGraph Studio")

    # Create mock tools for visualization
    # mock_tools = [MockTool(name="web_search")]

    # Build the research agent graph with mock implementations
    graph = build_research_agent()

    return graph


# Expose the graph for LangGraph Studio
graph = get_research_graph()
