"""
Graph definition for the research agent.

This module contains the definition of the research agent's graph structure,
focusing solely on the graph construction without execution logic.
"""

from typing import Any, Optional, List

from langgraph.graph import StateGraph, END

from src.state.state import AgentInput, AgentState
from src.agent.nodes import PlannerNode, WebSearchNode, FinalAnswerNode
from src.agent.edges import route_to_tool_or_end
from src.tools.web_search import WebSearchTool, SearchProvider
from src.models.llm import LLMContainer, setup_container
from src.utils.config import AgentConfig
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def build_research_agent(
    config: Optional[AgentConfig] = None,
) -> StateGraph:
    """Build the research agent graph.

    Args:
        config: Optional agent configuration

    Returns:
        StateGraph: The configured research agent graph
    """
    # Use default config if not provided
    if config is None:
        from src.utils.config import default_config

        config = default_config

    # Set up container and get models
    logger.debug("Setting up LLM container")
    container = setup_container(config)

    # Create web search tool
    web_search_tool = WebSearchTool(
        provider=config.search_provider.value, max_results=config.max_results_per_query
    )

    # Get models from container
    planner_base_model = container.planner_model()
    writer_model = container.writer_model()

    # Bind tools to planner model
    planner_model = planner_base_model.bind_tools([web_search_tool])

    # Build the graph with nodes
    logger.debug("Building research agent graph")
    builder = StateGraph(AgentState, input=AgentInput)

    # Add nodes - passing pre-configured models to each node that needs them
    builder.add_node("planner", PlannerNode(planner_model, config=config))
    builder.add_node("web_search", WebSearchNode(config=config))
    builder.add_node(
        "final_answer", FinalAnswerNode(writer_model=writer_model, config=config)
    )

    # Add edges
    builder.add_conditional_edges("planner", route_to_tool_or_end)
    builder.add_edge("web_search", "planner")
    builder.add_edge("final_answer", END)

    # Set the entry point
    builder.set_entry_point("planner")

    # Compile the graph
    run_config = {"run_name": "research_agent"}
    return builder.compile().with_config(run_config)
