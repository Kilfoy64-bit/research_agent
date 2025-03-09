"""
Graph definition for the research agent.

This module contains the definition of the research agent's graph structure,
focusing solely on the graph construction without execution logic.
"""

from typing import Any, Optional, List

from langgraph.graph import StateGraph, END

from src.state.state import AgentState
from src.agent.nodes import PlannerNode, WebSearchNode, FinalAnswerNode
from src.agent.edges import route_to_tool_or_end
from src.tools.web_search import WebSearchTool
from src.models.llm import setup_llm
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def build_research_agent(
    llm_with_tools: Optional[Any] = None, tools: Optional[List[Any]] = None
) -> StateGraph:
    """Build the research agent graph.

    Args:
        llm_with_tools: Optional pre-configured LLM with tools. If not provided,
                       one will be created internally.
        tools: Optional list of tools to use if llm_with_tools is not provided.
               If neither are provided, default tools will be used.

    Returns:
        StateGraph: The configured research agent graph
    """
    # Set up LLM with tools if not provided
    if llm_with_tools is None:
        logger.debug("No LLM provided, creating default LLM with tools")
        # Use provided tools or create default tools
        if tools is None:
            logger.debug("No tools provided, using default tools")
            tools = [WebSearchTool()]

        llm_with_tools = setup_llm(tools)

    # Build the graph with nodes
    logger.debug("Building research agent graph")
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("planner", PlannerNode(llm_with_tools))
    builder.add_node("web_search", WebSearchNode())
    builder.add_node("final_answer", FinalAnswerNode())

    # Add edges
    builder.add_conditional_edges("planner", route_to_tool_or_end)
    builder.add_edge("web_search", "planner")
    builder.add_edge("final_answer", END)

    # Set the entry point
    builder.set_entry_point("planner")

    # Compile the graph
    run_config = {"run_name": "research_agent"}
    return builder.compile().with_config(run_config)
