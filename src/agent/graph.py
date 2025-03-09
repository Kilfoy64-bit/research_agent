"""
Agent graph configuration and execution.
"""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.state.state import AgentState
from src.tools.web_search import WebSearchTool
from src.models.llm import setup_llm
from src.agent.nodes import (
    call_model,
    route_to_tool_or_end,
    run_web_search,
    final_answer,
)
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def build_research_agent(llm_with_tools: Any = None) -> StateGraph:
    """Build the research agent graph.

    Args:
        llm_with_tools: The language model with tools bound, or None for mock responses

    Returns:
        The compiled state graph for the research agent
    """
    logger.debug("Building research agent graph")
    workflow = StateGraph(AgentState)

    # Create a node builder that includes the llm
    def call_model_with_llm(state: AgentState) -> AgentState:
        return call_model(state, llm_with_tools)

    # Add nodes
    workflow.add_node("call_model", call_model_with_llm)
    workflow.add_node("web_search", run_web_search)
    workflow.add_node("final_answer", final_answer)

    # Set entry point
    workflow.set_entry_point("call_model")

    # Add edges
    workflow.add_conditional_edges("call_model", route_to_tool_or_end)

    # Add edge from web_search back to call_model
    workflow.add_edge("web_search", "call_model")

    # Add edge from final_answer to END
    workflow.add_edge("final_answer", END)

    logger.debug("Compiling research agent graph")
    # Compile the graph
    return workflow.compile()


def format_research_report(state: AgentState) -> str:
    """Format the final research report.

    Args:
        state: The final agent state

    Returns:
        A formatted research report as a string
    """
    logger.debug("Formatting research report")
    messages = state["messages"]
    final_message = messages[-1].content if messages else "No results"

    research_steps = state.get("research_steps", [])
    research_steps_formatted = "\n".join([f"- {step}" for step in research_steps])

    sources = state.get("sources", [])
    sources_formatted = "\n".join([f"- {source}" for source in sources])

    report = f"""
RESEARCH REPORT
--------------

RESULTS:
{final_message}

RESEARCH STEPS:
{research_steps_formatted}

SOURCES:
{sources_formatted}
"""
    logger.debug("Research report generated with %d characters", len(report))
    return report


def run_agent(query: str) -> str:
    """Run the research agent with a query.

    Args:
        query: The research query

    Returns:
        A formatted research report
    """
    logger.info("Running research agent with query: %s", query)

    # Setup tools and LLM
    tools = [WebSearchTool()]
    llm_with_tools = setup_llm(tools)
    # Build the agent
    logger.debug("Building research agent")
    agent = build_research_agent(llm_with_tools)

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

    # Format the report
    return format_research_report(result)
