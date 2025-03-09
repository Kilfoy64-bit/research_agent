"""
Agent graph configuration and execution.
"""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.state.state import AgentState
from src.tools.web_search import WebSearchTool
from src.models.llm import setup_llm
from src.agent.nodes import PlannerNode, WebSearchNode, FinalAnswerNode
from src.agent.edges import route_to_tool_or_end
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
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("planner", PlannerNode(llm_with_tools))
    builder.add_node("web_search", WebSearchNode())
    builder.add_node("final_answer", FinalAnswerNode())

    # Set entry point
    builder.set_entry_point("planner")

    # Add edges
    builder.add_conditional_edges("planner", route_to_tool_or_end)

    # Add edge from web_search back to planner
    builder.add_edge("web_search", "planner")

    # Add edge from final_answer to END
    builder.add_edge("final_answer", END)

    logger.debug("Compiling research agent graph")
    # Compile the graph
    return builder.compile()


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
