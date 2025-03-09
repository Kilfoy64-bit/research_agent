"""
Agent graph configuration and execution.
"""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.state.state import AgentState
from src.tools.web_search import WebSearchTool
from src.models.llm import setup_llm, get_mock_response
from src.agent.nodes import (
    call_model,
    route_to_tool_or_end,
    run_web_search,
    final_answer,
)


def build_research_agent(llm_with_tools: Any = None) -> StateGraph:
    """Build the research agent graph.

    Args:
        llm_with_tools: The language model with tools bound, or None for mock responses

    Returns:
        The compiled state graph for the research agent
    """
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

    # Compile the graph
    return workflow.compile()


def format_research_report(state: AgentState) -> str:
    """Format the final research report.

    Args:
        state: The final agent state

    Returns:
        A formatted research report as a string
    """
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
    return report


def run_agent(query: str) -> str:
    """Run the research agent with a query.

    Args:
        query: The research query

    Returns:
        A formatted research report
    """
    # Setup tools and LLM
    tools = [WebSearchTool()]
    llm_with_tools = setup_llm(tools)

    # Reset the mock response counter when starting a new run if using mocks
    if not llm_with_tools:
        get_mock_response.call_count = 0

    # Build the agent
    agent = build_research_agent(llm_with_tools)

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "research_steps": [],
        "sources": [],
    }

    # Run the agent
    result = agent.invoke(initial_state)

    # Format the report
    return format_research_report(result)
