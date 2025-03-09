"""
Agent nodes for the research agent graph.
"""

from typing import List, Literal, Any

from langchain_core.messages import ToolMessage

from src.state.state import AgentState
from src.tools.web_search import WebSearchTool
from src.models.llm import get_mock_response


def call_model(state: AgentState, llm_with_tools: Any = None) -> AgentState:
    """Call the language model to determine what to do next.

    Args:
        state: The current agent state
        llm_with_tools: The language model with tools bound, or None for mock responses

    Returns:
        Updated agent state with the model's response
    """
    messages = state["messages"]

    # Use the real LLM if available, otherwise use mock responses
    if llm_with_tools:
        response = llm_with_tools.invoke(messages)
    else:
        # Get a mock response
        response = get_mock_response()

    return {"messages": messages + [response]}


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
            return "web_search"

    # If no tool calls or unrecognized tool, go to final answer
    return "final_answer"


def run_web_search(state: AgentState) -> AgentState:
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

    # Run the tool
    search_results = WebSearchTool()._run(tool_args.get("query", ""))

    # Get existing steps and sources or initialize empty lists
    research_steps = state.get("research_steps", [])
    sources = state.get("sources", [])

    # Add results to research steps and sources
    research_steps.append(f"Searched for: {tool_args.get('query', '')}")
    sources.append("Web search results")

    # Create a tool response message
    tool_response = ToolMessage(
        content=search_results, tool_call_id=tool_call.get("id"), name="web_search"
    )

    # Return updated state
    return {
        "messages": messages + [tool_response],
        "research_steps": research_steps,
        "sources": sources,
    }


def final_answer(state: AgentState) -> AgentState:
    """Generate the final answer.

    Args:
        state: The current agent state

    Returns:
        The final agent state
    """
    # No changes to state, just marking it as finished
    return state
