#!/usr/bin/env python3
"""
Main entry point for the research agent application.
This implements a basic LangGraph-based research agent.
"""
import os
import json
from typing import Dict, List, Literal, TypedDict, Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
from langgraph.graph import StateGraph, END, MessagesState


# Load environment variables
load_dotenv()


class AgentState(MessagesState):
    """The state of our research agent."""

    messages: List[Any]
    research_steps: List[str] = []
    sources: List[str] = []


class WebSearchTool(BaseTool):
    """Tool for searching the web."""

    name: str = "web_search"
    description: str = "Search the web for information."

    def _run(self, query: str) -> str:
        """Run the tool."""
        # This is a placeholder implementation
        return f"Results for '{query}': Found some information about this topic."


# Helper function to create a mock response
def get_mock_response():
    """Get a mock response for testing purposes."""
    # For the first call, we'll simulate a tool call
    if not hasattr(get_mock_response, "call_count"):
        get_mock_response.call_count = 0

    if get_mock_response.call_count == 0:
        # First call - simulate a tool call
        response = AIMessage(
            content="I'll search for that information.",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "name": "web_search",
                    "args": {"query": "research agent information"},
                }
            ],
        )
    else:
        # Second call - simulate a final answer
        response = AIMessage(
            content=(
                "Based on the search results, here's what I found: Research agents are AI systems "
                "designed to conduct in-depth research on topics. They can search the web, analyze "
                "information, and compile reports. They're particularly useful for academic research, "
                "market analysis, and gathering information on complex topics."
            )
        )

    get_mock_response.call_count += 1
    return response


# Create available tools
tools = [WebSearchTool()]

# Set up the LLM - either use OpenAI if available or a mock function
api_key = os.getenv("OPENAI_API_KEY", "")
if OPENAI_AVAILABLE and api_key and api_key != "your_openai_api_key_here":
    print("Using OpenAI for LLM")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.3,
    )
    # Add tool calling capabilities
    llm_with_tools = llm.bind_tools(tools)
else:
    print("Using mock responses for testing (no valid API key found)")
    # We'll use a custom function since FakeListLLM doesn't work well with tool calls
    llm_with_tools = None  # Not used in this case


def call_model(state: AgentState) -> AgentState:
    """Call the language model to determine what to do next."""
    messages = state["messages"]

    # Use the real LLM if available, otherwise use mock responses
    if llm_with_tools:
        response = llm_with_tools.invoke(messages)
    else:
        # Get a mock response
        response = get_mock_response()

    return {"messages": messages + [response]}


def route_to_tool_or_end(state: AgentState) -> Literal["web_search", "final_answer"]:
    """Route to a tool or finish."""
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
    """Run the web search tool."""
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
    """Generate the final answer."""
    # No changes to state, just marking it as finished
    return state


def build_research_agent() -> StateGraph:
    """Build the research agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("call_model", call_model)
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
    """Format the final research report."""
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
    """Run the research agent with a query."""
    # Reset the mock response counter when starting a new run
    if not llm_with_tools:
        get_mock_response.call_count = 0

    # Build the agent
    agent = build_research_agent()

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


if __name__ == "__main__":
    print("Research Agent started!")
    user_query = input("Enter your research query: ")
    research_results = run_agent(user_query)
    print(research_results)
