"""
Language model setup and mock responses for the research agent.
"""

import os
from typing import List, Any, Dict, Optional

# Import conditionally since OpenAI might not be available
try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool


def get_mock_response() -> AIMessage:
    """Get a mock response for testing purposes.

    Returns:
        A mock AI message with either a tool call or a final answer
    """
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


def setup_llm(tools: List[BaseTool]) -> Optional[Any]:
    """Set up the language model with tools.

    Args:
        tools: List of tools to bind to the LLM

    Returns:
        LLM with tools bound, or None if using mock responses
    """
    # Set up the LLM - either use OpenAI if available or a mock function
    api_key = os.getenv("OPENAI_API_KEY", "")
    if OPENAI_AVAILABLE and api_key and api_key != "your_openai_api_key_here":
        print("Using OpenAI for LLM")
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.3,
        )
        # Add tool calling capabilities
        return llm.bind_tools(tools)
    else:
        print("Using mock responses for testing (no valid API key found)")
        # We'll use a custom function since FakeListLLM doesn't work well with tool calls
        return None  # Not used in this case, will use mock responses
