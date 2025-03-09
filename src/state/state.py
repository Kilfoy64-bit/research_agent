"""
State management for the research agent.
"""

from typing import List, Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """The state of our research agent."""

    messages: List[Any]
    research_steps: List[str] = []
    sources: List[str] = []
