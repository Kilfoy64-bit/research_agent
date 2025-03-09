"""
State management for the research agent.
"""

from typing import List, Any, Optional, Dict, Annotated
import operator
from pydantic import BaseModel, Field

from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class AgentInput(BaseModel):
    """Input for the research agent."""

    messages: List[BaseMessage] = Field(description="The research question to answer")


class ResearchSection(BaseModel):
    """A section of research content."""

    title: str = Field(description="Title of this research section")
    description: str = Field(
        description="Brief description of the research focus for this section"
    )
    content: str = Field(
        default="", description="Content of the section after research"
    )
    completed: bool = Field(
        default=False, description="Whether this section is completed"
    )


class SearchQuery(BaseModel):
    """A search query for web research."""

    query: str = Field(description="The search query text")
    section_id: Optional[int] = Field(
        default=None, description="ID of the section this query relates to"
    )


class AgentState(MessagesState):
    """The state of our research agent."""

    messages: List[BaseMessage] = []  # Default to empty list
    research_topic: Optional[str] = None
    sections: List[ResearchSection] = []
    current_section_index: Optional[int] = None
    research_steps: List[str] = []
    sources: List[str] = []
    search_queries: List[SearchQuery] = []
    search_results: Dict[str, Any] = {}
    search_iterations: int = 0
    max_search_iterations: int = 3
    formatted_report: Optional[str] = None
    # Special key for aggregating completed sections
    completed_sections: Annotated[List[ResearchSection], operator.add] = []
