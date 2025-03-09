"""
PlannerNode for the research agent.
"""

import re
from typing import Any, List, Optional, Dict

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from src.state.state import AgentState, ResearchSection, SearchQuery
from src.utils.logging import get_logger
from src.utils.config import AgentConfig
from src.utils.prompts import RESEARCH_PLAN_PROMPT, QUERY_GENERATION_PROMPT

logger = get_logger(__name__)


class PlannerNode:
    """Node that plans the next actions and generates queries."""

    def __init__(self, model: BaseChatModel, config: Optional[AgentConfig] = None):
        """Initialize the planner node.

        Args:
            model: The pre-configured model to use for planning
            config: Optional agent configuration
        """
        self.model = model
        self.config = config
        if self.config is None:
            from src.utils.config import default_config

            self.config = default_config

    def __call__(self, state: AgentState, config: RunnableConfig = None) -> AgentState:
        """Plan the next steps by calling the language model with the current messages.

        Args:
            state: The current agent state
            config: Optional runnable configuration

        Returns:
            Updated agent state with the planner's response
        """
        messages = state["messages"]
        logger.debug("Planning next steps with %d messages", len(messages))

        # If this is the first planning step, create a research plan
        sections = state.get("sections", [])
        if not sections:
            return self._create_research_plan(state, config)

        # If there's a current section, generate queries for it
        current_section_index = state.get("current_section_index")
        if current_section_index is not None:
            return self._generate_queries(state, config)

        # Otherwise, just pass through to the LLM for general planning
        response = self.model.invoke(messages, config=config)
        logger.debug("Planner response received")
        return {"messages": messages + [response]}

    def _create_research_plan(
        self, state: AgentState, config: Optional[RunnableConfig]
    ) -> AgentState:
        """Create the initial research plan.

        Args:
            state: Current agent state
            config: Optional runnable configuration

        Returns:
            Updated state with research sections
        """
        messages = state["messages"]
        last_message = messages[-1]["content"] if messages else ""

        # Extract research topic from the last message if not already set
        research_topic = state.get("research_topic")
        if not research_topic:
            research_topic = last_message
            logger.info(f"Setting research topic: {research_topic}")

        # Format the prompt with the topic and report structure
        prompt = RESEARCH_PLAN_PROMPT.format(
            topic=research_topic, report_structure=self.config.report_structure
        )

        # Get response from LLM
        logger.debug("Generating research plan")
        planner_messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Create a research plan for the topic: {research_topic}"
            ),
        ]
        response = self.model.invoke(planner_messages)
        plan_content = response.content

        # Parse the response to extract sections
        sections = self._parse_sections(plan_content)

        # Update state with the research plan
        updated_state = {
            "messages": messages + [response],
            "research_topic": research_topic,
            "sections": sections,
            "current_section_index": self._find_first_research_section(sections),
        }

        logger.info(f"Created research plan with {len(sections)} sections")
        return updated_state

    def _generate_queries(
        self, state: AgentState, config: Optional[RunnableConfig]
    ) -> AgentState:
        """Generate search queries for the current section.

        Args:
            state: Current agent state
            config: Optional runnable configuration

        Returns:
            Updated state with search queries
        """
        sections = state["sections"]
        current_section_index = state["current_section_index"]
        research_topic = state["research_topic"]

        # Get the current section
        if current_section_index >= len(sections):
            logger.warning("Current section index out of bounds")
            return state

        current_section = sections[current_section_index]

        # Prepare research context from existing search results if any
        research_context = ""
        if state.get("search_results"):
            research_context = (
                "Previous search results available but not included for brevity."
            )

        # Format the prompt for query generation
        prompt = QUERY_GENERATION_PROMPT.format(
            topic=research_topic,
            section_title=current_section.title,
            section_description=current_section.description,
            research_context=research_context,
            num_queries=self.config.search_queries_per_iteration,
        )

        # Get response from LLM
        logger.debug(f"Generating queries for section: {current_section.title}")
        query_messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Generate search queries for section: {current_section.title}"
            ),
        ]
        response = self.model.invoke(query_messages)

        # Parse the response to extract queries
        queries = self._parse_queries(response.content, current_section_index)

        # Update state with the generated queries
        updated_state = state.copy()
        updated_state["messages"] = state["messages"] + [response]
        updated_state["search_queries"] = queries

        logger.info(
            f"Generated {len(queries)} search queries for section {current_section.title}"
        )
        return updated_state

    def _parse_sections(self, content: str) -> List[ResearchSection]:
        """Parse section information from the LLM output.

        Args:
            content: The response content from the LLM

        Returns:
            List of research sections
        """
        sections = []

        # Very basic parsing - in a real implementation, we might want more robust parsing
        # or structured output from the LLM
        section_blocks = re.split(r"\n\s*\d+\.\s+", content)
        for block in section_blocks[
            1:
        ]:  # Skip the first split which is usually intro text
            lines = block.strip().split("\n")
            if not lines:
                continue

            title = lines[0].strip()
            description = ""
            requires_research = True

            for line in lines[1:]:
                if "research" in line.lower() and (
                    "false" in line.lower() or "no" in line.lower()
                ):
                    requires_research = False
                else:
                    description += line + " "

            sections.append(
                ResearchSection(
                    title=title,
                    description=description.strip(),
                    content="",
                    completed=not requires_research,  # Sections that don't require research are already complete
                )
            )

        return sections

    def _parse_queries(self, content: str, section_id: int) -> List[SearchQuery]:
        """Parse search queries from the LLM output.

        Args:
            content: The response content from the LLM
            section_id: The ID of the section these queries are for

        Returns:
            List of search queries
        """
        queries = []

        # Extract queries (looking for numbered lists, bullet points, or plain text)
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Skip empty lines and headers
            if not line or line.startswith("#") or line.startswith("<"):
                continue

            # Try to extract queries from numbered or bulleted lists
            match = re.search(r"^(?:\d+\.|\*|-)\s*(.+)$", line)
            if match:
                query_text = match.group(1).strip()
                queries.append(SearchQuery(query=query_text, section_id=section_id))
            elif line and not line.startswith('"') and not line.endswith('"'):
                # If not a list item, but looks like a query, add it
                queries.append(SearchQuery(query=line, section_id=section_id))

        return queries

    def _find_first_research_section(
        self, sections: List[ResearchSection]
    ) -> Optional[int]:
        """Find the index of the first section that requires research.

        Args:
            sections: List of research sections

        Returns:
            Index of the first section requiring research, or None if none found
        """
        for i, section in enumerate(sections):
            if not section.completed:
                return i
        return None
