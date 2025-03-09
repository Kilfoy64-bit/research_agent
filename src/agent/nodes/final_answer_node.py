"""
FinalAnswerNode for the research agent.
"""

import logging
import textwrap
from typing import List, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from src.state.state import AgentState, ResearchSection
from src.utils.config import AgentConfig
from src.utils.prompts import SECTION_WRITING_PROMPT, FINAL_REPORT_PROMPT
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FinalAnswerNode:
    """Node that generates the final answer or writes section content."""

    def __init__(
        self, writer_model: BaseChatModel, config: Optional[AgentConfig] = None
    ):
        """Initialize the final answer node.

        Args:
            writer_model: Pre-configured model for writing content
            config: Optional agent configuration
        """
        self.writer_model = writer_model
        self.config = config
        if self.config is None:
            from src.utils.config import default_config

            self.config = default_config

    def __call__(self, state: AgentState) -> AgentState:
        """Generate the final answer or write section content.

        Args:
            state: The current agent state

        Returns:
            The final agent state with the formatted report
        """
        # Check if we need to write a section or generate the final report
        current_section_index = state.get("current_section_index")
        sections = state.get("sections", [])

        if current_section_index is not None and current_section_index < len(sections):
            # Write content for the current section
            return self._write_section_content(state)
        else:
            # Generate the final report
            return self._generate_final_report(state)

    def _write_section_content(self, state: AgentState) -> AgentState:
        """Write content for the current section based on search results.

        Args:
            state: The current agent state

        Returns:
            Updated state with the section content
        """
        logger.info("Writing section content")

        # Get the current section
        current_section_index = state.get("current_section_index")
        sections = state.get("sections", [])
        current_section = sections[current_section_index]

        # Get search results for this section
        search_results = state.get("search_results", {})

        # Format search results for the prompt
        formatted_results = ""
        for query, results in search_results.items():
            formatted_results += f"Results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"Source {i}: {result.get('title', 'Untitled')}\n"
                formatted_results += f"URL: {result.get('url', 'No URL')}\n"
                formatted_results += (
                    f"Content: {result.get('content', 'No content')}\n\n"
                )

        # If no search results, use a placeholder
        if not formatted_results:
            formatted_results = (
                "No search results available. Please write based on general knowledge."
            )

        # Create the prompt for section writing
        prompt = SECTION_WRITING_PROMPT.format(
            section_title=current_section.title,
            section_description=current_section.description,
            search_results=formatted_results,
        )

        # Generate the section content
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Write content for the section: {current_section.title}"
            ),
        ]

        response = self.writer_model.invoke(messages)
        section_content = response.content

        # Update the section with the new content
        updated_sections = sections.copy()
        updated_sections[current_section_index] = ResearchSection(
            title=current_section.title,
            description=current_section.description,
            content=section_content,
            completed=True,
        )

        # Find the next section that needs research
        next_section_index = None
        for i in range(current_section_index + 1, len(updated_sections)):
            if not updated_sections[i].completed:
                next_section_index = i
                break

        # Add the completed section to the completed_sections list
        completed_sections = state.get("completed_sections", [])
        completed_sections.append(updated_sections[current_section_index])

        # Update the state
        updated_state = state.copy()
        updated_state.update(
            {
                "sections": updated_sections,
                "current_section_index": next_section_index,
                "completed_sections": completed_sections,
                "search_queries": [],  # Clear search queries for the next section
                "search_results": {},  # Clear search results for the next section
                "search_iterations": 0,  # Reset search iterations for the next section
            }
        )

        # If all sections are complete, generate the final report
        if next_section_index is None:
            return self._generate_final_report(updated_state)

        return updated_state

    def _generate_final_report(self, state: AgentState) -> AgentState:
        """Generate the final research report.

        Args:
            state: The final agent state

        Returns:
            State with the formatted report
        """
        logger.info("Generating final report")

        # Get all completed sections
        completed_sections = state.get("completed_sections", [])
        if not completed_sections:
            # If no completed sections, use the sections list
            completed_sections = state.get("sections", [])

        # Format the sections content for the prompt
        sections_content = ""
        for i, section in enumerate(completed_sections, 1):
            sections_content += f"Section {i}: {section.title}\n\n"
            sections_content += f"{section.content}\n\n"

        # Get the research topic
        research_topic = state.get("research_topic", "Unknown Topic")

        # Create the prompt for the final report
        prompt = FINAL_REPORT_PROMPT.format(
            topic=research_topic,
            report_structure=self.config.report_structure,
            sections_content=sections_content,
        )

        # Generate the final report
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Generate the final report for: {research_topic}"),
        ]

        response = self.writer_model.invoke(messages)
        final_report = response.content

        # Format the final report with metadata
        formatted_report = self.format_research_report(state, final_report)

        # Update the state
        updated_state = state.copy()
        updated_state["formatted_report"] = formatted_report

        return updated_state

    def format_research_report(self, state: AgentState, final_report: str) -> str:
        """Format the final research report with metadata.

        Args:
            state: The final agent state
            final_report: The generated report content

        Returns:
            A formatted research report as a string
        """
        logger.debug("Formatting research report")

        research_topic = state.get("research_topic", "Unknown Topic")
        research_steps = state.get("research_steps", [])
        research_steps_formatted = "\n".join([f"- {step}" for step in research_steps])

        sources = state.get("sources", [])
        sources_formatted = "\n".join([f"- {source}" for source in sources])

        report = textwrap.dedent(
            f"""
            RESEARCH REPORT: {research_topic}
            ==================================

            {final_report}

            ----------------------------------
            RESEARCH PROCESS:
            {research_steps_formatted}

            SOURCES:
            {sources_formatted}
            """
        )
        logger.debug("Research report generated with %d characters", len(report))
        return report
