"""
FinalAnswerNode for the research agent.
"""

import logging
import textwrap
from typing import List, Any

from src.state.state import AgentState
from src.utils.logging import get_logger

logger = logging.getLogger(__name__)


class FinalAnswerNode:
    """Node that generates the final answer."""

    def __call__(self, state: AgentState) -> AgentState:
        """Generate the final answer.

        Args:
            state: The current agent state

        Returns:
            The final agent state with the formatted report
        """
        logger.info("Generating final answer")

        formatted_report = self.format_research_report(state)
        state["formatted_report"] = formatted_report

        return state

    def format_research_report(self, state: AgentState) -> str:
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

        report = textwrap.dedent(
            """
            RESEARCH REPORT
            --------------

            RESULTS:
            {final_message}

            RESEARCH STEPS:
            {research_steps_formatted}

            SOURCES:
            {sources_formatted}
            """.format(
                final_message=final_message,
                research_steps_formatted=research_steps_formatted,
                sources_formatted=sources_formatted,
            )
        )
        logger.debug("Research report generated with %d characters", len(report))
        return report
