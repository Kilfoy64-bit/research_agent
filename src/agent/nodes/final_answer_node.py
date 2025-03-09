"""
FinalAnswerNode for the research agent.
"""

import logging

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
            The final agent state
        """
        logger.info("Generating final answer")
        # No changes to state, just marking it as finished
        return state
