"""
PlannerNode for the research agent.
"""

import logging
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

from src.state.state import AgentState
from src.utils.logging import get_logger

logger = logging.getLogger(__name__)


class PlannerNode:
    """Node that plans the next actions by calling the language model."""

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

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

        response = self.runnable.invoke(messages, config=config)

        logger.debug("Planner response received")
        return {"messages": messages + [response]}
