"""
LLM setup for the research agent.

This module sets up language models with dependency injection for the research agent.
"""

import os
from typing import Any, List, Optional, Dict

from dependency_injector import containers, providers
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.utils.config import AgentConfig, ModelProvider
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class LLMContainer(containers.DeclarativeContainer):
    """Container for LLM providers."""

    # Import configuration
    config = providers.Dependency(AgentConfig)

    # OpenAI model providers
    planner_model = providers.Factory(
        ChatOpenAI,
        model=providers.Callable(lambda config: config.planner_model, config=config),
        temperature=0,
    )

    writer_model = providers.Factory(
        ChatOpenAI,
        model=providers.Callable(
            lambda config: config.report_writer_model, config=config
        ),
        temperature=0,
    )


def setup_container(config: Optional[AgentConfig] = None) -> LLMContainer:
    """Set up the LLM container with configuration.

    Args:
        config: Optional agent configuration

    Returns:
        Configured LLM container
    """
    if config is None:
        from src.utils.config import default_config

        config = default_config

    # Create and configure container
    container = LLMContainer()
    container.config.override(config)

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
    else:
        logger.info("OpenAI API key found")

    return container
