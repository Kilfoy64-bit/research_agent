"""
Configuration management for the research agent.
"""

import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, ClassVar, Type, List

from langchain_core.runnables import RunnableConfig


class SearchProvider(Enum):
    """Available search providers."""

    TAVILY = "tavily"
    PLACEHOLDER = "placeholder"


class ModelProvider(Enum):
    """Available model providers."""

    OPENAI = "openai"


DEFAULT_REPORT_STRUCTURE = """
1. Introduction
   - Brief overview of the topic

2. Key Findings
   - Main insights from the research
   
3. Detailed Analysis
   - In-depth examination of subtopics
   
4. Conclusion
   - Summary of findings and implications
"""


@dataclass
class AgentConfig:
    """Configuration for the research agent."""

    # Report settings
    report_structure: str = DEFAULT_REPORT_STRUCTURE

    # Research settings
    max_search_iterations: int = 3
    search_queries_per_iteration: int = 2
    max_results_per_query: int = 5

    # Search provider settings
    search_provider: SearchProvider = SearchProvider.PLACEHOLDER

    # Model settings - default all to OpenAI
    planner_provider: ModelProvider = ModelProvider.OPENAI
    planner_model: str = "gpt-4o"  # Upgraded from gpt-3.5-turbo to gpt-4o
    report_writer_provider: ModelProvider = ModelProvider.OPENAI
    report_writer_model: str = "gpt-4-turbo"  # Upgraded from gpt-4 to gpt-4-turbo

    # Environment variables mapping
    env_vars: ClassVar[Dict[str, str]] = {
        "report_structure": "RESEARCH_REPORT_STRUCTURE",
        "max_search_iterations": "RESEARCH_MAX_SEARCH_ITERATIONS",
        "search_queries_per_iteration": "RESEARCH_QUERIES_PER_ITERATION",
        "max_results_per_query": "RESEARCH_MAX_RESULTS_PER_QUERY",
        "search_provider": "RESEARCH_SEARCH_PROVIDER",
        "planner_provider": "RESEARCH_PLANNER_PROVIDER",
        "planner_model": "RESEARCH_PLANNER_MODEL",
        "report_writer_provider": "RESEARCH_WRITER_PROVIDER",
        "report_writer_model": "RESEARCH_WRITER_MODEL",
    }

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        """Create a configuration from environment variables.

        Returns:
            An AgentConfig with values from environment variables
        """
        # Get fields that can be initialized
        config_fields = {f.name: f.type for f in fields(cls) if f.init}
        values: Dict[str, Any] = {}

        # Extract values from environment variables
        for field_name, env_name in cls.env_vars.items():
            if field_name in config_fields and env_name in os.environ:
                env_value = os.environ[env_name]

                # Handle different types
                field_type = config_fields[field_name]
                if field_type == int:
                    values[field_name] = int(env_value)
                elif field_type == float:
                    values[field_name] = float(env_value)
                elif field_type == bool:
                    values[field_name] = env_value.lower() in ("yes", "true", "t", "1")
                elif issubclass(field_type, Enum):
                    # Assuming the env value matches an enum value string
                    values[field_name] = field_type(env_value)
                else:
                    values[field_name] = env_value

        return cls(**values)

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "AgentConfig":
        """Create a configuration from a RunnableConfig.

        Args:
            config: The RunnableConfig object

        Returns:
            An AgentConfig instance
        """
        # Start with environment values
        agent_config = cls.from_environment()

        # Update with values from runnable config if provided
        if config and "configurable" in config:
            configurable = config["configurable"]

            # Update fields from configurable
            for field_name in (f.name for f in fields(cls) if f.init):
                if field_name in configurable:
                    setattr(agent_config, field_name, configurable[field_name])

        return agent_config


# Default configuration instance
default_config = AgentConfig()
