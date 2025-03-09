"""
Language model setup and mock responses for the research agent.
"""

import os
from typing import List, Any, Dict, Optional

from langchain_openai import ChatOpenAI

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def setup_llm(tools: List[BaseTool]) -> Optional[Any]:
    """Set up the language model with tools.

    Args:
        tools: List of tools to bind to the LLM

    Returns:
        LLM with tools bound, or None if using mock responses
    """
    logger.info("Using OpenAI for LLM")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return None

    logger.debug(
        f"Using API key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 9 else ''}"
    )

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o",
        temperature=0.1,
    )
    return llm.bind_tools(tools)
