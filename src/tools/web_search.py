"""
Web search tool for the research agent.
"""

from langchain_core.tools import BaseTool

from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Tool for searching the web."""

    name: str = "web_search"
    description: str = "Search the web for information."

    def _run(self, query: str) -> str:
        """Run the tool.

        Args:
            query: The search query

        Returns:
            The search results as a string
        """
        logger.info("Performing web search for query: %s", query)
        # This is a placeholder implementation
        result = f"Results for '{query}': Found some information about this topic."
        logger.debug("Search complete with %d characters of results", len(result))
        return result
