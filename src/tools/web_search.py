"""
Web search tool for the research agent.
"""

from langchain_core.tools import BaseTool


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
        # This is a placeholder implementation
        return f"Results for '{query}': Found some information about this topic."
