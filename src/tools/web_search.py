"""
Web search tool for the research agent.
"""

import os
import json
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class SearchProvider(Enum):
    """Supported search providers."""

    TAVILY = "tavily"
    PLACEHOLDER = "placeholder"


class WebSearchTool(BaseTool):
    """Tool for searching the web."""

    name: str = "web_search"
    description: str = "Search the web for information."
    search_provider: SearchProvider = Field(
        default=SearchProvider.PLACEHOLDER, description="Search provider to use"
    )
    max_results: int = Field(
        default=5, description="Maximum number of results to return"
    )

    def __init__(self, provider: str = "placeholder", max_results: int = 5):
        """Initialize the web search tool.

        Args:
            provider: The search provider to use ('tavily' or 'placeholder')
            max_results: Maximum number of results to return
        """
        # Initialize with provided values
        super().__init__(
            search_provider=SearchProvider(provider), max_results=max_results
        )

        # Initialize clients for selected provider
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients based on selected provider."""
        # Initialize Tavily client if selected
        if self.search_provider == SearchProvider.TAVILY:
            try:
                from tavily import TavilyClient, AsyncTavilyClient

                # Check if API key is set
                if not os.environ.get("TAVILY_API_KEY"):
                    logger.warning(
                        "TAVILY_API_KEY environment variable not set. Falling back to placeholder search."
                    )
                    self.search_provider = SearchProvider.PLACEHOLDER
                else:
                    self.tavily_client = TavilyClient()
                    self.tavily_async_client = AsyncTavilyClient()
                    logger.info("Using Tavily search API")
            except ImportError:
                logger.warning(
                    "Failed to import Tavily client. Falling back to placeholder search."
                )
                self.search_provider = SearchProvider.PLACEHOLDER

    def _run(self, query: str) -> str:
        """Run the tool.

        Args:
            query: The search query

        Returns:
            The search results as a string
        """
        logger.info("Performing web search for query: %s", query)

        if self.search_provider == SearchProvider.TAVILY:
            try:
                return self._run_tavily_search(query)
            except Exception as e:
                logger.error(f"Error performing Tavily search: {e}")
                # Fall back to placeholder search
                return self._run_placeholder_search(query)
        else:
            return self._run_placeholder_search(query)

    def _run_tavily_search(self, query: str) -> str:
        """Run search using Tavily API.

        Args:
            query: The search query

        Returns:
            Formatted search results as a string
        """
        response = self.tavily_client.search(
            query=query,
            max_results=self.max_results,
            include_raw_content=True,
            topic="general",
        )

        # Format the results
        results = []
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", f"Result {i}")
            url = result.get("url", "")
            content = result.get("content", "")

            results.append(f"Source {i}: {title}\nURL: {url}\nSummary: {content}\n")

        # Combine all results
        if results:
            formatted_results = "\n".join(results)
            return f"Search results for '{query}':\n\n{formatted_results}"
        else:
            return f"No results found for '{query}'."

    def _run_placeholder_search(self, query: str) -> str:
        """Run a placeholder search.

        Args:
            query: The search query

        Returns:
            Placeholder search results as a string
        """
        result = f"Results for '{query}': Found some information about this topic. Note: Using placeholder search."
        logger.debug("Search complete with %d characters of results", len(result))
        return result

    async def async_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Perform asynchronous search for multiple queries.

        Args:
            queries: List of search queries

        Returns:
            List of search responses
        """
        if self.search_provider == SearchProvider.TAVILY:
            try:
                return await self._async_tavily_search(queries)
            except Exception as e:
                logger.error(f"Error performing async Tavily search: {e}")
                return [{"query": q, "results": []} for q in queries]
        else:
            # For placeholder, just return dummy results
            return [
                {
                    "query": query,
                    "results": [
                        {
                            "title": f"Placeholder Result for {query}",
                            "url": f"https://example.com/search?q={query}",
                            "content": f"This is placeholder content for {query}.",
                            "raw_content": f"This is the full placeholder content for {query}.",
                        }
                    ],
                }
                for query in queries
            ]

    async def _async_tavily_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Perform asynchronous search with Tavily API.

        Args:
            queries: List of search queries

        Returns:
            List of search responses
        """
        search_tasks = []
        for query in queries:
            search_tasks.append(
                self.tavily_async_client.search(
                    query,
                    max_results=self.max_results,
                    include_raw_content=True,
                    topic="general",
                )
            )

        # Execute all searches concurrently
        search_docs = await asyncio.gather(*search_tasks)

        # Format the results
        results = []
        for i, response in enumerate(search_docs):
            results.append(
                {"query": queries[i], "results": response.get("results", [])}
            )

        return results
