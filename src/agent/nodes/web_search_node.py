"""
WebSearchNode for the research agent.
"""

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.messages import ToolMessage

from src.state.state import AgentState, SearchQuery
from src.tools.web_search import WebSearchTool
from src.utils.config import AgentConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WebSearchNode:
    """Node that runs web searches."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the web search node.

        Args:
            config: Optional agent configuration
        """
        # Set up configuration
        self.config = config
        if self.config is None:
            from src.utils.config import default_config

            self.config = default_config

        # Initialize web search tool
        self.web_search_tool = WebSearchTool(
            provider=self.config.search_provider.value,
            max_results=self.config.max_results_per_query,
        )

    async def __call__(self, state: AgentState) -> AgentState:
        """Run the web search tool.

        Args:
            state: The current agent state

        Returns:
            Updated agent state with the search results
        """
        # Get the search queries from state
        search_queries = state.get("search_queries", [])
        if not search_queries:
            # If no queries in state, try to extract from the last message
            messages = state["messages"]
            last_message = messages[-1]

            # Extract tool call if present
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                tool_args = tool_call.get("args", {})
                query = tool_args.get("query", "")
                search_queries = [SearchQuery(query=query)]

        # Still no queries? Return unchanged state
        if not search_queries:
            logger.warning("No search queries found in state or message")
            return state

        # Log the queries
        logger.info(f"Running web search for {len(search_queries)} queries")
        for i, query in enumerate(search_queries):
            logger.debug(f"Query {i+1}: {query.query}")

        # Extract just the query strings
        query_strings = [q.query for q in search_queries]

        # Perform the searches asynchronously
        try:
            search_results = await self.web_search_tool.async_search(query_strings)
        except Exception as e:
            logger.error(f"Error during async search: {e}")
            # Fall back to individual searches
            search_results = []
            for query in query_strings:
                result_text = self.web_search_tool._run(query)
                search_results.append(
                    {"query": query, "results": [{"content": result_text}]}
                )

        # Process the search results
        processed_results = self._process_search_results(search_results)

        # Update the current section with search results if applicable
        updated_state = self._update_section_with_results(state, processed_results)

        # Get existing steps and sources or initialize empty lists
        research_steps = state.get("research_steps", [])
        sources = state.get("sources", [])

        # Add results to research steps and sources
        for query in query_strings:
            research_steps.append(f"Searched for: {query}")
        sources.extend([f"Search results for: {query}" for query in query_strings])

        logger.debug(f"Updated research steps ({len(research_steps)} total)")
        logger.debug(f"Updated sources ({len(sources)} total)")

        # Increment search iterations
        search_iterations = state.get("search_iterations", 0) + 1

        # Create a combined result message for the conversation
        result_content = "Completed web searches:\n\n"
        for query, result in zip(query_strings, search_results):
            result_content += f"- Query: {query}\n"
            result_content += f"  Found {len(result.get('results', []))} results\n"

        # Create a tool response message
        tool_response = ToolMessage(content=result_content, name="web_search")

        # Return updated state
        updated_state.update(
            {
                "messages": state["messages"] + [tool_response],
                "research_steps": research_steps,
                "sources": sources,
                "search_results": processed_results,
                "search_iterations": search_iterations,
            }
        )

        return updated_state

    def _process_search_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw search results into a more usable format.

        Args:
            results: Raw search results from the web search tool

        Returns:
            Processed results
        """
        processed = {}

        for result in results:
            query = result.get("query", "unknown")
            items = result.get("results", [])

            # Format the content from each result
            formatted_items = []
            for item in items:
                formatted_item = {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "full_content": item.get("raw_content", item.get("content", "")),
                }
                formatted_items.append(formatted_item)

            processed[query] = formatted_items

        return processed

    def _update_section_with_results(
        self, state: AgentState, results: Dict[str, Any]
    ) -> AgentState:
        """Update the current section with search results.

        Args:
            state: Current agent state
            results: Processed search results

        Returns:
            Updated state
        """
        # Make a copy of the state to update
        updated_state = state.copy()

        # If no current section, just return the state
        current_section_index = state.get("current_section_index")
        if current_section_index is None:
            return updated_state

        # Get the sections
        sections = state.get("sections", [])
        if current_section_index >= len(sections):
            logger.warning("Current section index out of bounds")
            return updated_state

        # We don't modify the section directly here - that will be done in the writer node
        # Just return the updated state with the search results
        return updated_state
