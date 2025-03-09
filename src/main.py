#!/usr/bin/env python3
"""
Main entry point for the research agent application.
This implements a modular LangGraph-based research agent.
"""
import os
import argparse
import logging
from dotenv import load_dotenv

from src.agent.graph import run_agent
from src.utils.logging import configure_logging, get_logger, get_log_level_from_env

# Configure logging
configure_logging(log_level=get_log_level_from_env())
# Get logger for this module
logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def main() -> None:
    """Main entry point for the research agent."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Research Agent - A modular LangGraph-based research agent"
    )
    parser.add_argument("query", nargs="?", help="Research query to process")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=get_log_level_from_env(),
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Configure logging with command line arguments
    configure_logging(log_level=args.log_level)

    if args.interactive or args.query is None:
        # Interactive mode
        logger.info("Research Agent started!")
        user_query = input("Enter your research query: ")
    else:
        # Command line mode
        user_query = args.query
        logger.info("Processing query: %s", user_query)

    research_results = run_agent(user_query)
    logger.info("Research complete")
    print(research_results)


if __name__ == "__main__":
    main()
