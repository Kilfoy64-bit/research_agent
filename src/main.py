#!/usr/bin/env python3
import os
import argparse
import logging
import sys
from dotenv import load_dotenv

from src.agent.graph import run_agent
from src.utils.logging import configure_logging, get_logger, get_log_level_from_env


def main() -> None:
    """Main entry point for the research agent."""
    # Load environment variables with explicit path and override
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        print(f"Error: .env file not found at {env_path}")
        print("Please create a .env file with your API keys.")
        sys.exit(1)

    load_dotenv(env_path, override=True)

    # Verify OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set in .env file.")
        print("Please add your OpenAI API key to the .env file.")
        sys.exit(1)

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

    # Get logger for this module
    logger = get_logger(__name__)

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
