#!/usr/bin/env python3
"""
Main entry point for the research agent application.
This implements a modular LangGraph-based research agent.
"""
import os
import argparse
from dotenv import load_dotenv

from src.agent.graph import run_agent

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

    args = parser.parse_args()

    if args.interactive or args.query is None:
        # Interactive mode
        print("Research Agent started!")
        user_query = input("Enter your research query: ")
    else:
        # Command line mode
        user_query = args.query

    research_results = run_agent(user_query)
    print(research_results)


if __name__ == "__main__":
    main()
