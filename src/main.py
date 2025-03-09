#!/usr/bin/env python3
"""
Main entry point for the research agent application.
This implements a modular LangGraph-based research agent.
"""
import os
from dotenv import load_dotenv

from src.agent.graph import run_agent

# Load environment variables
load_dotenv()


def main() -> None:
    """Main entry point for the research agent."""
    print("Research Agent started!")
    user_query = input("Enter your research query: ")
    research_results = run_agent(user_query)
    print(research_results)


if __name__ == "__main__":
    main()
