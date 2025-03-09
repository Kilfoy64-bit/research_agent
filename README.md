# Research Agent

A general purpose research agent built with LangChain and LangGraph to help with information gathering, summarization, and analysis.

## Features

- Web search capabilities using Tavily API
- Scientific literature search with ArXiv
- Memory-based conversation history
- Advanced research workflow with LangGraph
- Modular and extensible architecture

## Getting Started

### Prerequisites

- Python 3.11+
- UV package manager (recommended) or pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/research_agent.git
   cd research_agent
   ```

2. Set up environment with UV:
   ```
   uv venv
   uv pip install -e .
   ```

3. Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

### Usage

Run the research agent:

```
python main.py
```

This will start an interactive console where you can ask research questions.

## Project Structure

```
research_agent/
├── .env                # Environment variables (not in version control)
├── .gitignore          # Git ignore file
├── main.py             # Main entry point
├── pyproject.toml      # Project metadata and dependencies
├── README.md           # Project documentation
├── src/                # Source code
│   └── research_agent/ # Main package
│       ├── __init__.py # Package initialization
│       ├── agent.py    # Agent implementation
│       └── tools/      # Custom tools for the agent
```

## Development

This project uses UV for dependency management. To add new dependencies:

```
uv pip install package_name
```

Then update `pyproject.toml` to include the new dependency.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for stateful agent workflow capabilities
