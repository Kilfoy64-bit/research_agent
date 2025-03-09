# Research Agent

A modular LangGraph-based research agent.

## Project Structure

The project is organized into several modules:

```
src/
├── agent/           # Agent-specific code
│   ├── graph.py     # Agent graph configuration and execution
│   └── nodes.py     # Graph node implementations
├── models/          # LLM configuration
│   └── llm.py       # LLM setup and mock implementations
├── state/           # State management
│   └── state.py     # Agent state definition
├── tools/           # Tools used by the agent
│   └── web_search.py # Web search tool implementation
└── main.py          # Main entry point
```

## Setup

1. Create a virtual environment:
```bash
uv venv
```

2. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
# Install the project with main dependencies
uv pip install -e .

# Install with OpenAI support
uv pip install ".[openai]"

# Install with development tools
uv pip install ".[dev]"

# Install with both OpenAI and development dependencies
uv pip install ".[openai,dev]"
```

4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the agent from the command line:

```bash
python -m src.main
```

## Extending the Agent

### Adding New Tools

1. Create a new tool in the `tools` directory
2. Import and add the tool to the tools list in `agent/graph.py`
3. Add a new node function in `agent/nodes.py` to handle the tool
4. Update the routing in `agent/nodes.py` to route to your new tool

### Modifying the Agent Graph

To modify the agent's workflow, edit the `build_research_agent` function in `agent/graph.py`.

## Development

Install development dependencies:

```bash
uv pip install ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black src
```

Run linter:

```bash
ruff check src
```
