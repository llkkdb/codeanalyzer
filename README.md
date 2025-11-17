# Code Analyzer

An intelligent code understanding system for exploring and analyzing codebases using natural language queries.

## Features

- **Natural Language Code Search**: Ask questions about your codebase in plain English
- **Context-Aware Responses**: Maintains session context for more relevant answers using RAG
- **Safe Command Execution**: Secure execution with shell injection protection and strict validation
- **Session Management**: Create and switch between multiple isolated analysis sessions with persistence
- **Modular Architecture**: Clean separation of concerns with well-organized modules
- **Comprehensive Error Handling**: Custom exception classes for better error reporting
- **Configurable**: File-based and environment variable configuration support
- **Production-Ready Logging**: Rotating log files with configurable levels

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (get one from [OpenAI](https://platform.openai.com/api-keys))

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/codeanalyzer.git
   cd codeanalyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

   Or install with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Configuration

1. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. (Optional) Configure system settings:
   ```bash
   codeanalyzer config --list-exclude
   codeanalyzer config --add-exclude .cache
   codeanalyzer config --max-file-size 2097152  # 2MB
   ```

## Usage

### Creating a Session

```bash
# Create a new session
codeanalyzer session new --name my-project

# Create with custom chunk size
codeanalyzer session new --name my-project --chunk-size 1500
```

### Asking Questions

```bash
# Ask a question about your codebase
codeanalyzer ask "How does authentication work in this project?"

# Retrieve more context documents
codeanalyzer ask "Where are API endpoints defined?" --k 10
```

### Managing Sessions

```bash
# List all sessions
codeanalyzer session list

# Switch to a different session
codeanalyzer session switch my-other-project

# Clean up old sessions (>30 days)
codeanalyzer session clean
```

### Example Workflow

```bash
# Create a new session
codeanalyzer session new --name my_analysis

# Ask a question about the code
codeanalyzer ask "Find all functions related to validation"

# Ask a follow-up question (maintains context)
codeanalyzer ask "How are these validation functions tested?"
```

## How It Works

1. **Command Generation**: Translates natural language queries into search commands
2. **File Discovery**: Executes safe commands to find relevant files
3. **Context Building**: Adds files to a vector database for semantic search
4. **Knowledge Extraction**: Retrieves relevant code snippets from the vector store
5. **Response Generation**: Uses LLMs to generate answers based on the code context

## Architecture

CodeAnalyzer is organized into modular components:

```
codeanalyzer/
├── core/           # Command execution and parsing
│   ├── parser.py   # Output parsing
│   ├── executor.py # Safe command execution
│   └── session.py  # Session management
├── storage/        # Persistence layer
│   └── manager.py  # Session storage
├── llm/            # LLM integration
│   └── system.py   # Code understanding system
├── config.py       # Configuration management
├── exceptions.py   # Custom exceptions
└── cli.py          # Command-line interface
```

## Security Features

- **Command Allowlist**: Only `find`, `grep`, and `rg` commands permitted
- **Flag Validation**: Strict validation of command flags
- **Path Restrictions**: Prevents searching sensitive directories
- **Shell Injection Protection**: Uses `shlex.split()` and `shell=False`
- **Size Limits**: Files >1MB excluded by default
- **Timeout Protection**: 10-second default timeout for commands
- **Environment Validation**: Checks for required API keys at startup

## Development

### Setting Up Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=codeanalyzer --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black codeanalyzer tests

# Lint code
ruff check codeanalyzer tests

# Type check
mypy codeanalyzer
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### ModuleNotFoundError

If you encounter import errors:
```bash
pip install -r requirements.txt
```

### OPENAI_API_KEY not set

Export your API key:
```bash
export OPENAI_API_KEY='your-key'
```

### Permission Errors

Ensure you have read/write permissions in the working directory for session storage and logs.

## Support

- Report bugs: [GitHub Issues](https://github.com/yourusername/codeanalyzer/issues)
- Documentation: [README.md](README.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)