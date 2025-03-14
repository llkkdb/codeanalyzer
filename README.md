# Code Analyzer

An intelligent code understanding system for exploring and analyzing codebases using natural language queries.

## Features

- **Natural Language Code Search**: Ask questions about your codebase in plain English
- **Context-Aware Responses**: Maintains session context for more relevant answers
- **Safe Command Execution**: Secure execution of system commands with strict validation
- **Session Management**: Create and switch between multiple isolated analysis sessions
- **Persistent Knowledge**: Save and restore sessions for continued analysis

## Installation

### Prerequisites

- Python 3.9+
- Git (for cloning the repository)

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

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

```
# Core dependencies
click==8.1.7
langchain-core==0.1.52
langchain-community==0.0.38
langchain-openai==0.1.6
chromadb>=0.4.18
openai>=1.24.0

# Supporting libraries
pyyaml==6.0.2
sqlalchemy==2.0.39
aiohttp==3.11.13
sentence-transformers>=2.2.2
```

## Configuration

1. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

2. (Optional) Create a custom knowledge guidelines file:
   ```bash
   echo "# Custom Knowledge Guidelines" > knowledge_guidelines.md
   ```

## Usage

### Basic Commands

Start a new session:
```bash
python code_understanding.py session new
```

Ask a question about your code:
```bash
python code_understanding.py ask "How does the error handling work in this codebase?"
```

List all sessions:
```bash
python code_understanding.py session list
```

Switch to an existing session:
```bash
python code_understanding.py session switch session_id
```

### Example Workflow

```bash
# Create a new session
python code_understanding.py session new --name my_analysis

# Ask a question about the code
python code_understanding.py ask "Find all functions related to validation"

# Ask a follow-up question (maintains context)
python code_understanding.py ask "How are these validation functions tested?"
```

## How It Works

1. **Command Generation**: Translates natural language queries into search commands
2. **File Discovery**: Executes safe commands to find relevant files
3. **Context Building**: Adds files to a vector database for semantic search
4. **Knowledge Extraction**: Retrieves relevant code snippets from the vector store
5. **Response Generation**: Uses LLMs to generate answers based on the code context

## Security Features

- Allowlist-based command validation
- Strict flag validation for system commands 
- Path depth restrictions
- Timeout limits for command execution
- Secure parsing of command outputs

## Development

### Running Tests

```bash
PYTHONPATH=/path/to/codeanalyzer pytest tests/
```

### Project Structure

```
codeanalyzer/
├── code_understanding.py  # Main application code
├── knowledge_guidelines.md  # System prompts for LLMs
├── sessions/  # Stored session metadata
└── chroma_sessions/  # Vector database storage
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure functionality
5. Submit a pull request

## License

[MIT License](LICENSE)