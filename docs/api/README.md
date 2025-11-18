# API Reference

This directory contains API reference documentation for the CodeAnalyzer package.

## Modules

### Core Modules

#### `codeanalyzer.CodeUnderstandingSystem`
Main entry point for the code understanding system.

**Methods:**
- `ask(question: str, session_id: Optional[str] = None) -> str` - Ask a question about the codebase
- `create_session(session_id: str) -> CodeSession` - Create a new analysis session
- `load_session(session_id: str) -> CodeSession` - Load an existing session
- `list_sessions() -> List[str]` - List all available sessions

**Example:**
```python
from codeanalyzer import CodeUnderstandingSystem

system = CodeUnderstandingSystem()
session = system.create_session("my-project")
answer = system.ask("What does the authentication module do?", "my-project")
print(answer)
```

---

#### `codeanalyzer.config.Config`
Configuration management for CodeAnalyzer.

**Properties:**
- `model_name: str` - Name of the LLM model to use (default: "gpt-3.5-turbo")
- `chunk_size: int` - Size of code chunks for processing (default: 500)
- `max_files: int` - Maximum number of files to process (default: 100)
- `log_level: str` - Logging level (default: "INFO")
- `storage_dir: Path` - Directory for session storage

**Methods:**
- `load() -> None` - Load configuration from file
- `save() -> None` - Save configuration to file
- `setup_logging() -> None` - Configure logging based on settings

**Example:**
```python
from codeanalyzer.config import Config

config = Config()
config.model_name = "gpt-4"
config.chunk_size = 1000
config.save()
```

---

### Session Management

#### `codeanalyzer.core.session.CodeSession`
Represents an analysis session for a codebase.

**Properties:**
- `session_id: str` - Unique identifier for the session
- `files: Set[str]` - Set of file paths in the session
- `metadata: Dict[str, Any]` - Session metadata
- `created_at: datetime` - Session creation timestamp
- `updated_at: datetime` - Last update timestamp

**Methods:**
- `add_file(file_path: str) -> None` - Add a file to the session
- `remove_file(file_path: str) -> None` - Remove a file from the session
- `get_context() -> str` - Get session context summary

**Example:**
```python
from codeanalyzer import CodeUnderstandingSystem

system = CodeUnderstandingSystem()
session = system.create_session("my-project")

# Add files to session
session.add_file("/path/to/src/main.py")
session.add_file("/path/to/src/utils.py")

# Get session info
print(f"Session has {len(session.files)} files")
print(f"Created at: {session.created_at}")
```

---

#### `codeanalyzer.storage.manager.SessionManager`
Manages session persistence and lifecycle.

**Methods:**
- `create_session(session_id: str) -> CodeSession` - Create new session
- `load_session(session_id: str) -> Optional[CodeSession]` - Load existing session
- `persist_session(session: CodeSession) -> None` - Save session to storage
- `delete_session(session_id: str) -> None` - Delete a session
- `list_sessions() -> List[str]` - List all session IDs
- `cleanup_old_sessions(days: int = 30) -> int` - Remove old sessions

**Example:**
```python
from codeanalyzer.storage.manager import SessionManager
from pathlib import Path

manager = SessionManager(storage_dir=Path("./sessions"))

# Create and persist session
session = manager.create_session("my-project")
session.add_file("main.py")
manager.persist_session(session)

# Load session later
loaded = manager.load_session("my-project")
```

---

### File Processing

#### `codeanalyzer.file_processor.FileProcessor`
Processes code files for analysis.

**Methods:**
- `is_text_file(file_path: Path) -> bool` - Check if file is text
- `chunk_code(content: str, chunk_size: int = 500) -> List[str]` - Split code into chunks
- `extract_imports(content: str) -> List[str]` - Extract import statements
- `detect_language(file_path: Path) -> str` - Detect programming language

**Example:**
```python
from codeanalyzer.file_processor import FileProcessor
from pathlib import Path

processor = FileProcessor()

# Check if file is text
if processor.is_text_file(Path("code.py")):
    content = Path("code.py").read_text()

    # Chunk the code
    chunks = processor.chunk_code(content, chunk_size=500)
    print(f"Split into {len(chunks)} chunks")

    # Extract imports
    imports = processor.extract_imports(content)
    print(f"Found imports: {imports}")
```

---

### Command Execution

#### `codeanalyzer.core.executor.SafeCommandExecutor`
Safely executes shell commands with validation.

**Methods:**
- `execute_grep(pattern: str, path: str) -> str` - Execute grep command
- `execute_find(pattern: str, path: str) -> str` - Execute find command
- `execute_ripgrep(pattern: str, path: str) -> str` - Execute ripgrep command
- `is_safe_command(command: List[str]) -> bool` - Validate command safety

**Example:**
```python
from codeanalyzer.core.executor import SafeCommandExecutor

executor = SafeCommandExecutor()

# Search for pattern
results = executor.execute_grep("class.*Test", "/path/to/codebase")
print(results)

# Find files
files = executor.execute_find("*.py", "/path/to/codebase")
print(files)
```

---

#### `codeanalyzer.core.parser.CommandOutputParser`
Parses output from shell commands.

**Methods:**
- `parse_grep_output(output: str) -> List[Dict[str, Any]]` - Parse grep results
- `parse_find_output(output: str) -> List[str]` - Parse find results
- `parse_ripgrep_output(output: str) -> List[Dict[str, Any]]` - Parse rg results

**Example:**
```python
from codeanalyzer.core.parser import CommandOutputParser

parser = CommandOutputParser()

# Parse grep output
grep_output = "file.py:10:def test():"
results = parser.parse_grep_output(grep_output)
# Results: [{'file': 'file.py', 'line': 10, 'content': 'def test():'}]
```

---

### Exceptions

#### `codeanalyzer.exceptions`
Custom exception hierarchy for error handling.

**Exception Classes:**
- `CodeAnalyzerError` - Base exception for all errors
- `ConfigurationError` - Configuration-related errors
- `SessionError` - Session management errors
- `SessionNotFoundError` - Session doesn't exist
- `CommandExecutionError` - Command execution failures
- `FileProcessingError` - File processing errors
- `ParsingError` - Output parsing errors

**Example:**
```python
from codeanalyzer.exceptions import SessionNotFoundError, ConfigurationError

try:
    session = system.load_session("nonexistent")
except SessionNotFoundError as e:
    print(f"Session not found: {e}")

try:
    config = Config(invalid_option=True)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

---

## CLI Commands

### `codeanalyzer ask`
Ask questions about your codebase.

```bash
codeanalyzer ask "What does the authentication module do?" --session my-project
```

**Options:**
- `--session, -s` - Session ID to use
- `--context, -c` - Additional context files
- `--model, -m` - LLM model to use

---

### `codeanalyzer session`
Manage analysis sessions.

```bash
# Create session
codeanalyzer session create my-project

# List sessions
codeanalyzer session list

# Delete session
codeanalyzer session delete my-project

# Show session info
codeanalyzer session info my-project
```

---

### `codeanalyzer config`
Manage configuration.

```bash
# Show config
codeanalyzer config show

# Set value
codeanalyzer config set model_name gpt-4

# Get value
codeanalyzer config get model_name
```

---

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API key (required)
- `CODEANALYZER_CONFIG` - Path to config file (optional)
- `CODEANALYZER_STORAGE_DIR` - Session storage directory (optional)

---

## Type Hints

CodeAnalyzer uses type hints throughout the codebase:

```python
from typing import Optional, List, Dict, Any
from pathlib import Path

def process_files(
    file_paths: List[Path],
    chunk_size: int = 500,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Process multiple files."""
    ...
```

---

## Best Practices

### Session Management
```python
# Use context manager for automatic cleanup
with system.session_manager.create_session("temp") as session:
    session.add_file("main.py")
    system.ask("What does this do?", "temp")
# Session automatically saved
```

### Error Handling
```python
from codeanalyzer.exceptions import CodeAnalyzerError

try:
    answer = system.ask("question", "my-session")
except CodeAnalyzerError as e:
    logger.error(f"Analysis failed: {e}")
    # Handle error appropriately
```

### Performance
```python
# Process files in batches
from pathlib import Path

files = list(Path("src").rglob("*.py"))
for i in range(0, len(files), 10):
    batch = files[i:i+10]
    for file in batch:
        session.add_file(str(file))
```

---

## Further Reading

- [Architecture Documentation](../architecture.md)
- [Examples Guide](../examples.md)
- [Troubleshooting](../troubleshooting.md)
- [Deployment Guide](../deployment.md)
