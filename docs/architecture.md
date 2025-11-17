# CodeAnalyzer Architecture

This document provides a comprehensive overview of the CodeAnalyzer system architecture, component design, and data flow.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Module Details](#module-details)
- [Data Flow](#data-flow)
- [Security Model](#security-model)
- [Configuration System](#configuration-system)
- [Session Management](#session-management)
- [Extension Points](#extension-points)

## System Overview

CodeAnalyzer is a modular code understanding system that combines:
- **Static Code Analysis**: Search commands (grep, find, ripgrep)
- **Vector Databases**: Semantic search with ChromaDB
- **Large Language Models**: Natural language understanding with OpenAI
- **RAG (Retrieval-Augmented Generation)**: Context-aware responses

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│                     (codeanalyzer.cli)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    LLM Integration                           │
│            (codeanalyzer.llm.system)                         │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Command Gen      │  │  Answer Gen      │                │
│  │ (LLM)            │  │  (LLM + RAG)     │                │
│  └──────────────────┘  └──────────────────┘                │
└───────────────────────┬─────────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
┌───▼────────────┐  ┌──▼───────────┐  ┌───▼──────────────┐
│ Command        │  │  Session     │  │  Storage         │
│ Executor       │  │  Manager     │  │  Manager         │
│ (core.exec)    │  │  (core.sess) │  │  (storage.mgr)   │
└────────────────┘  └──────────────┘  └──────────────────┘
    │                   │                   │
    │                   ▼                   │
    │          ┌──────────────────┐         │
    └─────────►│  Vector Store    │◄────────┘
               │  (ChromaDB)      │
               └──────────────────┘
```

## Component Architecture

### 1. Core Components (`codeanalyzer/core/`)

#### Command Parser (`parser.py`)
**Responsibility**: Parse command outputs into file paths

```python
class CommandOutputParser:
    """Parses grep/find/rg outputs into Path objects."""

    def parse(command: str, output: str) -> List[Path]
```

**Features**:
- Supports grep, find, and ripgrep output formats
- Handles relative and absolute paths
- Deduplicates results
- Filters binary files and invalid paths

#### Command Executor (`executor.py`)
**Responsibility**: Safely execute search commands

```python
class SafeCommandExecutor:
    """Execute commands with strict security validation."""

    async def execute_async(command: str) -> List[Path]
    def validate_command(command: str) -> Tuple[bool, str]
```

**Security Features**:
- ✅ Shell injection prevention (no `shell=True`)
- ✅ Command whitelist (grep, find, rg only)
- ✅ Flag validation
- ✅ Path sanitization
- ✅ Max depth limits
- ✅ Async execution with timeouts

**Validation Rules**:
```python
ALLOWED_COMMANDS = {
    "grep": {
        "allowed_flags": ["-r", "-H", "-n", "-i", "-E", "-l"],
        "required_flags": []
    },
    "find": {
        "allowed_flags": ["-name", "-type", "-maxdepth"],
        "max_depth": 3
    },
    "rg": {
        "allowed_flags": ["--files", "-l", "-n"]
    }
}
```

#### Session (`session.py`)
**Responsibility**: Manage analysis context

```python
class CodeSession:
    """Isolated context for code analysis."""

    session_id: str
    context_files: Set[Path]
    vector_store: Chroma
    embeddings: HuggingFaceEmbeddings
    last_accessed: datetime
```

**Features**:
- File chunking (configurable size with overlap)
- Binary file detection
- Vector store integration
- Async file processing
- LRU-based file management

**File Chunking Strategy**:
```
File (5000 chars)
├─ Chunk 1: [0:1000]     (1000 chars)
├─ Chunk 2: [800:1800]   (1000 chars, 200 overlap)
├─ Chunk 3: [1600:2600]  (1000 chars, 200 overlap)
└─ Chunk 4: [2400:3400]  (1000 chars, 200 overlap)
                           ...
```

### 2. Storage Layer (`codeanalyzer/storage/`)

#### Session Manager (`manager.py`)
**Responsibility**: Multi-session lifecycle management

```python
class SessionManager:
    """Manage multiple code analysis sessions."""

    sessions: Dict[str, CodeSession]
    active_session: Optional[CodeSession]
    max_sessions: int = 10
```

**Features**:
- Session creation and switching
- Persistence to disk (JSON + pickle)
- Auto-loading of recent sessions
- Background cleanup thread
- Thread-safe operations (locks)
- Session expiration (24 hours inactive)

**Persistence Format**:
```
sessions/
├── session-abc123/
│   ├── meta.json         # Metadata (id, created, accessed)
│   ├── context.json      # File list
│   └── vector_store/     # ChromaDB data
│       ├── chroma.sqlite3
│       └── ...
└── session-xyz789/
    └── ...
```

### 3. LLM Integration (`codeanalyzer/llm/`)

#### Code Understanding System (`system.py`)
**Responsibility**: Orchestrate LLM interactions

```python
class CodeUnderstandingSystem:
    """Main system coordinating all components."""

    def generate_search_commands(query: str) -> List[str]
    def execute_search(commands: List[str]) -> List[Path]
    def ask(query: str, k: int = 5) -> str
```

**LLM Workflows**:

1. **Command Generation**:
   ```
   User Query → LLM Prompt → Search Commands
   "Find auth code" → ["grep -rHn authentication",
                       "find . -name *auth*.py"]
   ```

2. **Answer Generation**:
   ```
   User Query → Vector Search → Context Retrieval → LLM → Answer
   "How does auth work?" → [file1.py:10-50, file2.py:100-120]
                         → LLM synthesis → "Authentication uses..."
   ```

### 4. CLI Layer (`codeanalyzer/cli.py`)

**Responsibility**: User interface

```python
@click.group()
def cli():
    """CodeAnalyzer CLI."""

Commands:
├── ask              # Ask questions
├── session          # Session management
│   ├── new         # Create session
│   ├── switch      # Switch session
│   ├── list        # List sessions
│   └── clean       # Clean old sessions
└── config          # Configuration
```

**Click Features**:
- Progress bars for long operations
- Colored output
- Error handling with helpful hints
- Obj passing for state management

### 5. Configuration (`codeanalyzer/config.py`)

**Responsibility**: System configuration

```python
class Config:
    """Configuration management."""

    @property
    def allowed_commands() -> Dict
    @property
    def max_file_size() -> int
    @property
    def excluded_dirs() -> List[str]
    @property
    def log_level() -> str
```

**Configuration Sources** (priority order):
1. Environment variables (`CODEANALYZER_*`)
2. Config file (`~/.config/codeanalyzer/config.json`)
3. Defaults

**Logging Configuration**:
- RotatingFileHandler (10MB, 5 backups)
- Console and file output
- Configurable levels
- Structured logging

### 6. Exception Hierarchy (`codeanalyzer/exceptions.py`)

```
CodeAnalyzerError (base)
├── CommandValidationError
├── CommandExecutionError
├── SessionNotFoundError
├── FileProcessingError
├── ConfigurationError
└── LLMError
```

## Data Flow

### Query Processing Flow

```
1. User Input
   └─► "Find authentication code"

2. CLI Command
   └─► codeanalyzer.cli.ask()

3. LLM Command Generation
   └─► system.generate_search_commands()
       └─► OpenAI LLM generates:
           ["grep -rHn authentication",
            "find . -name *auth*.py -maxdepth 3"]

4. Command Validation
   └─► executor.validate_command()
       └─► Check whitelist, flags, paths
       └─► ✓ Valid / ✗ Reject

5. Command Execution
   └─► executor.execute_async()
       └─► subprocess.run(..., shell=False)
       └─► Returns: [Path("auth.py"), Path("middleware/auth.py")]

6. Output Parsing
   └─► parser.parse(command, output)
       └─► Deduplicate, convert to Path objects

7. File Processing
   └─► session.add_files(found_files)
       └─► Read files
       └─► Chunk into segments
       └─► Generate embeddings
       └─► Store in ChromaDB

8. Context Retrieval
   └─► session.query_context(query, k=5)
       └─► Vector similarity search
       └─► Returns top-k relevant chunks

9. Answer Generation
   └─► system.ask(query)
       └─► Build prompt with context
       └─► LLM generates answer
       └─► Return to user

10. Response Display
    └─► CLI formats and displays answer
```

## Security Model

### Defense in Depth

1. **Input Validation** (CLI Layer)
   - Query sanitization
   - Parameter validation

2. **Command Whitelisting** (Executor)
   - Only grep/find/rg allowed
   - Strict flag validation
   - No shell metacharacters

3. **Shell Injection Prevention**
   - `shlex.split()` for command parsing
   - `shell=False` in subprocess
   - No user input in shell strings

4. **Path Sanitization**
   - Resolve symlinks
   - Block traversal attempts
   - Limit to project directory

5. **Resource Limits**
   - Max file size (10MB default)
   - Max search depth (3 levels)
   - Session timeout (24 hours)
   - Max sessions (10 concurrent)

6. **Error Handling**
   - Specific exceptions (no broad catches)
   - No sensitive info in errors
   - Proper logging

### Threat Model

**Protected Against**:
- ✅ Shell injection
- ✅ Path traversal
- ✅ Resource exhaustion
- ✅ Command injection
- ✅ Symlink attacks

**Out of Scope**:
- Code execution in analyzed files
- LLM prompt injection
- API key exposure

## Configuration System

### Environment Variables

```bash
OPENAI_API_KEY              # Required - OpenAI API key
CODEANALYZER_CONFIG         # Config file path
CODEANALYZER_LOG_DIR        # Log directory
CODEANALYZER_LOG_LEVEL      # Log level (DEBUG/INFO/WARNING/ERROR)
CODEANALYZER_SESSION_DIR    # Session storage directory
```

### Config File Structure

```json
{
  "max_file_size": 10485760,
  "log_level": "INFO",
  "excluded_dirs": [".git", "node_modules", ".venv"],
  "excluded_extensions": [".pyc", ".pyo", ".so"],
  "allowed_commands": {
    "grep": {"allowed_flags": ["-r", "-H", "-n"]},
    "find": {"allowed_flags": ["-name", "-type"], "max_depth": 3},
    "rg": {"allowed_flags": ["--files", "-l"]}
  }
}
```

## Session Management

### Session Lifecycle

```
CREATE → ACTIVE → INACTIVE → PERSISTED → LOADED → ACTIVE
  ↓                                         ↑
  └─────── AUTO-CLEANUP (24h) ─────────────┘
```

### Session Operations

1. **Create**:
   ```python
   session = manager.create_session("my-session", chunk_size=1000)
   ```

2. **Switch**:
   ```python
   manager.switch_session("my-session")
   ```

3. **Persist**:
   ```python
   manager.persist_session(session)
   # Saves to: sessions/{session_id}/
   ```

4. **Load**:
   ```python
   session = manager.load_session("my-session")
   ```

5. **Cleanup**:
   ```python
   # Automatic: runs every hour
   # Manual: manager.cleanup_old_sessions(days=30)
   ```

### Concurrency Model

- **Thread-safe**: Uses `threading.Lock` for session dict
- **Async I/O**: File operations use asyncio
- **Background cleanup**: Separate thread for cleanup
- **No race conditions**: Lock-protected session access

## Extension Points

### 1. Custom LLM Providers

```python
# Future: Abstract LLM interface
class LLMProvider(ABC):
    @abstractmethod
    async def generate_commands(query: str) -> List[str]:
        pass

    @abstractmethod
    async def answer(query: str, context: str) -> str:
        pass

# Implementations:
# - OpenAIProvider (current)
# - AnthropicProvider (future)
# - LocalLLMProvider (future)
```

### 2. Custom Vector Stores

```python
# Future: Abstract vector store
class VectorStore(ABC):
    @abstractmethod
    def add_documents(documents: List[Document]):
        pass

    @abstractmethod
    def similarity_search(query: str, k: int) -> List[Document]:
        pass

# Implementations:
# - ChromaDBStore (current)
# - PineconeStore (future)
# - WeaviateStore (future)
```

### 3. Custom Commands

```python
# Extend ALLOWED_COMMANDS in config
config.set("allowed_commands", {
    "ag": {  # The Silver Searcher
        "allowed_flags": ["-l", "--files"]
    }
})
```

### 4. Custom Embeddings

```python
# Future: Configurable embedding models
config.set("embedding_model", "all-mpnet-base-v2")
```

## Performance Considerations

### Optimization Strategies

1. **Caching**:
   - LLM responses (future)
   - Embedding generation (future)
   - Command results (future)

2. **Async Operations**:
   - File I/O is async
   - Command execution is async
   - Vector operations are sync (library limitation)

3. **Chunking Strategy**:
   - Configurable chunk size (default: 1000)
   - Overlap for context (default: 200)
   - Binary file detection (skip)

4. **Resource Limits**:
   - Max file size: 10MB
   - Max sessions: 10
   - Max search depth: 3

### Scalability

**Current Limits**:
- Single machine deployment
- Local vector store
- Synchronous LLM calls

**Future Improvements**:
- Distributed vector store
- Batch processing
- Streaming responses
- Horizontal scaling

## Monitoring and Observability

### Logging

```python
# Structured logging with context
logger.info("Session created", extra={
    "session_id": session.session_id,
    "chunk_size": session.chunk_size
})
```

### Metrics (Future)

```python
# Potential metrics to track
- queries_per_session
- average_response_time
- files_processed_count
- vector_store_size
- session_duration
```

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [LangChain Documentation](https://python.langchain.com)
- [Click Documentation](https://click.palletsprojects.com)
