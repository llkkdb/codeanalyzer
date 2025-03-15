"""
Improved Code Understanding System with better performance
and file management capabilities.
"""
import os
import re
import subprocess
import uuid
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
import click
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("code_system.log"), logging.StreamHandler()]
)
logger = logging.getLogger("code_system")

# Security configuration
ALLOWED_COMMANDS = {
    "find": {"flags": ["-name", "-type", "-path", "-exec", "-maxdepth"], "max_depth": 3},
    "grep": {"flags": ["-r", "-i", "-l", "-H", "-n"], "max_depth": None},
    "rg": {"flags": ["--files", "-g", "-l"], "max_depth": 3}
}

# File processing configuration
MAX_FILE_SIZE = 1024 * 1024  # 1 MB
EXCLUDED_DIRS = [".git", "node_modules", "__pycache__", "venv", ".idea", ".vscode"]
EXCLUDED_EXTENSIONS = [".jpg", ".png", ".gif", ".bin", ".exe", ".dll", ".so", ".pyc"]
DEFAULT_CHUNK_SIZE = 1000

class CommandOutputParser:
    """Parse command outputs to extract file paths with improved caching"""
    # Compiled regex patterns - do this once at module level for performance
    PATTERNS = {
        "grep": re.compile(r"^(.*?):\d+:"),  # Extract path from grep -Hn output
        "find": None,                        # Direct path output
        "rg": re.compile(r"^(.*?)(:\d+){2}") # Ripgrep format
    }

    @classmethod
    def parse(cls, command: str, output: str) -> List[Path]:
        """Parse command output based on command type with better error handling"""
        cmd_type = command.split()[0]
        parser = cls.PATTERNS.get(cmd_type)
        
        paths = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            
            try:
                if parser is None:  # Direct output (like find)
                    path_str = line
                elif isinstance(parser, re.Pattern):
                    match = parser.match(line)
                    if match:
                        path_str = match.group(1)
                    else:
                        continue
                else:  # Fallback for unknown commands
                    path_str = line.split(":")[0]
                
                path = Path(path_str).resolve()
                if path.exists() and path.is_file() and cls._should_include_file(path):
                    paths.append(path)
            except Exception as e:
                logger.debug(f"Error parsing path from '{line}': {e}")
                continue
                
        return paths
    
    @staticmethod
    def _should_include_file(path: Path) -> bool:
        """Check if file should be included based on size and extension"""
        # Skip excluded extensions
        if path.suffix.lower() in EXCLUDED_EXTENSIONS:
            return False
            
        # Skip files in excluded directories
        for excluded in EXCLUDED_DIRS:
            if excluded in path.parts:
                return False
                
        # Skip files that are too large
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                logger.info(f"Skipping file {path}: exceeds size limit")
                return False
        except OSError:
            return False
            
        return True

class SafeCommandExecutor:
    """Validate and execute commands safely with better timeout handling"""
    def __init__(self, timeout: int = 10):
        self.allow_list = ALLOWED_COMMANDS
        self.timeout = timeout
        self._command_cache = {}  # Cache results for repeated commands
        self._cache_lock = threading.Lock()

    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Check if command is allowed with improved security checks"""
        parts = command.split()
        if not parts:
            return False, "Empty command"
        
        cmd = parts[0]
        if cmd not in self.allow_list:
            return False, f"Command {cmd} not allowed"

        # Check flags
        allowed_flags = self.allow_list[cmd]["flags"]
        for part in parts[1:]:
            if part.startswith("-"):
                if cmd == "grep" and len(part) > 1 and part[1] != "-":
                    # Split combined flags like -Hn into [-H, -n]
                    flags = [f"-{char}" for char in part[1:]]
                    for flag in flags:
                        if flag not in allowed_flags:
                            return False, f"Disallowed flag {flag} for {cmd}"
                elif part not in allowed_flags:
                    return False, f"Disallowed flag {part} for {cmd}"

        # Enhanced path and argument validation
        excluded_patterns = ["/etc", "/var", "/usr", "/bin", "/sbin", "~", "$", "|", ";", ">", "<", "&"]
        for pattern in excluded_patterns:
            if pattern in command:
                return False, f"Command contains disallowed pattern: {pattern}"

        # Command-specific validation
        if cmd == "find":
            # Check for root path usage
            if any(p.startswith("/") for p in parts if not p.startswith("-")):
                return False, "Find command cannot search from root directory"
            
            # Verify maxdepth parameter
            max_depth = self.allow_list[cmd]["max_depth"]
            if "-maxdepth" in parts:
                idx = parts.index("-maxdepth") + 1
                if idx < len(parts):
                    if not parts[idx].isdigit():
                        return False, "Invalid maxdepth value"
                    if int(parts[idx]) > max_depth:
                        return False, f"Max depth exceeds {max_depth}"
                else:
                    return False, "Missing maxdepth value"
            else:
                # Automatically add maxdepth to make command safer
                command += f" -maxdepth {max_depth}"
        
        return True, command  # Return potentially modified command

    async def execute_async(self, command: str) -> List[Path]:
        """Execute command asynchronously with caching"""
        # Check cache first
        with self._cache_lock:
            cache_key = command
            cache_entry = self._command_cache.get(cache_key)
            if cache_entry:
                cache_time, result = cache_entry
                # Cache valid for 5 minutes
                if time.time() - cache_time < 300:
                    logger.debug(f"Cache hit for command: {command}")
                    return result

        # Validate command
        valid, result = self.validate_command(command)
        if not valid:
            logger.warning(f"Invalid command: {result}")
            return []
            
        validated_command = result
        
        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    validated_command, 
                    shell=True, 
                    check=False,  # Don't raise exception, handle errors manually
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True, 
                    timeout=self.timeout
                )
            )
            
            if result.returncode != 0:
                logger.warning(f"Command failed: {result.stderr}")
                paths = []
            else:
                paths = CommandOutputParser.parse(command, result.stdout)
                
            # Update cache
            with self._cache_lock:
                self._command_cache[cache_key] = (time.time(), paths)
                
            # Clean old cache entries if cache is too large
            if len(self._command_cache) > 100:
                self._clean_cache()
                
            return paths
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {command}")
            return []
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return []
    
    def execute(self, command: str) -> List[Path]:
        """Synchronous wrapper for execute_async"""
        return asyncio.run(self.execute_async(command))
    
    def _clean_cache(self):
        """Remove old cache entries"""
        now = time.time()
        with self._cache_lock:
            old_keys = [k for k, (t, _) in self._command_cache.items() if now - t > 300]
            for k in old_keys:
                del self._command_cache[k]

class CodeSession:
    """Isolated session context with improved file handling"""
    def __init__(self, session_id: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.vector_store = None
        self.context_files: Set[Path] = set()
        self.search_history: List[Dict] = []
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = Path(f"chroma_sessions/{session_id}")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self._file_processing_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._file_metadata = {}  # Store file modification times

    async def add_files_async(self, paths: List[Path]) -> int:
        """Add files to session context with chunking asynchronously"""
        with self._file_processing_lock:
            new_paths = []
            for p in paths:
                if p not in self.context_files:
                    # Check if file has been modified
                    try:
                        mtime = p.stat().st_mtime
                        if p in self._file_metadata and self._file_metadata[p] == mtime:
                            continue  # File hasn't changed
                        self._file_metadata[p] = mtime
                        new_paths.append(p)
                    except OSError:
                        continue
                        
            if not new_paths:
                return 0

            # Process files in thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, self._process_files, new_paths)
            
            if docs:
                if self.vector_store:
                    # Add documents to existing collection
                    await loop.run_in_executor(None, lambda: self.vector_store.add_documents(docs))
                    await loop.run_in_executor(None, lambda: self.vector_store.persist())
                else:
                    # Initialize vector store
                    self.vector_store = await loop.run_in_executor(
                        None,
                        lambda: Chroma.from_documents(
                            docs, 
                            self.embeddings,
                            persist_directory=str(self.persist_directory)
                        )
                    )
                    await loop.run_in_executor(None, lambda: self.vector_store.persist())
                    
            return len(docs)

    def add_files(self, paths: List[Path]) -> int:
        """Synchronous wrapper for add_files_async"""
        return asyncio.run(self.add_files_async(paths))
        
    def _process_files(self, paths: List[Path]) -> List[Document]:
        """Process files into chunks with better error handling"""
        docs = []
        
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
            chunk_results = list(executor.map(self._process_single_file, paths))
            
        for path, chunks in chunk_results:
            if chunks:
                docs.extend(chunks)
                self.context_files.add(path)
                
        return docs
        
    def _process_single_file(self, path: Path) -> Tuple[Path, List[Document]]:
        """Process a single file into document chunks"""
        chunks = []
        try:
            # Skip binary files
            if self._is_binary_file(path):
                logger.info(f"Skipping binary file: {path}")
                return path, []
                
            content = path.read_text(encoding="utf-8", errors="ignore")
            
            # Adjust chunk size based on file type
            chunk_size = self._get_adjusted_chunk_size(path)
            
            # Create overlapping chunks for better context
            overlap = min(200, chunk_size // 4)
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i:i+chunk_size]
                if not chunk_content.strip():
                    continue  # Skip empty chunks
                    
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata={
                        "source": str(path),
                        "chunk": i // (chunk_size - overlap) + 1,
                        "file_type": path.suffix,
                        "last_modified": path.stat().st_mtime
                    }
                ))
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            
        return path, chunks
            
    def _is_binary_file(self, path: Path) -> bool:
        """Check if file is binary"""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk  # Simple heuristic for binary files
        except Exception:
            return True
            
    def _get_adjusted_chunk_size(self, path: Path) -> int:
        """Adjust chunk size based on file type"""
        # Use larger chunks for documentation files
        if path.suffix.lower() in ['.md', '.txt', '.rst']:
            return self.chunk_size * 2
        # Use smaller chunks for code files
        elif path.suffix.lower() in ['.py', '.js', '.java', '.cpp', '.h', '.c']:
            return self.chunk_size
        # Default chunk size for other files
        return self.chunk_size

    async def query_context_async(self, question: str, k: int = 5) -> str:
        """Query session-specific context asynchronously"""
        self.last_accessed = datetime.now()
        
        if not self.vector_store:
            return ""
            
        try:
            with self._query_lock:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self.vector_store.similarity_search(question, k=k)
                )
                
                # Track this query in history
                self.search_history.append({
                    "query": question,
                    "timestamp": datetime.now().isoformat(),
                    "results": [doc.metadata["source"] for doc in results]
                })
                
                return "\n\n".join(
                    f"File: {doc.metadata['source']} (Chunk {doc.metadata['chunk']})\n"
                    f"{doc.page_content[:500]}..." 
                    for doc in results
                )
        except Exception as e:
            logger.error(f"Error querying context: {e}")
            return ""

    def query_context(self, question: str, k: int = 5) -> str:
        """Synchronous wrapper for query_context_async"""
        return asyncio.run(self.query_context_async(question, k))

class SessionManager:
    """Manage multiple code sessions with cleanup and improved persistence"""
    def __init__(self, storage_dir: Path = Path("sessions"), max_sessions: int = 10):
        self.storage_dir = storage_dir
        self.active_session: Optional[CodeSession] = None
        self.sessions: Dict[str, CodeSession] = {}
        self.storage_dir.mkdir(exist_ok=True)
        self.max_sessions = max_sessions
        self._sessions_lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_inactive_sessions,
            daemon=True
        )
        self._cleanup_thread.start()
        
        # Auto-load persisted sessions
        self._auto_load_sessions()

    def create_session(self, session_id: Optional[str] = None, chunk_size: int = DEFAULT_CHUNK_SIZE) -> CodeSession:
        """Create new isolated session with cleanup if needed"""
        with self._sessions_lock:
            session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
            
            # Cleanup if too many sessions
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_oldest_session()
                
            session = CodeSession(session_id, chunk_size=chunk_size)
            self.sessions[session_id] = session
            self.active_session = session
            return session

    def switch_session(self, session_id: str) -> bool:
        """Switch to existing session"""
        with self._sessions_lock:
            if session_id in self.sessions:
                self.active_session = self.sessions[session_id]
                self.active_session.last_accessed = datetime.now()
                return True
            return False

    async def persist_session_async(self, session: CodeSession):
        """Save session to disk asynchronously"""
        session_dir = self.storage_dir / session.session_id
        session_dir.mkdir(exist_ok=True)
        
        if session.vector_store:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: session.vector_store.persist())
        
        meta = {
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "context_files": [str(p) for p in session.context_files],
            "search_history": session.search_history,
            "chroma_directory": str(session.persist_directory),
            "file_metadata": {str(k): v for k, v in session._file_metadata.items()}
        }
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: (session_dir / "meta.json").write_text(json.dumps(meta))
        )

    def persist_session(self, session: CodeSession):
        """Synchronous wrapper for persist_session_async"""
        asyncio.run(self.persist_session_async(session))

    def load_session(self, session_id: str) -> Optional[CodeSession]:
        """Load a previously persisted session"""
        with self._sessions_lock:
            session_dir = self.storage_dir / session_id
            if not session_dir.exists():
                return None
                
            try:
                if not (session_dir / "meta.json").exists():
                    return None
                    
                meta = json.loads((session_dir / "meta.json").read_text())
                session = CodeSession(session_id)
                session.created_at = datetime.fromisoformat(meta["created_at"])
                session.last_accessed = datetime.fromisoformat(meta.get("last_accessed", meta["created_at"]))
                session.context_files = set(Path(p) for p in meta["context_files"] if Path(p).exists())
                session.search_history = meta["search_history"]
                
                # Restore file metadata
                if "file_metadata" in meta:
                    session._file_metadata = {Path(k): v for k, v in meta["file_metadata"].items()}
                
                # Load the Chroma DB if it exists
                chroma_dir = Path(meta.get("chroma_directory", f"chroma_sessions/{session_id}"))
                if chroma_dir.exists():
                    try:
                        session.vector_store = Chroma(
                            persist_directory=str(chroma_dir),
                            embedding_function=session.embeddings
                        )
                    except Exception as e:
                        logger.error(f"Error loading vector store for session {session_id}: {e}")
                        
                self.sessions[session_id] = session
                return session
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                return None
    
    def _auto_load_sessions(self):
        """Auto-load recent sessions at startup"""
        if not self.storage_dir.exists():
            return
            
        try:
            # Get all session directories with meta.json files
            session_dirs = [
                d for d in self.storage_dir.iterdir() 
                if d.is_dir() and (d / "meta.json").exists()
            ]
            
            # Sort by last accessed time
            session_dirs_with_time = []
            for d in session_dirs:
                try:
                    meta = json.loads((d / "meta.json").read_text())
                    last_accessed = datetime.fromisoformat(meta.get("last_accessed", meta["created_at"]))
                    session_dirs_with_time.append((d, last_accessed))
                except Exception:
                    continue
                    
            # Sort by most recently accessed
            session_dirs_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Load most recent sessions up to max_sessions
            for d, _ in session_dirs_with_time[:self.max_sessions]:
                try:
                    self.load_session(d.name)
                except Exception:
                    continue
                    
            # Set active session to most recently used
            if session_dirs_with_time and self.sessions:
                most_recent = session_dirs_with_time[0][0].name
                if most_recent in self.sessions:
                    self.active_session = self.sessions[most_recent]
                    
        except Exception as e:
            logger.error(f"Error auto-loading sessions: {e}")
                
    def _cleanup_inactive_sessions(self):
        """Periodically clean up inactive sessions"""
        while True:
            try:
                time.sleep(60 * 60)  # Check every hour
                
                with self._sessions_lock:
                    now = datetime.now()
                    inactive_sessions = []
                    
                    for session_id, session in list(self.sessions.items()):
                        # If session hasn't been accessed in 24 hours
                        if (now - session.last_accessed).total_seconds() > 24 * 60 * 60:
                            # Persist session before removing
                            self.persist_session(session)
                            inactive_sessions.append(session_id)
                            
                    # Remove inactive sessions
                    for session_id in inactive_sessions:
                        logger.info(f"Cleaning up inactive session: {session_id}")
                        del self.sessions[session_id]
                        
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def _cleanup_oldest_session(self):
        """Remove oldest session when max sessions is reached"""
        if not self.sessions:
            return
            
        oldest_session_id = min(
            self.sessions.items(),
            key=lambda x: x[1].last_accessed
        )[0]
        
        # Don't remove active session
        if self.active_session and oldest_session_id == self.active_session.session_id:
            # Find next oldest
            next_oldest = sorted(
                [s for s_id, s in self.sessions.items() if s_id != self.active_session.session_id],
                key=lambda s: s.last_accessed
            )
            if next_oldest:
                oldest_session_id = next_oldest[0].session_id
            else:
                return  # Only the active session exists
                
        # Persist before removing
        oldest_session = self.sessions[oldest_session_id]
        self.persist_session(oldest_session)
        del self.sessions[oldest_session_id]
        logger.info(f"Removed oldest session {oldest_session_id} due to session limit")

class CodeUnderstandingSystem:
    """Improved system with asynchronous processing and better caching"""
    def __init__(self, llm=None, command_timeout: int = 10):
        self.llm = llm if llm is not None else ChatOpenAI(temperature=0.1)
        self.command_executor = SafeCommandExecutor(timeout=command_timeout)
        self.session_manager = SessionManager()
        self.knowledge_guidelines = self._load_knowledge_guidelines()
        self._command_generation_cache = {}
        self._command_gen_lock = threading.Lock()

    def _load_knowledge_guidelines(self) -> str:
        try:
            return Path("knowledge_guidelines.md").read_text()
        except FileNotFoundError:
            return "# Default Knowledge Guidelines\n"

    async def generate_search_commands_async(self, query: str) -> List[str]:
        """Generate search commands with caching"""
        # Check cache first
        with self._command_gen_lock:
            cache_key = query
            if cache_key in self._command_generation_cache:
                cache_time, commands = self._command_generation_cache[cache_key]
                # Cache valid for 1 hour
                if time.time() - cache_time < 3600:
                    return commands
        
        # Not in cache or expired, generate new commands
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.knowledge_guidelines}\nGenerate Linux commands to find relevant code files. Focus on precision to avoid too many results."),
            ("human", "Query: {query}\nRespond ONLY with commands separated by newlines:")
        ])
        
        try:
            response = await prompt.ainvoke(self.llm, {"query": query})
            commands = [cmd.strip() for cmd in response.content.split("\n") if cmd.strip()]
            
            # Ensure all find commands use maxdepth
            for i, cmd in enumerate(commands):
                if cmd.startswith("find") and "-maxdepth" not in cmd:
                    parts = cmd.split()
                    commands[i] = f"{cmd} -maxdepth 3"
            
            # Cache result
            with self._command_gen_lock:
                self._command_generation_cache[cache_key] = (time.time(), commands)
                
            return commands
        except Exception as e:
            logger.error(f"Error generating commands: {e}")
            return []

    def generate_search_commands(self, query: str) -> List[str]:
        """Synchronous wrapper for generate_search_commands_async"""
        return asyncio.run(self.generate_search_commands_async(query))

    async def execute_search_async(self, commands: List[str]) -> List[Path]:
        """Execute search commands in parallel"""
        found_files = set()
        
        # Execute commands concurrently
        tasks = [self.command_executor.execute_async(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for cmd, result in zip(commands, results):
            if isinstance(result, Exception):
                logger.error(f"Command failed: {cmd} - {result}")
            else:
                found_files.update(result)
                
        return list(found_files)

    def execute_search(self, commands: List[str]) -> List[Path]:
        """Synchronous wrapper for execute_search_async"""
        return asyncio.run(self.execute_search_async(commands))

    async def ask_async(self, query: str, k: int = 5) -> str:
        """Process a query asynchronously"""
        if not self.session_manager.active_session:
            return "No active session! Create one with 'session new'"
            
        # Generate and execute commands in parallel
        commands = await self.generate_search_commands_async(query)
        found_files = await self.execute_search_async(commands)
        
        # Add files to session and get context
        session = self.session_manager.active_session
        await session.add_files_async(found_files)
        context = await session.query_context_async(query, k=k)
        
        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.knowledge_guidelines}\nContext:\n{context}"),
            ("human", "Question: {query}")
        ])
        response = await prompt.ainvoke(self.llm, {"query": query})
        
        # Persist session after use
        await self.session_manager.persist_session_async(session)
        
        return response.content

    def ask(self, query: str, k: int = 5) -> str:
        """Synchronous wrapper for ask_async"""
        return asyncio.run(self.ask_async(query, k))

@click.group()
@click.pass_context
def cli(ctx):
    """Improved Code Understanding System with Sessions"""
    ctx.obj = CodeUnderstandingSystem()

@cli.command()
@click.argument("query")
@click.option("--k", default=5, help="Number of context documents to retrieve")
@click.pass_obj
def ask(system: CodeUnderstandingSystem, query: str, k: int):
    """Ask a question in current session with progress feedback"""
    try:
        if not system.session_manager.active_session:
            click.echo("No active session! Create one with 'session new'")
            return

        with click.progressbar(length=4, label="Processing query") as bar:
            # Generate commands
            commands = system.generate_search_commands(query)
            click.echo(f"Generated commands:\n" + "\n".join(commands))
            bar.update(1)
            
            # Search for files
            found_files = system.execute_search(commands)
            click.echo(f"Found {len(found_files)} relevant files")
            bar.update(1)
            
            # Process files
            session = system.session_manager.active_session
            session.add_files(found_files)
            bar.update(1)
            
            # Generate answer
            answer = system.ask(query, k=k)
            bar.update(1)
            
        click.echo(f"\nAnswer:\n{answer}")
    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error(f"Error in ask command: {e}", exc_info=True)

@cli.group()
def session():
    """Session management commands"""
    pass

@session.command(name="new")
@click.option("--name", default=None, help="Session name")
@click.option("--chunk-size", default=DEFAULT_CHUNK_SIZE, 
              help="Chunk size for document processing")
@click.pass_obj
def new_session(system: CodeUnderstandingSystem, name: Optional[str], chunk_size: int):
    """Create new session with custom parameters"""
    try:
        session_id = name or f"session_{uuid.uuid4().hex[:8]}"
        session = system.session_manager.create_session(session_id)
        session.chunk_size = chunk_size
        click.echo(f"Created new session: {session.session_id}")
    except Exception as e:
        click.echo(f"Error creating session: {e}")

@session.command(name="switch")
@click.argument("session_id")
@click.pass_obj
def switch_session(system: CodeUnderstandingSystem, session_id: str):
    """Switch active session"""
    try:
        if system.session_manager.switch_session(session_id):
            click.echo(f"Switched to session: {session_id}")
        else:
            # Try to load from disk
            session = system.session_manager.load_session(session_id)
            if session:
                system.session_manager.active_session = session
                click.echo(f"Loaded and switched to session: {session_id}")
            else:
                click.echo(f"Session {session_id} not found!")
    except Exception as e:
        click.echo(f"Error switching session: {e}")

@session.command(name="list")
@click.pass_obj
def list_sessions(system: CodeUnderstandingSystem):
    """List all sessions with additional metadata"""
    try:
        click.echo("Active sessions:")
        for session_id, session in system.session_manager.sessions.items():
            active = system.session_manager.active_session and \
                     system.session_manager.active_session.session_id == session_id
            files_count = len(session.context_files)
            last_accessed = session.last_accessed.strftime("%Y-%m-%d %H:%M")
            click.echo(f" {'*' if active else ' '} {session_id} - {files_count} files - Last accessed: {last_accessed}")
        
        # Check disk for persisted sessions
        if system.session_manager.storage_dir.exists():
            persisted = []
            for d in system.session_manager.storage_dir.iterdir():
                if d.is_dir() and (d / "meta.json").exists() and d.name not in system.session_manager.sessions:
                    try:
                        meta = json.loads((d / "meta.json").read_text())
                        created = datetime.fromisoformat(meta["created_at"]).strftime("%Y-%m-%d")
                        last_accessed = datetime.fromisoformat(
                            meta.get("last_accessed", meta["created_at"])
                        ).strftime("%Y-%m-%d")
                        files_count = len(meta.get("context_files", []))
                        persisted.append((d.name, created, last_accessed, files_count))
                    except Exception:
                        persisted.append((d.name, "Unknown", "Unknown", 0))
                        
            if persisted:
                click.echo("\nPersisted sessions (not loaded):")
                for session_id, created, last_accessed, files_count in persisted:
                    click.echo(f"   {session_id} - {files_count} files - Created: {created} - Last accessed: {last_accessed}")
    except Exception as e:
        click.echo(f"Error listing sessions: {e}")

@session.command(name="clean")
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
@click.pass_obj
def clean_sessions(system: CodeUnderstandingSystem, force: bool):
    """Clean up old/unused sessions"""
    try:
        if not force:
            if not click.confirm("This will remove sessions older than 30 days. Continue?"):
                return
                
        storage_dir = system.session_manager.storage_dir
        if not storage_dir.exists():
            click.echo("No sessions to clean")
            return
            
        count = 0
        now = datetime.now()
        for d in storage_dir.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                try:
                    meta = json.loads((d / "meta.json").read_text())
                    last_accessed = datetime.fromisoformat(
                        meta.get("last_accessed", meta["created_at"])
                    )
                    
                    # If older than 30 days and not currently loaded
                    if (now - last_accessed).days > 30 and d.name not in system.session_manager.sessions:
                        # Remove the session directory
                        import shutil
                        shutil.rmtree(d)
                        count += 1
                except Exception as e:
                    click.echo(f"Error cleaning session {d.name}: {e}")
                    
        click.echo(f"Cleaned up {count} old sessions")
    except Exception as e:
        click.echo(f"Error cleaning sessions: {e}")

@cli.command(name="config")
@click.option("--max-file-size", type=int, help="Maximum file size in bytes")
@click.option("--add-exclude", help="Add directory or extension to exclusion list")
@click.option("--remove-exclude", help="Remove directory or extension from exclusion list")
@click.option("--list-exclude", is_flag=True, help="List excluded directories and extensions")
def configure(max_file_size, add_exclude, remove_exclude, list_exclude):
    """Configure system parameters"""
    config_file = Path("code_system_config.json")
    
    # Load existing config if available
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
        except Exception:
            config = {}
    else:
        config = {
            "max_file_size": MAX_FILE_SIZE,
            "excluded_dirs": EXCLUDED_DIRS.copy(),
            "excluded_extensions": EXCLUDED_EXTENSIONS.copy()
        }
    
    # Update config
    if max_file_size:
        config["max_file_size"] = max_file_size
        click.echo(f"Set maximum file size to {max_file_size} bytes")
        
    if add_exclude:
        if add_exclude.startswith("."):
            if add_exclude not in config["excluded_extensions"]:
                config["excluded_extensions"].append(add_exclude)
                click.echo(f"Added {add_exclude} to excluded extensions")
        else:
            if add_exclude not in config["excluded_dirs"]:
                config["excluded_dirs"].append(add_exclude)
                click.echo(f"Added {add_exclude} to excluded directories")
                
    if remove_exclude:
        if remove_exclude.startswith("."):
            if remove_exclude in config["excluded_extensions"]:
                config["excluded_extensions"].remove(remove_exclude)
                click.echo(f"Removed {remove_exclude} from excluded extensions")
        else:
            if remove_exclude in config["excluded_dirs"]:
                config["excluded_dirs"].remove(remove_exclude)
                click.echo(f"Removed {remove_exclude} from excluded directories")
    
    # List exclusions
    if list_exclude:
        click.echo("Excluded directories:")
        for d in config["excluded_dirs"]:
            click.echo(f"  - {d}")
        click.echo("Excluded extensions:")
        for e in config["excluded_extensions"]:
            click.echo(f"  - {e}")
            
    # Save config
    config_file.write_text(json.dumps(config, indent=2))

if __name__ == "__main__":
    cli()