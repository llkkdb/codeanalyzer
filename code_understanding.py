import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import click
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Security configuration
ALLOWED_COMMANDS = {
    "find": {"flags": ["-name", "-type", "-path", "-exec"], "max_depth": 3},
    "grep": {"flags": ["-r", "-i", "-l", "-H", "-n"], "max_depth": None},
    "rg": {"flags": ["--files", "-g", "-l"], "max_depth": 3}
}

class CommandOutputParser:
    """Parse command outputs to extract file paths"""
    PATTERNS = {
        "grep": re.compile(r"^(.*?):\d+:"),  # Extract path from grep -Hn output
        "find": lambda x: x.strip(),          # Direct path output
        "rg": re.compile(r"^(.*?)(:\d+){2}")  # Ripgrep format
    }

    @classmethod
    def parse(cls, command: str, output: str) -> List[Path]:
        """Parse command output based on command type"""
        cmd_type = command.split()[0]
        parser = cls.PATTERNS.get(cmd_type)
        
        paths = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            
            if parser:
                match = parser.match(line)
                if match:
                    path_str = match.group(1) if isinstance(parser, re.Pattern) else line
            else:  # Fallback for unknown commands
                path_str = line.split(":")[0]
            
            try:
                path = Path(path_str).resolve()
                if path.exists():
                    paths.append(path)
            except:
                continue
        return paths

class SafeCommandExecutor:
    """Validate and execute commands safely"""
    def __init__(self):
        self.allow_list = ALLOWED_COMMANDS

    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Check if command is allowed"""
        parts = command.split()
        if not parts:
            return False, "Empty command"
        
        cmd = parts[0]
        if cmd not in self.allow_list:
            return False, f"Command {cmd} not allowed"
        
        # Check flags
        allowed_flags = self.allow_list[cmd]["flags"]
        for part in parts[1:]:
            if part.startswith("-") and part not in allowed_flags:
                return False, f"Disallowed flag {part} for {cmd}"
        
        # Check path depth
        if self.allow_list[cmd].get("max_depth"):
            path_parts = [p for p in parts if not p.startswith("-")]
            if len(path_parts) > self.allow_list[cmd]["max_depth"]:
                return False, f"Path depth exceeded for {cmd}"
        
        return True, ""

    def execute(self, command: str) -> List[Path]:
        """Execute validated command and return parsed paths"""
        valid, msg = self.validate_command(command)
        if not valid:
            raise ValueError(f"Invalid command: {msg}")
        
        try:
            result = subprocess.run(
                command, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5
            )
            return CommandOutputParser.parse(command, result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e.stderr}")
            return []
        except subprocess.TimeoutExpired:
            print("Command timed out")
            return []

class CodeSession:
    """Isolated session context"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.vector_store: Optional[FAISS] = None
        self.context_files: Set[Path] = set()
        self.search_history: List[Dict] = []
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def add_files(self, paths: List[Path], chunk_size: int = 1000):
        """Add files to session context with chunking"""
        new_paths = [p for p in paths if p not in self.context_files]
        if not new_paths:
            return

        docs = []
        for path in new_paths:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                chunks = [content[i:i+chunk_size] 
                          for i in range(0, len(content), chunk_size)]
                for i, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": str(path), "chunk": i+1}
                    ))
                self.context_files.add(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if docs:
            if self.vector_store:
                self.vector_store.add_documents(docs)
            else:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def query_context(self, question: str, k: int = 5) -> str:
        """Query session-specific context"""
        if not self.vector_store:
            return ""
            
        results = self.vector_store.similarity_search(question, k=k)
        return "\n\n".join(
            f"File: {doc.metadata['source']} (Chunk {doc.metadata['chunk']})\n"
            f"{doc.page_content[:500]}..." 
            for doc in results
        )

class SessionManager:
    """Manage multiple code sessions"""
    def __init__(self, storage_dir: Path = Path("sessions")):
        self.storage_dir = storage_dir
        self.active_session: Optional[CodeSession] = None
        self.sessions: Dict[str, CodeSession] = {}
        self.storage_dir.mkdir(exist_ok=True)

    def create_session(self, session_id: Optional[str] = None) -> CodeSession:
        """Create new isolated session"""
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        session = CodeSession(session_id)
        self.sessions[session_id] = session
        self.active_session = session
        return session

    def switch_session(self, session_id: str) -> bool:
        """Switch to existing session"""
        if session_id in self.sessions:
            self.active_session = self.sessions[session_id]
            return True
        return False

    def persist_session(self, session: CodeSession):
        """Save session to disk"""
        session_dir = self.storage_dir / session.session_id
        session_dir.mkdir(exist_ok=True)
        
        if session.vector_store:
            session.vector_store.save_local(str(session_dir / "faiss_index"))
        
        meta = {
            "created_at": session.created_at.isoformat(),
            "context_files": [str(p) for p in session.context_files],
            "search_history": session.search_history
        }
        (session_dir / "meta.json").write_text(json.dumps(meta))

class CodeUnderstandingSystem:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1)
        self.command_executor = SafeCommandExecutor()
        self.session_manager = SessionManager()
        self.knowledge_guidelines = self._load_knowledge_guidelines()

    def _load_knowledge_guidelines(self) -> str:
        try:
            return Path("knowledge_guidelines.md").read_text()
        except FileNotFoundError:
            return "# Default Knowledge Guidelines\n"

    def generate_search_commands(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.knowledge_guidelines}\nGenerate Linux commands to find relevant code."),
            ("human", "Query: {query}\nRespond ONLY with commands separated by newlines:")
        ])
        response = (prompt | self.llm).invoke({"query": query})
        return [cmd.strip() for cmd in response.content.split("\n") if cmd.strip()]

    def execute_search(self, commands: List[str]) -> List[Path]:
        found_files = set()
        for cmd in commands:
            try:
                found_files.update(self.command_executor.execute(cmd))
            except Exception as e:
                print(f"Error executing {cmd}: {e}")
        return list(found_files)

@click.group()
@click.pass_context
def cli(ctx):
    """Code Understanding System with Sessions"""
    ctx.obj = CodeUnderstandingSystem()

@cli.command()
@click.argument("query")
@click.pass_obj
def ask(system: CodeUnderstandingSystem, query: str):
    """Ask a question in current session"""
    if not system.session_manager.active_session:
        click.echo("No active session! Create one with 'session new'")
        return

    # Generate and execute commands
    commands = system.generate_search_commands(query)
    click.echo(f"Generated commands:\n" + "\n".join(commands))
    
    found_files = system.execute_search(commands)
    click.echo(f"Found {len(found_files)} files")
    
    # Add to session context
    session = system.session_manager.active_session
    session.add_files(found_files)
    
    # Generate answer
    context = session.query_context(query)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{system.knowledge_guidelines}\nContext:\n{context}"),
        ("human", "Question: {query}")
    ])
    answer = (prompt | system.llm).invoke({"query": query}).content
    click.echo(f"\nAnswer:\n{answer}")

@cli.group()
def session():
    """Session management commands"""
    pass

@session.command(name="new")
@click.option("--name", default=None)
@click.pass_obj
def new_session(system: CodeUnderstandingSystem, name: Optional[str]):
    """Create new session"""
    session = system.session_manager.create_session(name)
    click.echo(f"Created new session: {session.session_id}")

@session.command(name="switch")
@click.argument("session_id")
@click.pass_obj
def switch_session(system: CodeUnderstandingSystem, session_id: str):
    """Switch active session"""
    if system.session_manager.switch_session(session_id):
        click.echo(f"Switched to session: {session_id}")
    else:
        click.echo(f"Session {session_id} not found!")

@session.command(name="list")
@click.pass_obj
def list_sessions(system: CodeUnderstandingSystem):
    """List all sessions"""
    click.echo("Active sessions:")
    for session_id in system.session_manager.sessions:
        active = system.session_manager.active_session.session_id == session_id
        click.echo(f" {'*' if active else ' '} {session_id}")

if __name__ == "__main__":
    cli()