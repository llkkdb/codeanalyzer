"""
Code session management for isolated analysis contexts.
"""
import asyncio
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ..config import Config
from ..exceptions import FileProcessingError

logger = logging.getLogger("codeanalyzer.session")


class CodeSession:
    """
    Isolated session context with improved file handling.

    Each session maintains its own vector store, context files, and search history.
    """

    def __init__(self, session_id: str, config: Config = None, chunk_size: int = None):
        """
        Initialize a code session.

        Args:
            session_id: Unique identifier for this session.
            config: Configuration object. If None, uses defaults.
            chunk_size: Chunk size for document processing. If None, uses config default.
        """
        self.session_id = session_id
        self.config = config or Config()
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.vector_store = None
        self.context_files: Set[Path] = set()
        self.search_history: List[Dict[str, Any]] = []
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = Path(f"chroma_sessions/{session_id}")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size or self.config.default_chunk_size
        self._file_processing_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._file_metadata: Dict[Path, float] = {}  # Store file modification times

    async def add_files_async(self, paths: List[Path]) -> int:
        """
        Add files to session context with chunking asynchronously.

        Args:
            paths: List of file paths to add.

        Returns:
            Number of documents added.
        """
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
                    except OSError as e:
                        logger.warning(f"Cannot stat file {p}: {e}")
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
        """Synchronous wrapper for add_files_async."""
        return asyncio.run(self.add_files_async(paths))

    def _process_files(self, paths: List[Path]) -> List[Document]:
        """
        Process files into chunks with better error handling.

        Args:
            paths: List of file paths to process.

        Returns:
            List of Document objects.
        """
        docs = []

        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
            chunk_results = list(executor.map(self._process_single_file, paths))

        for path, chunks in chunk_results:
            if chunks:
                docs.extend(chunks)
                self.context_files.add(path)

        return docs

    def _process_single_file(self, path: Path) -> Tuple[Path, List[Document]]:
        """
        Process a single file into document chunks.

        Args:
            path: Path to file to process.

        Returns:
            Tuple of (path, list of Document chunks).
        """
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
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Error processing {path}: {e}")

        return path, chunks

    def _is_binary_file(self, path: Path) -> bool:
        """
        Check if file is binary.

        Args:
            path: Path to check.

        Returns:
            True if file is binary, False otherwise.
        """
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk  # Simple heuristic for binary files
        except (OSError, IOError):
            return True

    def _get_adjusted_chunk_size(self, path: Path) -> int:
        """
        Adjust chunk size based on file type.

        Args:
            path: Path to file.

        Returns:
            Adjusted chunk size.
        """
        # Use larger chunks for documentation files
        if path.suffix.lower() in ['.md', '.txt', '.rst']:
            return self.chunk_size * 2
        # Use smaller chunks for code files
        elif path.suffix.lower() in ['.py', '.js', '.java', '.cpp', '.h', '.c']:
            return self.chunk_size
        # Default chunk size for other files
        return self.chunk_size

    async def query_context_async(self, question: str, k: int = 5) -> str:
        """
        Query session-specific context asynchronously.

        Args:
            question: Question to query.
            k: Number of documents to retrieve.

        Returns:
            Context string from relevant documents.
        """
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
        """Synchronous wrapper for query_context_async."""
        return asyncio.run(self.query_context_async(question, k))
