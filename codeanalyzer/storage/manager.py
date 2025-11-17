"""
Session management with persistence and cleanup.
"""
import asyncio
import json
import time
import threading
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from langchain_community.vectorstores import Chroma

from ..core.session import CodeSession
from ..config import Config
from ..exceptions import SessionNotFoundError

logger = logging.getLogger("codeanalyzer.manager")


class SessionManager:
    """
    Manage multiple code sessions with cleanup and improved persistence.

    Features:
    - Session creation and switching
    - Automatic persistence to disk
    - Background cleanup of inactive sessions
    - Auto-loading of recent sessions on startup
    """

    def __init__(self, config: Config = None, storage_dir: Path = None, max_sessions: int = None):
        """
        Initialize session manager.

        Args:
            config: Configuration object. If None, uses defaults.
            storage_dir: Directory for session storage. If None, uses 'sessions'.
            max_sessions: Maximum number of concurrent sessions. If None, uses config default.
        """
        self.config = config or Config()
        self.storage_dir = storage_dir or Path("sessions")
        self.active_session: Optional[CodeSession] = None
        self.sessions: Dict[str, CodeSession] = {}
        self.storage_dir.mkdir(exist_ok=True)
        self.max_sessions = max_sessions or self.config.max_sessions
        self._sessions_lock = threading.Lock()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_inactive_sessions,
            daemon=True
        )
        self._cleanup_thread.start()

        # Auto-load persisted sessions
        self._auto_load_sessions()

    def create_session(self, session_id: Optional[str] = None, chunk_size: int = None) -> CodeSession:
        """
        Create new isolated session with cleanup if needed.

        Args:
            session_id: Optional session ID. If None, generates a unique ID.
            chunk_size: Optional chunk size. If None, uses config default.

        Returns:
            Newly created CodeSession.
        """
        with self._sessions_lock:
            session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

            # Cleanup if too many sessions
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_oldest_session()

            session = CodeSession(session_id, self.config, chunk_size)
            self.sessions[session_id] = session
            self.active_session = session
            logger.info(f"Created session: {session_id}")
            return session

    def switch_session(self, session_id: str) -> bool:
        """
        Switch to existing session.

        Args:
            session_id: ID of session to switch to.

        Returns:
            True if switch succeeded, False if session not found.
        """
        with self._sessions_lock:
            if session_id in self.sessions:
                self.active_session = self.sessions[session_id]
                self.active_session.last_accessed = datetime.now()
                logger.info(f"Switched to session: {session_id}")
                return True
            return False

    async def persist_session_async(self, session: CodeSession):
        """
        Save session to disk asynchronously.

        Args:
            session: Session to persist.
        """
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
            lambda: (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        )
        logger.debug(f"Persisted session: {session.session_id}")

    def persist_session(self, session: CodeSession):
        """Synchronous wrapper for persist_session_async."""
        asyncio.run(self.persist_session_async(session))

    def load_session(self, session_id: str) -> Optional[CodeSession]:
        """
        Load a previously persisted session.

        Args:
            session_id: ID of session to load.

        Returns:
            Loaded CodeSession or None if not found.
        """
        with self._sessions_lock:
            session_dir = self.storage_dir / session_id
            if not session_dir.exists():
                return None

            try:
                if not (session_dir / "meta.json").exists():
                    return None

                meta = json.loads((session_dir / "meta.json").read_text())
                session = CodeSession(session_id, self.config)
                session.created_at = datetime.fromisoformat(meta["created_at"])
                session.last_accessed = datetime.fromisoformat(
                    meta.get("last_accessed", meta["created_at"])
                )
                session.context_files = set(
                    Path(p) for p in meta["context_files"] if Path(p).exists()
                )
                session.search_history = meta["search_history"]

                # Restore file metadata
                if "file_metadata" in meta:
                    session._file_metadata = {
                        Path(k): v for k, v in meta["file_metadata"].items()
                    }

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
                logger.info(f"Loaded session: {session_id}")
                return session
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error loading session {session_id}: {e}")
                return None

    def _auto_load_sessions(self):
        """Auto-load recent sessions at startup."""
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
                    last_accessed = datetime.fromisoformat(
                        meta.get("last_accessed", meta["created_at"])
                    )
                    session_dirs_with_time.append((d, last_accessed))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            # Sort by most recently accessed
            session_dirs_with_time.sort(key=lambda x: x[1], reverse=True)

            # Load most recent sessions up to max_sessions
            for d, _ in session_dirs_with_time[:self.max_sessions]:
                try:
                    self.load_session(d.name)
                except Exception as e:
                    logger.warning(f"Failed to auto-load session {d.name}: {e}")

            # Set active session to most recently used
            if session_dirs_with_time and self.sessions:
                most_recent = session_dirs_with_time[0][0].name
                if most_recent in self.sessions:
                    self.active_session = self.sessions[most_recent]
                    logger.info(f"Active session set to: {most_recent}")

        except Exception as e:
            logger.error(f"Error auto-loading sessions: {e}")

    def _cleanup_inactive_sessions(self):
        """Periodically clean up inactive sessions."""
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
        """Remove oldest session when max sessions is reached."""
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
