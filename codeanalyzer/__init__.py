"""
CodeAnalyzer - An intelligent code understanding system for exploring and analyzing codebases.

This package provides tools for:
- Natural language code search
- Context-aware responses using RAG
- Safe command execution with validation
- Session management with persistence
"""

__version__ = "0.1.0"

from .core import CommandOutputParser, SafeCommandExecutor, CodeSession
from .storage import SessionManager
from .llm import CodeUnderstandingSystem
from .config import Config, setup_logging
from .exceptions import (
    CodeAnalyzerError,
    CommandValidationError,
    CommandExecutionError,
    SessionNotFoundError,
    FileProcessingError,
    ConfigurationError,
    LLMError,
)

__all__ = [
    "__version__",
    "CommandOutputParser",
    "SafeCommandExecutor",
    "CodeSession",
    "SessionManager",
    "CodeUnderstandingSystem",
    "Config",
    "setup_logging",
    "CodeAnalyzerError",
    "CommandValidationError",
    "CommandExecutionError",
    "SessionNotFoundError",
    "FileProcessingError",
    "ConfigurationError",
    "LLMError",
]
