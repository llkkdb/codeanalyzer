"""
Custom exceptions for the codeanalyzer package.
"""


class CodeAnalyzerError(Exception):
    """Base exception for all codeanalyzer errors."""
    pass


class CommandValidationError(CodeAnalyzerError):
    """Raised when a command fails validation."""
    pass


class CommandExecutionError(CodeAnalyzerError):
    """Raised when a command execution fails."""
    pass


class SessionNotFoundError(CodeAnalyzerError):
    """Raised when a session cannot be found."""
    pass


class FileProcessingError(CodeAnalyzerError):
    """Raised when file processing fails."""
    pass


class ConfigurationError(CodeAnalyzerError):
    """Raised when there's a configuration error."""
    pass


class LLMError(CodeAnalyzerError):
    """Raised when LLM operations fail."""
    pass
