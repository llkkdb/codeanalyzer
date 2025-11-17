"""Unit tests for custom exceptions."""
import pytest

from codeanalyzer.exceptions import (
    CodeAnalyzerError,
    CommandValidationError,
    CommandExecutionError,
    SessionNotFoundError,
    FileProcessingError,
    ConfigurationError,
    LLMError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_base_exception_is_exception(self):
        """Test that base exception inherits from Exception."""
        assert issubclass(CodeAnalyzerError, Exception)

    def test_command_validation_error_hierarchy(self):
        """Test CommandValidationError hierarchy."""
        assert issubclass(CommandValidationError, CodeAnalyzerError)
        assert issubclass(CommandValidationError, Exception)

    def test_command_execution_error_hierarchy(self):
        """Test CommandExecutionError hierarchy."""
        assert issubclass(CommandExecutionError, CodeAnalyzerError)
        assert issubclass(CommandExecutionError, Exception)

    def test_session_not_found_error_hierarchy(self):
        """Test SessionNotFoundError hierarchy."""
        assert issubclass(SessionNotFoundError, CodeAnalyzerError)
        assert issubclass(SessionNotFoundError, Exception)

    def test_file_processing_error_hierarchy(self):
        """Test FileProcessingError hierarchy."""
        assert issubclass(FileProcessingError, CodeAnalyzerError)
        assert issubclass(FileProcessingError, Exception)

    def test_configuration_error_hierarchy(self):
        """Test ConfigurationError hierarchy."""
        assert issubclass(ConfigurationError, CodeAnalyzerError)
        assert issubclass(ConfigurationError, Exception)

    def test_llm_error_hierarchy(self):
        """Test LLMError hierarchy."""
        assert issubclass(LLMError, CodeAnalyzerError)
        assert issubclass(LLMError, Exception)


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_base_exception_message(self):
        """Test CodeAnalyzerError with message."""
        error = CodeAnalyzerError("Test error message")
        assert str(error) == "Test error message"

    def test_command_validation_error_message(self):
        """Test CommandValidationError with message."""
        error = CommandValidationError("Invalid command")
        assert str(error) == "Invalid command"
        assert isinstance(error, CodeAnalyzerError)

    def test_command_execution_error_message(self):
        """Test CommandExecutionError with message."""
        error = CommandExecutionError("Execution failed")
        assert str(error) == "Execution failed"
        assert isinstance(error, CodeAnalyzerError)

    def test_session_not_found_error_message(self):
        """Test SessionNotFoundError with message."""
        error = SessionNotFoundError("Session xyz not found")
        assert str(error) == "Session xyz not found"
        assert isinstance(error, CodeAnalyzerError)

    def test_file_processing_error_message(self):
        """Test FileProcessingError with message."""
        error = FileProcessingError("Cannot process file.txt")
        assert str(error) == "Cannot process file.txt"
        assert isinstance(error, CodeAnalyzerError)

    def test_configuration_error_message(self):
        """Test ConfigurationError with message."""
        error = ConfigurationError("Invalid config file")
        assert str(error) == "Invalid config file"
        assert isinstance(error, CodeAnalyzerError)

    def test_llm_error_message(self):
        """Test LLMError with message."""
        error = LLMError("LLM API call failed")
        assert str(error) == "LLM API call failed"
        assert isinstance(error, CodeAnalyzerError)


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    def test_raise_base_exception(self):
        """Test raising CodeAnalyzerError."""
        with pytest.raises(CodeAnalyzerError) as exc_info:
            raise CodeAnalyzerError("Base error")
        assert str(exc_info.value) == "Base error"

    def test_raise_command_validation_error(self):
        """Test raising CommandValidationError."""
        with pytest.raises(CommandValidationError) as exc_info:
            raise CommandValidationError("Validation failed")
        assert str(exc_info.value) == "Validation failed"

    def test_raise_command_execution_error(self):
        """Test raising CommandExecutionError."""
        with pytest.raises(CommandExecutionError) as exc_info:
            raise CommandExecutionError("Execution failed")
        assert str(exc_info.value) == "Execution failed"

    def test_raise_session_not_found_error(self):
        """Test raising SessionNotFoundError."""
        with pytest.raises(SessionNotFoundError) as exc_info:
            raise SessionNotFoundError("Session not found")
        assert str(exc_info.value) == "Session not found"

    def test_raise_file_processing_error(self):
        """Test raising FileProcessingError."""
        with pytest.raises(FileProcessingError) as exc_info:
            raise FileProcessingError("Processing failed")
        assert str(exc_info.value) == "Processing failed"

    def test_raise_configuration_error(self):
        """Test raising ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Config error")
        assert str(exc_info.value) == "Config error"

    def test_raise_llm_error(self):
        """Test raising LLMError."""
        with pytest.raises(LLMError) as exc_info:
            raise LLMError("LLM error")
        assert str(exc_info.value) == "LLM error"


class TestExceptionCatching:
    """Tests for catching exceptions."""

    def test_catch_specific_as_base(self):
        """Test catching specific exception as base exception."""
        with pytest.raises(CodeAnalyzerError):
            raise CommandValidationError("Test")

    def test_catch_with_base_exception(self):
        """Test catching any CodeAnalyzerError."""
        exceptions_raised = []

        for exc_class in [
            CommandValidationError,
            CommandExecutionError,
            SessionNotFoundError,
            FileProcessingError,
            ConfigurationError,
            LLMError,
        ]:
            try:
                raise exc_class("Test error")
            except CodeAnalyzerError as e:
                exceptions_raised.append(type(e).__name__)

        assert len(exceptions_raised) == 6
        assert "CommandValidationError" in exceptions_raised
        assert "LLMError" in exceptions_raised

    def test_exception_with_nested_cause(self):
        """Test exception with nested cause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError("Config failed") from e
        except ConfigurationError as e:
            assert str(e) == "Config failed"
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"


class TestExceptionAttributes:
    """Tests for exception attributes."""

    def test_exception_args(self):
        """Test exception args attribute."""
        error = CommandValidationError("Test", "Extra", "Args")
        assert error.args == ("Test", "Extra", "Args")

    def test_exception_str_representation(self):
        """Test string representation of exceptions."""
        error = LLMError("API rate limit exceeded")
        assert "API rate limit exceeded" in str(error)
        assert "API rate limit exceeded" in repr(error)

    def test_exception_with_format_string(self):
        """Test exception with formatted message."""
        session_id = "abc123"
        error = SessionNotFoundError(f"Session '{session_id}' not found")
        assert "abc123" in str(error)
        assert "not found" in str(error)


class TestExceptionDocstrings:
    """Tests for exception docstrings."""

    def test_base_exception_has_docstring(self):
        """Test that base exception has docstring."""
        assert CodeAnalyzerError.__doc__ is not None

    def test_all_exceptions_have_docstrings(self):
        """Test that all exception classes have docstrings."""
        exceptions = [
            CodeAnalyzerError,
            CommandValidationError,
            CommandExecutionError,
            SessionNotFoundError,
            FileProcessingError,
            ConfigurationError,
            LLMError,
        ]

        for exc_class in exceptions:
            assert exc_class.__doc__ is not None, f"{exc_class.__name__} missing docstring"


class TestExceptionUsagePatterns:
    """Tests for common exception usage patterns."""

    def test_validation_pattern(self):
        """Test typical validation error pattern."""

        def validate_command(cmd):
            if not cmd.strip():
                raise CommandValidationError("Command cannot be empty")
            if "rm -rf /" in cmd:
                raise CommandValidationError("Dangerous command not allowed")

        # Valid command
        validate_command("grep -r test")

        # Invalid commands
        with pytest.raises(CommandValidationError, match="cannot be empty"):
            validate_command("  ")

        with pytest.raises(CommandValidationError, match="Dangerous"):
            validate_command("rm -rf /")

    def test_error_chaining_pattern(self):
        """Test error chaining pattern."""

        def process_file(filename):
            try:
                # Simulate file operation
                raise IOError(f"Cannot read {filename}")
            except IOError as e:
                raise FileProcessingError(f"Failed to process {filename}") from e

        with pytest.raises(FileProcessingError) as exc_info:
            process_file("test.txt")

        assert "test.txt" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, IOError)

    def test_session_lookup_pattern(self):
        """Test session lookup error pattern."""

        def get_session(session_id, sessions):
            if session_id not in sessions:
                raise SessionNotFoundError(
                    f"Session '{session_id}' not found. "
                    f"Available sessions: {list(sessions.keys())}"
                )
            return sessions[session_id]

        sessions = {"session1": {}, "session2": {}}

        # Valid lookup
        assert get_session("session1", sessions) == {}

        # Invalid lookup
        with pytest.raises(SessionNotFoundError, match="session3.*not found"):
            get_session("session3", sessions)

    def test_llm_error_with_details(self):
        """Test LLM error with detailed information."""

        def call_llm(prompt):
            # Simulate API error
            raise LLMError(
                "OpenAI API call failed: Rate limit exceeded. "
                "Please try again in 60 seconds."
            )

        with pytest.raises(LLMError, match="Rate limit"):
            call_llm("Test prompt")


class TestExceptionEdgeCases:
    """Tests for exception edge cases."""

    def test_exception_with_none_message(self):
        """Test exception with None as message."""
        error = CodeAnalyzerError(None)
        assert str(error) == "None"

    def test_exception_with_empty_message(self):
        """Test exception with empty message."""
        error = ConfigurationError("")
        assert str(error) == ""

    def test_exception_with_numeric_message(self):
        """Test exception with numeric message."""
        error = FileProcessingError(404)
        assert str(error) == "404"

    def test_exception_equality(self):
        """Test exception equality."""
        error1 = CommandValidationError("Test")
        error2 = CommandValidationError("Test")

        # Exceptions are not equal even with same message
        assert error1 is not error2
        # But their string representations are
        assert str(error1) == str(error2)

    def test_exception_with_special_characters(self):
        """Test exception with special characters in message."""
        special_msg = "Error with 'quotes', \"double quotes\", and\nnewlines\ttabs"
        error = LLMError(special_msg)
        assert special_msg in str(error)
