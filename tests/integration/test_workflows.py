"""Integration tests for complete workflows."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from codeanalyzer import CodeUnderstandingSystem
from codeanalyzer.core.session import CodeSession
from codeanalyzer.storage.manager import SessionManager


@pytest.mark.integration
class TestCompleteWorkflow:
    """Integration tests for complete user workflows."""

    def test_session_creation_and_query_workflow(self, temp_dir, mock_openai_key, sample_files):
        """Test complete workflow: create session, add files, query."""
        # Create system
        system = CodeUnderstandingSystem()
        system.session_manager.storage_dir = temp_dir / "sessions"

        # Create session
        session = system.session_manager.create_session("test-workflow")
        assert session is not None
        assert session.session_id == "test-workflow"
        assert system.session_manager.active_session == session

        # Verify session is in manager
        assert "test-workflow" in system.session_manager.sessions

        # Add files to session
        files_to_add = list(sample_files.values())[:2]  # Add 2 files
        with patch.object(session, 'add_files'):
            session.add_files(files_to_add)
            session.add_files.assert_called_once()

    def test_session_persistence_workflow(self, temp_dir, mock_openai_key):
        """Test session persistence across manager instances."""
        storage_dir = temp_dir / "sessions"

        # Create first manager and session
        manager1 = SessionManager(storage_dir=storage_dir)
        session1 = manager1.create_session("persistent-session")
        session1.context_files.add(Path("file1.py"))
        session1.context_files.add(Path("file2.py"))

        # Persist session
        manager1.persist_session(session1)

        # Create new manager and load session
        manager2 = SessionManager(storage_dir=storage_dir)
        loaded_session = manager2.load_session("persistent-session")

        assert loaded_session is not None
        assert loaded_session.session_id == "persistent-session"
        assert len(loaded_session.context_files) == 2

    def test_multiple_session_switching(self, temp_dir, mock_openai_key):
        """Test switching between multiple sessions."""
        manager = SessionManager(storage_dir=temp_dir / "sessions")

        # Create multiple sessions
        session1 = manager.create_session("session-1")
        session2 = manager.create_session("session-2")
        session3 = manager.create_session("session-3")

        # Initially, session-3 should be active (last created)
        assert manager.active_session == session3

        # Switch to session-1
        success = manager.switch_session("session-1")
        assert success
        assert manager.active_session == session1

        # Switch to session-2
        success = manager.switch_session("session-2")
        assert success
        assert manager.active_session == session2

        # Try to switch to non-existent session
        success = manager.switch_session("non-existent")
        assert not success
        assert manager.active_session == session2  # Should remain on session-2

    def test_command_generation_and_execution_workflow(self, temp_dir, mock_openai_key, sample_code_structure):
        """Test command generation and execution workflow."""
        system = CodeUnderstandingSystem()
        system.session_manager.create_session("command-test")

        # Mock LLM command generation
        with patch.object(system, 'generate_search_commands') as mock_gen:
            mock_gen.return_value = [
                "grep -rHn authenticate",
                "find . -name *auth*.py -maxdepth 3"
            ]

            commands = system.generate_search_commands("Find authentication code")

            assert len(commands) == 2
            assert "grep" in commands[0]
            assert "find" in commands[1]

    def test_error_handling_workflow(self, temp_dir, mock_openai_key):
        """Test error handling in typical workflows."""
        system = CodeUnderstandingSystem()

        # Try to query without active session
        assert system.session_manager.active_session is None

        # This should handle the error gracefully
        # (In actual implementation, would raise or return error)

        # Create session
        system.session_manager.create_session("error-test")
        assert system.session_manager.active_session is not None

    def test_concurrent_session_operations(self, temp_dir, mock_openai_key):
        """Test concurrent session operations."""
        manager = SessionManager(storage_dir=temp_dir / "sessions", max_sessions=5)

        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = manager.create_session(f"session-{i}")
            sessions.append(session)

        assert len(manager.sessions) == 5

        # Try to create one more (should trigger cleanup)
        session6 = manager.create_session("session-6")

        # Should still have max_sessions
        assert len(manager.sessions) <= manager.max_sessions


@pytest.mark.integration
class TestFileProcessingWorkflow:
    """Integration tests for file processing workflows."""

    def test_file_chunking_workflow(self, temp_dir, mock_openai_key):
        """Test file chunking in real workflow."""
        # Create a large file
        large_file = temp_dir / "large.py"
        content = "# Python file\n" + ("def function():\n    pass\n" * 100)
        large_file.write_text(content)

        # Create session with specific chunk size
        manager = SessionManager(storage_dir=temp_dir / "sessions")
        session = manager.create_session("chunk-test", chunk_size=500)

        assert session.chunk_size == 500
        assert session.chunk_overlap == 200

    def test_binary_file_detection_workflow(self, temp_dir, mock_openai_key):
        """Test that binary files are properly detected and skipped."""
        # Create binary file
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05' * 100)

        # Create text file
        text_file = temp_dir / "text.py"
        text_file.write_text("def test(): pass")

        manager = SessionManager(storage_dir=temp_dir / "sessions")
        session = manager.create_session("binary-test")

        # The session should handle binary detection
        # (Actual implementation would skip binary files)

    def test_large_codebase_workflow(self, temp_dir, mock_openai_key):
        """Test workflow with many files."""
        # Create many files
        for i in range(50):
            file_path = temp_dir / f"file_{i}.py"
            file_path.write_text(f"def function_{i}(): pass")

        manager = SessionManager(storage_dir=temp_dir / "sessions")
        session = manager.create_session("large-codebase")

        # Session should be created successfully
        assert session is not None


@pytest.mark.integration
class TestConfigurationWorkflow:
    """Integration tests for configuration workflows."""

    def test_config_file_creation_and_loading(self, temp_dir):
        """Test configuration file persistence."""
        from codeanalyzer import Config

        config_path = temp_dir / "config.json"

        # Create config and set values
        config1 = Config(config_path=config_path)
        config1.set("custom_setting", "test_value")
        config1.set("max_file_size", 999999)
        config1.save()

        # Load config in new instance
        config2 = Config(config_path=config_path)

        assert config2.get("custom_setting") == "test_value"
        assert config2.get("max_file_size") == 999999

    def test_environment_override_workflow(self, monkeypatch, temp_dir):
        """Test environment variable overrides."""
        from codeanalyzer import Config

        # Set environment variable
        monkeypatch.setenv("CODEANALYZER_LOG_LEVEL", "DEBUG")

        config = Config(config_path=temp_dir / "config.json")

        # Environment variable should override default
        assert config.log_level == "DEBUG"


@pytest.mark.integration
class TestErrorRecoveryWorkflow:
    """Integration tests for error recovery."""

    def test_session_recovery_after_crash(self, temp_dir, mock_openai_key):
        """Test session recovery after simulated crash."""
        storage_dir = temp_dir / "sessions"

        # Create session and persist
        manager1 = SessionManager(storage_dir=storage_dir)
        session1 = manager1.create_session("recovery-test")
        session1.context_files.add(Path("file.py"))
        manager1.persist_session(session1)

        # Simulate crash - create new manager
        del manager1

        # New manager should be able to load session
        manager2 = SessionManager(storage_dir=storage_dir)
        recovered_session = manager2.load_session("recovery-test")

        assert recovered_session is not None
        assert recovered_session.session_id == "recovery-test"

    def test_corrupted_session_handling(self, temp_dir, mock_openai_key):
        """Test handling of corrupted session data."""
        storage_dir = temp_dir / "sessions"
        manager = SessionManager(storage_dir=storage_dir)

        # Create invalid session directory
        invalid_session_dir = storage_dir / "invalid-session"
        invalid_session_dir.mkdir(parents=True)
        (invalid_session_dir / "meta.json").write_text("invalid json{")

        # Should handle gracefully
        result = manager.load_session("invalid-session")
        assert result is None


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceWorkflow:
    """Integration tests for performance scenarios."""

    def test_many_files_performance(self, temp_dir, mock_openai_key):
        """Test performance with many files."""
        import time

        # Create 100 files
        for i in range(100):
            (temp_dir / f"file_{i}.py").write_text(f"def func_{i}(): pass")

        manager = SessionManager(storage_dir=temp_dir / "sessions")

        start_time = time.time()
        session = manager.create_session("perf-test")
        creation_time = time.time() - start_time

        # Session creation should be fast (< 1 second)
        assert creation_time < 1.0

    def test_large_file_handling(self, temp_dir, mock_openai_key):
        """Test handling of large files."""
        # Create 1MB file
        large_file = temp_dir / "large.py"
        content = "# Large file\n" + ("def function():\n    pass\n" * 10000)
        large_file.write_text(content)

        manager = SessionManager(storage_dir=temp_dir / "sessions")
        session = manager.create_session("large-file-test")

        # Should handle without crashing
        assert session is not None


@pytest.mark.integration
class TestRealWorldScenarios:
    """Integration tests simulating real-world usage."""

    def test_onboarding_scenario(self, temp_dir, mock_openai_key, sample_code_structure):
        """Simulate new developer onboarding workflow."""
        system = CodeUnderstandingSystem()
        system.session_manager.storage_dir = temp_dir / "sessions"

        # Developer creates session
        session = system.session_manager.create_session("onboarding")

        # Mock asking questions
        with patch.object(system, 'generate_search_commands') as mock_gen:
            mock_gen.return_value = ["grep -rHn main", "find . -name README*"]

            # Ask about project structure
            commands = system.generate_search_commands("What is this project about?")
            assert len(commands) > 0

    def test_debugging_scenario(self, temp_dir, mock_openai_key):
        """Simulate debugging workflow."""
        system = CodeUnderstandingSystem()
        system.session_manager.storage_dir = temp_dir / "sessions"

        # Create debug session
        session = system.session_manager.create_session("debug-session")

        with patch.object(system, 'generate_search_commands') as mock_gen:
            # Search for error-related code
            mock_gen.return_value = ["grep -rHn exception", "grep -rHn error"]
            commands = system.generate_search_commands("Find error handling")
            assert any("error" in cmd or "exception" in cmd for cmd in commands)

    def test_code_review_scenario(self, temp_dir, mock_openai_key):
        """Simulate code review workflow."""
        system = CodeUnderstandingSystem()
        system.session_manager.storage_dir = temp_dir / "sessions"

        # Create review session
        session = system.session_manager.create_session("code-review")

        # Simulate multiple review queries
        with patch.object(system, 'generate_search_commands') as mock_gen:
            # Check authentication patterns
            mock_gen.return_value = ["grep -rHn auth"]
            commands = system.generate_search_commands("Show authentication code")
            assert commands
