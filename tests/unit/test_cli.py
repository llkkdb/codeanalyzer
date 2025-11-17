"""Unit tests for CLI commands."""
import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from codeanalyzer.cli import cli
from codeanalyzer import CodeUnderstandingSystem
from codeanalyzer.core.session import CodeSession


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_system():
    """Create a mock CodeUnderstandingSystem."""
    system = Mock(spec=CodeUnderstandingSystem)
    system.session_manager = Mock()
    system.session_manager.active_session = None
    system.session_manager.sessions = {}
    return system


class TestAskCommand:
    """Tests for the 'ask' command."""

    def test_ask_without_active_session(self, runner):
        """Test ask command fails gracefully without active session."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_sys.session_manager.active_session = None
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['ask', 'test query'])

            assert result.exit_code == 0
            assert "No active session" in result.output
            assert "session new" in result.output

    def test_ask_with_active_session(self, runner):
        """Test ask command with active session."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_session = Mock()
            mock_sys.session_manager.active_session = mock_session
            mock_sys.generate_search_commands.return_value = ["grep -rHn test"]
            mock_sys.execute_search.return_value = []
            mock_sys.ask.return_value = "Test answer"
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['ask', 'test query'])

            assert result.exit_code == 0
            assert "Test answer" in result.output
            mock_sys.ask.assert_called_once()

    def test_ask_with_custom_k(self, runner):
        """Test ask command with custom k parameter."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_session = Mock()
            mock_sys.session_manager.active_session = mock_session
            mock_sys.generate_search_commands.return_value = ["grep -rHn test"]
            mock_sys.execute_search.return_value = []
            mock_sys.ask.return_value = "Test answer"
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['ask', 'test query', '--k', '10'])

            assert result.exit_code == 0
            mock_sys.ask.assert_called_with('test query', k=10)

    def test_ask_handles_error(self, runner):
        """Test ask command handles errors gracefully."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_session = Mock()
            mock_sys.session_manager.active_session = mock_session
            mock_sys.generate_search_commands.side_effect = Exception("Test error")
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['ask', 'test query'])

            assert result.exit_code == 0
            assert "error" in result.output.lower()


class TestSessionCommands:
    """Tests for session management commands."""

    def test_session_new_default(self, runner):
        """Test creating new session with default name."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_session = Mock()
            mock_session.session_id = "session_12345678"
            mock_sys.session_manager.create_session.return_value = mock_session
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['session', 'new'])

            assert result.exit_code == 0
            assert "session_" in result.output
            mock_sys.session_manager.create_session.assert_called_once()

    def test_session_new_with_name(self, runner):
        """Test creating new session with custom name."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_session = Mock()
            mock_session.session_id = "my-session"
            mock_sys.session_manager.create_session.return_value = mock_session
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['session', 'new', '--name', 'my-session'])

            assert result.exit_code == 0
            assert "my-session" in result.output

    def test_session_new_with_chunk_size(self, runner):
        """Test creating new session with custom chunk size."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_session = Mock()
            mock_session.session_id = "test-session"
            mock_sys.session_manager.create_session.return_value = mock_session
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['session', 'new', '--chunk-size', '500'])

            assert result.exit_code == 0
            args, kwargs = mock_sys.session_manager.create_session.call_args
            # Check that chunk_size was passed
            assert 500 in args or kwargs.get('chunk_size') == 500

    def test_session_switch_existing(self, runner):
        """Test switching to existing session."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_sys.session_manager.switch_session.return_value = True
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['session', 'switch', 'test-session'])

            assert result.exit_code == 0
            assert "Switched to session: test-session" in result.output

    def test_session_switch_not_found(self, runner):
        """Test switching to non-existent session."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_sys.session_manager.switch_session.return_value = False
            mock_sys.session_manager.load_session.return_value = None
            mock_sys_class.return_value = mock_sys

            result = runner.invoke(cli, ['session', 'switch', 'nonexistent'])

            assert result.exit_code == 0
            assert "not found" in result.output
            assert "session list" in result.output  # Helpful hint

    def test_session_list_empty(self, runner):
        """Test listing sessions when none exist."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_sys.session_manager.sessions = {}
            mock_sys.session_manager.active_session = None
            mock_sys.session_manager.storage_dir = Path("/tmp/sessions")
            mock_sys_class.return_value = mock_sys

            with patch('pathlib.Path.exists', return_value=False):
                result = runner.invoke(cli, ['session', 'list'])

                assert result.exit_code == 0
                assert "Active sessions:" in result.output

    def test_session_list_with_sessions(self, runner):
        """Test listing sessions with active sessions."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            from datetime import datetime
            mock_sys = Mock()
            mock_session = Mock()
            mock_session.session_id = "test-session"
            mock_session.context_files = [Path("file1.py"), Path("file2.py")]
            mock_session.last_accessed = datetime.now()

            mock_sys.session_manager.sessions = {"test-session": mock_session}
            mock_sys.session_manager.active_session = mock_session
            mock_sys.session_manager.storage_dir = Path("/tmp/sessions")
            mock_sys_class.return_value = mock_sys

            with patch('pathlib.Path.exists', return_value=False):
                result = runner.invoke(cli, ['session', 'list'])

                assert result.exit_code == 0
                assert "test-session" in result.output
                assert "2 files" in result.output
                assert "*" in result.output  # Active marker

    def test_session_clean(self, runner):
        """Test cleaning old sessions."""
        with patch('codeanalyzer.cli.CodeUnderstandingSystem') as mock_sys_class:
            mock_sys = Mock()
            mock_sys.session_manager.storage_dir = Path("/tmp/sessions")
            mock_sys_class.return_value = mock_sys

            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.iterdir', return_value=[]):
                    result = runner.invoke(cli, ['session', 'clean', '--days', '30'])

                    assert result.exit_code == 0


class TestConfigCommand:
    """Tests for the 'config' command."""

    def test_config_set_max_file_size(self, runner):
        """Test setting max file size."""
        with patch('codeanalyzer.cli.Config') as mock_config_class:
            mock_cfg = Mock()
            mock_config_class.return_value = mock_cfg

            result = runner.invoke(cli, ['config', '--max-file-size', '5000000'])

            assert result.exit_code == 0
            mock_cfg.set.assert_called_with("max_file_size", 5000000)
            mock_cfg.save.assert_called_once()

    def test_config_add_exclude(self, runner):
        """Test adding exclusion."""
        with patch('codeanalyzer.cli.Config') as mock_config_class:
            mock_cfg = Mock()
            mock_cfg.get.return_value = []
            mock_config_class.return_value = mock_cfg

            result = runner.invoke(cli, ['config', '--add-exclude', '.pyc'])

            assert result.exit_code == 0
            assert mock_cfg.set.called

    def test_config_list_exclude(self, runner):
        """Test listing exclusions."""
        with patch('codeanalyzer.cli.Config') as mock_config_class:
            mock_cfg = Mock()
            mock_cfg.get.side_effect = [
                [".git", ".venv"],  # excluded_dirs
                [".pyc", ".pyo"]    # excluded_extensions
            ]
            mock_config_class.return_value = mock_cfg

            result = runner.invoke(cli, ['config', '--list-exclude'])

            assert result.exit_code == 0
            assert ".git" in result.output
            assert ".pyc" in result.output


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert "CodeAnalyzer" in result.output
        assert "ask" in result.output
        assert "session" in result.output

    def test_session_help(self, runner):
        """Test session subcommand help."""
        result = runner.invoke(cli, ['session', '--help'])

        assert result.exit_code == 0
        assert "new" in result.output
        assert "switch" in result.output
        assert "list" in result.output

    def test_invalid_command(self, runner):
        """Test invalid command handling."""
        result = runner.invoke(cli, ['invalid-command'])

        assert result.exit_code != 0
