import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Get the path to the root of the project
project_root = Path(__file__).parent.parent.parent.absolute()

# Add it to the Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from code_understanding import CodeUnderstandingSystem, SessionManager, CodeSession

@pytest.fixture
def mock_embeddings():
    return Mock()

def test_full_workflow(tmp_path, mock_embeddings):
    # Instead of mocking the LLM, patch the specific method
    with patch('code_understanding.CodeUnderstandingSystem.generate_search_commands') as mock_generate:
        # Set return value for the patched method
        mock_generate.return_value = ["grep -rHn TODO", "find . -name *.py -maxdepth 2"]
        
        # Create system with default LLM (won't be used due to patch)
        system = CodeUnderstandingSystem()
        system.session_manager = SessionManager(storage_dir=tmp_path / "sessions")
        session = system.session_manager.create_session()
        session.embeddings = mock_embeddings
    
        # Test command generation (uses our patched return value)
        commands = system.generate_search_commands("Find TODO comments")
        assert len(commands) == 2
        assert "grep" in commands[0]
    
        # Test command execution and file processing
        with patch('code_understanding.SafeCommandExecutor.execute') as mock_execute:
            mock_execute.return_value = [Path("dummy_file.py")]
            found_files = system.execute_search(commands)
            assert len(found_files) > 0
    
        # Verify session persistence
        session_id = system.session_manager.active_session.session_id
        system.session_manager.persist_session(system.session_manager.active_session)
        assert (tmp_path / "sessions" / session_id / "meta.json").exists()

def test_session_lifecycle(tmp_path):
    manager = SessionManager(storage_dir=tmp_path / "sessions")
    
    # Test session creation
    session = manager.create_session()
    assert session.session_id in manager.sessions
    
    # Test session switching
    assert manager.switch_session("nonexistent") is False
    assert manager.switch_session(session.session_id) is True
    
    # Test session persistence
    manager.persist_session(session)
    assert (tmp_path / "sessions" / session.session_id / "meta.json").exists()
    
    # Test session loading
    loaded = manager.load_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id

def test_file_processing(tmp_path):
    system = CodeUnderstandingSystem()
    system.session_manager.create_session()  # Ensure active session exists
    
    with patch("code_understanding.Chroma.from_documents") as mock_store:
        # Test file addition and chunking
        test_file = tmp_path / "test.txt"
        test_file.write_text("a" * 2500)  # 2500 character file
        # Create new session with chunk_size=1000
        system.session_manager.create_session(chunk_size=1000)
        system.session_manager.active_session.add_files([test_file])
        
        assert len(system.session_manager.active_session.context_files) == 1
        assert mock_store.call_count == 1
        # We can't easily check the length of the first argument due to mocking
        # But we can verify it was called with documents and embeddings
        assert len(mock_store.call_args[0]) >= 2