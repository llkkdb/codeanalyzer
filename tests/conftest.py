"""Pytest fixtures for CodeAnalyzer tests."""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime
import tempfile
import shutil

from codeanalyzer import CodeUnderstandingSystem, Config
from codeanalyzer.core.session import CodeSession
from codeanalyzer.storage.manager import SessionManager
from codeanalyzer.core.executor import SafeCommandExecutor
from codeanalyzer.core.parser import CommandOutputParser


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config file."""
    config_file = temp_dir / "config.json"
    return config_file


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set mock OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123456789")


@pytest.fixture
def config(temp_config_file):
    """Create a Config instance with temp config file."""
    return Config(config_path=temp_config_file)


@pytest.fixture
def command_executor():
    """Create a SafeCommandExecutor instance."""
    return SafeCommandExecutor()


@pytest.fixture
def command_parser():
    """Create a CommandOutputParser instance."""
    return CommandOutputParser()


@pytest.fixture
def mock_session():
    """Create a mock CodeSession."""
    session = Mock(spec=CodeSession)
    session.session_id = "test-session-123"
    session.context_files = set()
    session.vector_store = None
    session.last_accessed = datetime.now()
    session.chunk_size = 1000
    session.chunk_overlap = 200
    return session


@pytest.fixture
def session_manager(temp_dir):
    """Create a SessionManager with temp storage."""
    return SessionManager(storage_dir=temp_dir / "sessions")


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.invoke.return_value = Mock(content="Test LLM response")
    return llm


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return embeddings


@pytest.fixture
def code_understanding_system(mock_openai_key, temp_dir):
    """Create a CodeUnderstandingSystem instance."""
    return CodeUnderstandingSystem()


@pytest.fixture
def sample_files(temp_dir):
    """Create sample code files for testing."""
    files = {}

    # Python file
    python_file = temp_dir / "example.py"
    python_file.write_text("""
def calculate_sum(a, b):
    '''Calculate the sum of two numbers.'''
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, value):
        self.result += value
        return self.result
""")
    files['python'] = python_file

    # JavaScript file
    js_file = temp_dir / "script.js"
    js_file.write_text("""
function authenticate(username, password) {
    // Authentication logic
    return validateCredentials(username, password);
}

const API_KEY = 'test-key';
""")
    files['javascript'] = js_file

    # README file
    readme_file = temp_dir / "README.md"
    readme_file.write_text("""
# Test Project

This is a test project for CodeAnalyzer.

## Features
- Authentication
- Calculation
- Data processing
""")
    files['readme'] = readme_file

    # Config file
    config_file = temp_dir / "config.json"
    config_file.write_text('{"setting1": "value1", "setting2": 42}')
    files['config'] = config_file

    return files


@pytest.fixture
def sample_code_structure(temp_dir):
    """Create a sample code directory structure."""
    # Create directory structure
    src_dir = temp_dir / "src"
    src_dir.mkdir()

    tests_dir = temp_dir / "tests"
    tests_dir.mkdir()

    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()

    # Create files
    (src_dir / "main.py").write_text("def main(): pass")
    (src_dir / "utils.py").write_text("def helper(): pass")
    (tests_dir / "test_main.py").write_text("def test_main(): pass")
    (docs_dir / "README.md").write_text("# Documentation")

    return {
        'root': temp_dir,
        'src': src_dir,
        'tests': tests_dir,
        'docs': docs_dir
    }


@pytest.fixture
def grep_output():
    """Sample grep command output."""
    return """
file1.py:10:def authenticate(username, password):
file1.py:15:    return check_credentials(username, password)
file2.py:20:class Authentication:
file2.py:25:    def __init__(self):
src/auth.py:30:def validate_token(token):
"""


@pytest.fixture
def find_output():
    """Sample find command output."""
    return """
./file1.py
./file2.py
./src/auth.py
./tests/test_auth.py
./docs/authentication.md
"""


@pytest.fixture
def rg_output():
    """Sample ripgrep command output."""
    return """
file1.py:10:def authenticate(username, password):
file2.py:20:class Authentication:
src/auth.py:30:def validate_token(token):
"""


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables for each test."""
    # Clear potentially interfering env vars
    env_vars_to_clear = [
        'CODEANALYZER_CONFIG',
        'CODEANALYZER_LOG_DIR',
        'CODEANALYZER_LOG_LEVEL',
        'CODEANALYZER_SESSION_DIR',
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.similarity_search.return_value = [
        Mock(page_content="Test content 1", metadata={"source": "file1.py"}),
        Mock(page_content="Test content 2", metadata={"source": "file2.py"}),
    ]
    return store


@pytest.fixture
def session_with_files(mock_session, sample_files):
    """Create a session with sample files added."""
    mock_session.context_files = set(sample_files.values())
    return mock_session


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API"
    )
