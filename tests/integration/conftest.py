import sys
import os
from pathlib import Path

# Add the parent directory to sys.path so tests can find the module
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Now imports should work
import pytest
from unittest.mock import Mock, patch

# Force SQLite version override
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

@pytest.fixture
def mock_llm():
    mock = Mock()
    response_mock = Mock()
    response_mock.content = "grep -r TODO .\ngit grep -i FIXME"
    response_mock.split.return_value = response_mock.content.split("\n")
    response_mock.split.return_value = response_mock.content.split("\n")
    mock.invoke.return_value = response_mock
    return mock

@pytest.fixture
def mock_embeddings():
    return Mock()

@pytest.fixture
def mock_chroma():
    mock = Mock()
    mock.get.return_value = {'documents': []}
    return mock