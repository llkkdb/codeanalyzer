import pytest
import sys
sys.path.append('.')
from code_understanding import SafeCommandExecutor

@pytest.fixture
def executor():
    return SafeCommandExecutor()

def test_valid_grep_command(executor):
    cmd = "grep -r -Hn calculation"
    valid, msg = executor.validate_command(cmd)
    assert valid
    assert msg == ""

def test_invalid_flag(executor):
    cmd = "grep -Z calculation"  # -Z not allowed
    valid, msg = executor.validate_command(cmd)
    assert not valid
    assert "Disallowed flag" in msg

def test_max_depth_violation(executor):
    # Test with valid path but exceeded maxdepth
    cmd = "find . -name *.cs -maxdepth 5"
    valid, msg = executor.validate_command(cmd)
    assert not valid
    assert "Max depth exceeds 3" in msg