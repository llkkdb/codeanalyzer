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
    cmd = "find / -name *.cs"  # Exceeds max depth
    valid, msg = executor.validate_command(cmd)
    assert not valid
    assert "Path depth exceeded" in msg