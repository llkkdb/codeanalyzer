import sys
import os
from pathlib import Path

# Get the path to the root of the project
project_root = Path(__file__).parent.parent.parent.absolute()

# Add it to the Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import works
from code_understanding import CodeSession, CodeUnderstandingSystem

@pytest.fixture
def system():
    return CodeUnderstandingSystem()

def test_basic_calculation_search(system):
    commands = system.generate_search_commands("calculation")
    assert len(commands) > 0
    assert any("grep" in cmd for cmd in commands)
    assert any("find" in cmd for cmd in commands)
    assert any("rg" in cmd for cmd in commands)

def test_multi_filetype_search(system):
    commands = system.generate_search_commands("tax calculation")
    assert any("*.{cs,xml,run}" in cmd for cmd in commands)

def test_path_depth_validation(system):
    commands = system.generate_search_commands("calculation")
    assert not any("/ -name" in cmd for cmd in commands)