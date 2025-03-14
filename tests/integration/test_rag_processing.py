import pytest
import sys
sys.path.append('.')
from code_understanding import CodeUnderstandingSystem, CodeSession

@pytest.fixture
def system():
    return CodeUnderstandingSystem()

def test_file_chunking(system):
    session = CodeSession("test")
    files = [
        "tests/fixtures/navigator/std/Data/Config/XmlForm/ProductConfigurator/RuleBasedConfigurator/CalculationMnt.cs",
        "tests/fixtures/navigator/std/Data/Config/XmlForm/ProductConfigurator/RuleBasedConfigurator/CalculationStepMnt.xml"
    ]
    session.add_files([Path(f) for f in files])
    
    assert len(session.context_files) == 2
    assert session.vector_store is not None

def test_query_context(system):
    session = CodeSession("test")
    files = [
        "tests/fixtures/navigator/std/Data/Config/XmlForm/ProductConfigurator/RuleBasedConfigurator/CalculationMnt.cs"
    ]
    session.add_files([Path(f) for f in files])
    
    result = session.query_context("tax calculation")
    assert "CalculateTax" in result
    assert "20% tax" in result