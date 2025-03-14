import sys
import os
from pathlib import Path
import pytest

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

def test_file_chunking():
    """Test if files are properly chunked when added to a session"""
    session = CodeSession("test_chunking")
    
    # Create a test file with known content
    test_file = Path("test_chunk.txt")
    content = "a" * 2000  # 2000 characters
    test_file.write_text(content)
    
    with patch("code_understanding.Chroma.from_documents") as mock_from_docs:
        mock_from_docs.return_value = Mock()
        # Add the file with a chunk size of 500
        session.add_files([test_file], chunk_size=500)
        
        # Verify the file was added to the context
        assert test_file in session.context_files
        
        # Verify the chunking (4 chunks of 500 characters)
        args, _ = mock_from_docs.call_args
        assert len(args[0]) == 4
        
    # Clean up
    test_file.unlink()

def test_query_context(system):
    """Test if querying context returns expected results"""
    session = CodeSession("test")
    
    # Mock file content with "CalculateTax" in it
    test_file = Path("test_tax.cs")
    test_file.write_text("""
    public class TaxCalculator {
        public decimal CalculateTax(decimal amount) {
            return amount * 0.2m;
        }
    }
    """)
    
    # Mock the vector store
    mock_results = [
        Mock(
            page_content="public decimal CalculateTax(decimal amount) { return amount * 0.2m; }",
            metadata={"source": str(test_file), "chunk": 1}
        )
    ]
    
    with patch("code_understanding.Chroma.from_documents") as mock_from_docs:
        mock_store = Mock()
        mock_store.similarity_search.return_value = mock_results
        mock_from_docs.return_value = mock_store
        
        # Add file to session
        session.add_files([test_file])
        
        # Query context
        result = session.query_context("tax calculation")
        
        # Verify results
        assert "CalculateTax" in result
        assert str(test_file) in result
    
    # Clean up
    test_file.unlink()