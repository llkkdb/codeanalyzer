"""Unit tests for command output parser."""
import pytest
from pathlib import Path
from unittest.mock import Mock

from codeanalyzer.core.parser import CommandOutputParser


@pytest.fixture
def parser():
    """Create a CommandOutputParser instance."""
    return CommandOutputParser()


class TestGrepParsing:
    """Tests for grep output parsing."""

    def test_parse_grep_simple(self, parser):
        """Test parsing simple grep output."""
        output = """
        file1.py:10:def test_function():
        file2.py:20:class TestClass:
        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 2
        assert Path("file1.py") in result
        assert Path("file2.py") in result

    def test_parse_grep_with_path(self, parser):
        """Test parsing grep output with paths."""
        output = """
        /path/to/file1.py:10:def test():
        /path/to/dir/file2.py:20:class Test:
        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 2
        assert Path("/path/to/file1.py") in result
        assert Path("/path/to/dir/file2.py") in result

    def test_parse_grep_relative_paths(self, parser):
        """Test parsing grep output with relative paths."""
        output = """
        ./src/file1.py:10:def test():
        ../other/file2.py:20:class Test:
        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 2
        # Relative paths should be converted to Path objects
        assert any("file1.py" in str(p) for p in result)
        assert any("file2.py" in str(p) for p in result)

    def test_parse_grep_empty_output(self, parser):
        """Test parsing empty grep output."""
        result = parser.parse("grep -rHn nonexistent", "")

        assert len(result) == 0

    def test_parse_grep_with_binary_files(self, parser):
        """Test grep output with binary file warnings."""
        output = """
        Binary file matches.pyc matches
        file1.py:10:def test():
        """
        result = parser.parse("grep -rHn test", output)

        # Should filter out binary file matches
        assert len(result) == 1
        assert Path("file1.py") in result

    def test_parse_grep_deduplication(self, parser):
        """Test that duplicate files are deduplicated."""
        output = """
        file1.py:10:match 1
        file1.py:20:match 2
        file2.py:30:match 3
        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 2
        assert Path("file1.py") in result
        assert Path("file2.py") in result


class TestFindParsing:
    """Tests for find output parsing."""

    def test_parse_find_simple(self, parser):
        """Test parsing simple find output."""
        output = """
        ./file1.py
        ./dir/file2.py
        ./dir/subdir/file3.py
        """
        result = parser.parse("find . -name *.py", output)

        assert len(result) == 3
        assert any("file1.py" in str(p) for p in result)
        assert any("file2.py" in str(p) for p in result)
        assert any("file3.py" in str(p) for p in result)

    def test_parse_find_absolute_paths(self, parser):
        """Test parsing find output with absolute paths."""
        output = """
        /home/user/project/file1.py
        /home/user/project/src/file2.py
        """
        result = parser.parse("find /home/user/project -name *.py", output)

        assert len(result) == 2
        assert Path("/home/user/project/file1.py") in result
        assert Path("/home/user/project/src/file2.py") in result

    def test_parse_find_empty(self, parser):
        """Test parsing empty find output."""
        result = parser.parse("find . -name nonexistent", "")

        assert len(result) == 0

    def test_parse_find_with_directories(self, parser):
        """Test find output with directory entries."""
        output = """
        ./src
        ./src/file1.py
        ./tests
        ./tests/test_file.py
        """
        result = parser.parse("find . -name *", output)

        # Should include all paths (both files and directories)
        assert len(result) == 4

    def test_parse_find_deduplication(self, parser):
        """Test that duplicate paths are deduplicated."""
        output = """
        ./file1.py
        ./file1.py
        ./file2.py
        """
        result = parser.parse("find . -name *.py", output)

        assert len(result) == 2


class TestRipgrepParsing:
    """Tests for ripgrep (rg) output parsing."""

    def test_parse_rg_files_list(self, parser):
        """Test parsing rg --files output."""
        output = """
        file1.py
        src/file2.py
        tests/test_file.py
        """
        result = parser.parse("rg --files", output)

        assert len(result) == 3
        assert any("file1.py" in str(p) for p in result)
        assert any("file2.py" in str(p) for p in result)
        assert any("test_file.py" in str(p) for p in result)

    def test_parse_rg_search_results(self, parser):
        """Test parsing rg search output with line numbers."""
        output = """
        file1.py:10:def test():
        file2.py:20:class Test:
        file3.py:30:    return test
        """
        result = parser.parse("rg test", output)

        assert len(result) == 3
        assert Path("file1.py") in result
        assert Path("file2.py") in result
        assert Path("file3.py") in result

    def test_parse_rg_with_json_output(self, parser):
        """Test parsing rg JSON output."""
        # rg can output JSON format, parser should handle it gracefully
        output = """
        {"type":"match","data":{"path":{"text":"file1.py"}}}
        """
        # Parser should extract path from standard format lines
        # JSON lines might be ignored or handled separately
        result = parser.parse("rg --json test", output)

        # Should handle gracefully (may be empty or extract paths)
        assert isinstance(result, list)

    def test_parse_rg_empty(self, parser):
        """Test parsing empty rg output."""
        result = parser.parse("rg nonexistent", "")

        assert len(result) == 0


class TestParserEdgeCases:
    """Edge case tests for parser."""

    def test_parse_unknown_command(self, parser):
        """Test parsing output from unknown command."""
        output = """
        some_file.txt
        another_file.txt
        """
        # Should fall back to generic parsing
        result = parser.parse("unknown command", output)

        assert len(result) == 2

    def test_parse_multiline_matches(self, parser):
        """Test parsing output with multiline matches."""
        output = """
        file1.py:10:def test():
        file1.py-11-    pass
        file1.py:12:    return None
        file2.py:20:class Test:
        """
        result = parser.parse("grep -A1 test", output)

        # Should extract unique files
        assert len(result) == 2
        assert Path("file1.py") in result
        assert Path("file2.py") in result

    def test_parse_special_characters_in_path(self, parser):
        """Test parsing paths with special characters."""
        output = """
        ./path with spaces/file.py:10:test
        ./path-with-dashes/file.py:20:test
        ./path_with_underscores/file.py:30:test
        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 3

    def test_parse_windows_paths(self, parser):
        """Test parsing Windows-style paths."""
        output = r"""
        C:\Users\test\file1.py:10:test
        C:\Users\test\dir\file2.py:20:test
        """
        result = parser.parse("grep -rHn test", output)

        # Should handle Windows paths
        assert len(result) == 2

    def test_parse_mixed_path_formats(self, parser):
        """Test parsing mixed path formats."""
        output = """
        file1.py:10:test
        ./file2.py:20:test
        /absolute/file3.py:30:test
        ../relative/file4.py:40:test
        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 4

    def test_parse_with_whitespace(self, parser):
        """Test parsing output with extra whitespace."""
        output = """

        file1.py:10:test

        file2.py:20:test

        """
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 2

    def test_parse_filters_invalid_lines(self, parser):
        """Test that invalid lines are filtered out."""
        output = """
        file1.py:10:test
        This is not a valid line
        file2.py:20:test
        Another invalid line
        """
        result = parser.parse("grep -rHn test", output)

        # Should only include valid file paths
        assert len(result) == 2
        assert Path("file1.py") in result
        assert Path("file2.py") in result

    def test_parse_large_output(self, parser):
        """Test parsing large output."""
        # Generate 1000 lines of grep output
        lines = [f"file{i}.py:{i}:test" for i in range(1000)]
        output = "\n".join(lines)

        result = parser.parse("grep -rHn test", output)

        assert len(result) == 1000

    def test_parse_preserves_path_type(self, parser):
        """Test that parser returns Path objects."""
        output = "file1.py:10:test"
        result = parser.parse("grep -rHn test", output)

        assert len(result) == 1
        assert isinstance(result[0], Path)

    def test_parse_handles_none_output(self, parser):
        """Test parsing handles None output gracefully."""
        # Parser should handle None without crashing
        # Depending on implementation, might raise or return empty
        try:
            result = parser.parse("grep test", None)
            assert isinstance(result, list)
        except (TypeError, AttributeError):
            # If it raises, that's also acceptable
            pass

    def test_parse_handles_unicode(self, parser):
        """Test parsing output with unicode characters."""
        output = """
        файл.py:10:тест
        文件.py:20:测试
        """
        result = parser.parse("grep -rHn test", output)

        # Should handle unicode in filenames
        assert len(result) == 2
