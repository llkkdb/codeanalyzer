"""
Performance benchmarks for CodeAnalyzer.

This module provides benchmarks for measuring the performance of key operations
in the CodeAnalyzer system. Run with:

    python -m pytest tests/performance/benchmark.py -v

Or run specific benchmarks:

    python -m pytest tests/performance/benchmark.py::test_file_processing_speed -v
"""

import time
import tempfile
from pathlib import Path
from typing import List, Callable
import pytest

# These benchmarks are optional and won't run in normal test suite
pytestmark = pytest.mark.benchmark


class PerformanceTimer:
    """Context manager for timing code execution."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"\n{self.name}: {self.elapsed:.4f}s")


def benchmark(func: Callable, iterations: int = 10) -> dict:
    """
    Run a function multiple times and collect timing statistics.

    Args:
        func: Function to benchmark
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'min': min(times),
        'max': max(times),
        'avg': sum(times) / len(times),
        'total': sum(times),
        'iterations': iterations
    }


@pytest.fixture
def large_python_file(tmp_path):
    """Create a large Python file for benchmarking."""
    file_path = tmp_path / "large_file.py"

    # Generate ~10,000 lines of Python code
    lines = []
    for i in range(1000):
        lines.append(f"def function_{i}(arg1, arg2, arg3):")
        lines.append(f'    """Function {i} documentation."""')
        lines.append(f"    result = arg1 + arg2 + arg3")
        lines.append(f"    return result * {i}")
        lines.append("")

        lines.append(f"class Class_{i}:")
        lines.append(f'    """Class {i} documentation."""')
        lines.append(f"    ")
        lines.append(f"    def method_{i}(self, x):")
        lines.append(f'        """Method {i} documentation."""')
        lines.append(f"        return x * {i}")

    file_path.write_text("\n".join(lines))
    return file_path


@pytest.fixture
def large_codebase(tmp_path):
    """Create a large codebase structure for benchmarking."""
    # Create directory structure
    dirs = [
        "src",
        "src/models",
        "src/views",
        "src/controllers",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "docs",
    ]

    for dir_name in dirs:
        (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)

    # Create 100 Python files with realistic content
    files_created = []
    for i in range(100):
        module_dir = dirs[i % len(dirs)]
        file_path = tmp_path / module_dir / f"module_{i}.py"

        content = f'''"""
Module {i} - Example module for benchmarking.
"""

import os
import sys
from typing import List, Dict, Optional

class Component_{i}:
    """Component class for module {i}."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def process(self, data: List[str]) -> Dict[str, int]:
        """Process data and return results."""
        results = {{}}
        for item in data:
            results[item] = len(item) * self.value
        return results

    def validate(self, input_data: Optional[str] = None) -> bool:
        """Validate input data."""
        if input_data is None:
            return False
        return len(input_data) > 0

def helper_function_{i}(x: int, y: int) -> int:
    """Helper function {i}."""
    return x + y + {i}
'''
        file_path.write_text(content)
        files_created.append(file_path)

    return tmp_path, files_created


def test_file_reading_speed(large_python_file):
    """Benchmark file reading performance."""
    from codeanalyzer.file_processor import FileProcessor

    processor = FileProcessor()

    def read_file():
        content = large_python_file.read_text()
        return len(content)

    stats = benchmark(read_file, iterations=100)

    print(f"\nFile Reading Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold (should read in < 50ms on average)
    assert stats['avg'] < 0.05, f"File reading too slow: {stats['avg']}s"


def test_file_chunking_speed(large_python_file):
    """Benchmark file chunking performance."""
    from codeanalyzer.file_processor import FileProcessor

    processor = FileProcessor()
    content = large_python_file.read_text()

    def chunk_content():
        chunks = processor.chunk_code(content, chunk_size=500)
        return len(chunks)

    stats = benchmark(chunk_content, iterations=50)

    print(f"\nFile Chunking Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.1, f"Chunking too slow: {stats['avg']}s"


def test_session_creation_speed(tmp_path):
    """Benchmark session creation performance."""
    from codeanalyzer.session_manager import SessionManager

    storage_dir = tmp_path / "sessions"
    manager = SessionManager(storage_dir=storage_dir)

    session_count = 0

    def create_session():
        nonlocal session_count
        session = manager.create_session(f"benchmark_session_{session_count}")
        session_count += 1
        return session

    stats = benchmark(create_session, iterations=50)

    print(f"\nSession Creation Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.05, f"Session creation too slow: {stats['avg']}s"


def test_session_persistence_speed(tmp_path):
    """Benchmark session save/load performance."""
    from codeanalyzer.session_manager import SessionManager
    from codeanalyzer.models import Session

    storage_dir = tmp_path / "sessions"
    manager = SessionManager(storage_dir=storage_dir)
    session = manager.create_session("benchmark_persist")

    # Add some data to the session
    for i in range(100):
        session.add_file(f"/path/to/file_{i}.py")

    def save_and_load():
        manager.persist_session(session)
        loaded = manager.load_session("benchmark_persist")
        return loaded

    stats = benchmark(save_and_load, iterations=20)

    print(f"\nSession Persistence Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.2, f"Session persistence too slow: {stats['avg']}s"


def test_command_parsing_speed():
    """Benchmark command output parsing performance."""
    from codeanalyzer.command_parser import CommandParser

    parser = CommandParser()

    # Create realistic grep output
    grep_output = "\n".join([
        f"/path/to/file_{i}.py:{i*10}:def function_{i}():"
        for i in range(1000)
    ])

    def parse_grep():
        results = parser.parse_grep_output(grep_output)
        return len(results)

    stats = benchmark(parse_grep, iterations=100)

    print(f"\nCommand Parsing Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.05, f"Command parsing too slow: {stats['avg']}s"


def test_config_loading_speed(tmp_path):
    """Benchmark configuration loading performance."""
    from codeanalyzer.config import Config

    config_file = tmp_path / "benchmark_config.json"
    config = Config(config_file=config_file)

    # Set various config values
    config.model_name = "gpt-4"
    config.chunk_size = 1000
    config.max_files = 500
    config.save()

    def load_config():
        new_config = Config(config_file=config_file)
        new_config.load()
        return new_config

    stats = benchmark(load_config, iterations=100)

    print(f"\nConfig Loading Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.01, f"Config loading too slow: {stats['avg']}s"


def test_large_codebase_scanning(large_codebase):
    """Benchmark scanning a large codebase."""
    from codeanalyzer.file_processor import FileProcessor

    codebase_dir, files = large_codebase
    processor = FileProcessor()

    def scan_codebase():
        python_files = list(codebase_dir.rglob("*.py"))
        return len(python_files)

    stats = benchmark(scan_codebase, iterations=50)

    print(f"\nCodebase Scanning Benchmark:")
    print(f"  Files: {len(files)}")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.5, f"Codebase scanning too slow: {stats['avg']}s"


def test_binary_detection_speed(tmp_path):
    """Benchmark binary file detection performance."""
    from codeanalyzer.file_processor import FileProcessor

    processor = FileProcessor()

    # Create test files
    text_file = tmp_path / "text.py"
    text_file.write_text("print('hello world')\n" * 1000)

    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(bytes(range(256)) * 100)

    def detect_binary():
        is_text = processor.is_text_file(text_file)
        is_binary = not processor.is_text_file(binary_file)
        return is_text and is_binary

    stats = benchmark(detect_binary, iterations=100)

    print(f"\nBinary Detection Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")
    print(f"  Min: {stats['min']*1000:.2f}ms")
    print(f"  Max: {stats['max']*1000:.2f}ms")

    # Assert performance threshold
    assert stats['avg'] < 0.02, f"Binary detection too slow: {stats['avg']}s"


@pytest.mark.slow
def test_end_to_end_performance(large_codebase, monkeypatch):
    """
    Benchmark complete end-to-end workflow performance.

    This test simulates a complete workflow:
    1. Initialize system
    2. Create session
    3. Scan codebase
    4. Process files
    5. Save session
    """
    # Mock OpenAI API key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    from codeanalyzer.code_understanding import CodeUnderstandingSystem

    codebase_dir, files = large_codebase

    with PerformanceTimer("End-to-End Workflow"):
        # Initialize system
        system = CodeUnderstandingSystem()

        # Create session
        session = system.session_manager.create_session("e2e_benchmark")

        # Scan and add files
        python_files = list(codebase_dir.rglob("*.py"))
        for file_path in python_files[:20]:  # Process first 20 files
            session.add_file(str(file_path))

        # Save session
        system.session_manager.persist_session(session)

        # Load session
        loaded_session = system.session_manager.load_session("e2e_benchmark")

        assert len(loaded_session.files) == 20

    print(f"\nProcessed {len(python_files[:20])} files")


if __name__ == "__main__":
    """Run benchmarks directly."""
    print("Running CodeAnalyzer Performance Benchmarks\n")
    print("=" * 60)
    pytest.main([__file__, "-v", "-s"])
