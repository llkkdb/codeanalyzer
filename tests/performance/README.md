# Performance Benchmarks

This directory contains performance benchmarks for the CodeAnalyzer system. These benchmarks help identify performance bottlenecks and track performance improvements over time.

## Running Benchmarks

### Run All Benchmarks

```bash
pytest tests/performance/ -v -s
```

The `-s` flag shows timing output in the console.

### Run Specific Benchmarks

```bash
# File processing benchmarks
pytest tests/performance/benchmark.py::test_file_reading_speed -v -s
pytest tests/performance/benchmark.py::test_file_chunking_speed -v -s

# Session management benchmarks
pytest tests/performance/benchmark.py::test_session_creation_speed -v -s
pytest tests/performance/benchmark.py::test_session_persistence_speed -v -s

# Parsing benchmarks
pytest tests/performance/benchmark.py::test_command_parsing_speed -v -s

# End-to-end benchmarks (slower)
pytest tests/performance/benchmark.py::test_end_to_end_performance -v -s
```

### Run with pytest-benchmark (Optional)

If you have `pytest-benchmark` installed, you can get more detailed statistics:

```bash
pip install pytest-benchmark
pytest tests/performance/ --benchmark-only
```

## Benchmark Categories

### 1. File Processing

- **File Reading**: Measures speed of reading large files
- **File Chunking**: Measures speed of splitting code into chunks
- **Binary Detection**: Measures speed of detecting binary vs. text files
- **Codebase Scanning**: Measures speed of scanning directory trees

### 2. Session Management

- **Session Creation**: Measures speed of creating new sessions
- **Session Persistence**: Measures speed of saving and loading sessions

### 3. Command Parsing

- **Command Output Parsing**: Measures speed of parsing grep, find, and ripgrep output

### 4. Configuration

- **Config Loading**: Measures speed of loading configuration files

### 5. End-to-End

- **Complete Workflow**: Measures performance of complete workflows including:
  - System initialization
  - Session creation
  - File scanning and processing
  - Session persistence

## Performance Thresholds

The benchmarks include performance assertions to catch regressions:

| Operation | Expected Max Time | Benchmark |
|-----------|------------------|-----------|
| File Reading (10K lines) | < 50ms | `test_file_reading_speed` |
| File Chunking | < 100ms | `test_file_chunking_speed` |
| Session Creation | < 50ms | `test_session_creation_speed` |
| Session Save/Load | < 200ms | `test_session_persistence_speed` |
| Command Parsing (1K results) | < 50ms | `test_command_parsing_speed` |
| Config Loading | < 10ms | `test_config_loading_speed` |
| Binary Detection | < 20ms | `test_binary_detection_speed` |
| Codebase Scan (100 files) | < 500ms | `test_large_codebase_scanning` |

## Interpreting Results

Each benchmark outputs timing statistics:

```
File Reading Benchmark:
  Average: 12.34ms
  Min: 10.12ms
  Max: 15.67ms
```

- **Average**: Mean execution time across all iterations
- **Min**: Fastest execution time
- **Max**: Slowest execution time

### Good Performance Indicators

✓ Average times well below thresholds
✓ Low variance between min and max
✓ Consistent performance across runs

### Performance Issues

✗ Average times near or above thresholds
✗ High variance between min and max
✗ Performance degrading over time

## Adding New Benchmarks

To add a new benchmark:

1. Create a test function starting with `test_`
2. Use the `benchmark()` helper for timing
3. Add performance assertions
4. Document expected performance in this README

Example:

```python
def test_my_operation_speed():
    """Benchmark my operation performance."""
    from codeanalyzer.my_module import MyOperation

    operation = MyOperation()

    def run_operation():
        return operation.process()

    stats = benchmark(run_operation, iterations=100)

    print(f"\nMy Operation Benchmark:")
    print(f"  Average: {stats['avg']*1000:.2f}ms")

    assert stats['avg'] < 0.1, f"Operation too slow: {stats['avg']}s"
```

## Continuous Performance Monitoring

For production deployments, consider:

1. **Baseline Recording**: Record benchmark results for each release
2. **Regression Detection**: Compare new results against baseline
3. **Performance Budgets**: Set maximum allowed times for operations
4. **CI Integration**: Run benchmarks in CI/CD pipeline

Example CI integration:

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run benchmarks
        run: |
          pytest tests/performance/ -v -s
```

## Profiling

For detailed profiling of specific operations:

```bash
# Profile with cProfile
python -m cProfile -o profile.stats -m pytest tests/performance/benchmark.py::test_file_reading_speed

# View profiling results
python -m pstats profile.stats
```

Or use line_profiler for line-by-line profiling:

```bash
pip install line_profiler
kernprof -l -v tests/performance/benchmark.py
```

## Performance Tips

### File Processing
- Use streaming for large files instead of loading entirely into memory
- Cache file metadata to avoid repeated stat() calls
- Use batch processing for multiple files

### Session Management
- Lazy load session data when possible
- Use efficient serialization formats (msgpack, pickle)
- Implement session caching for frequently accessed sessions

### Command Parsing
- Pre-compile regex patterns
- Use string methods instead of regex when possible
- Process output in streaming fashion for large outputs

## Troubleshooting

### Benchmarks Failing

If benchmarks fail due to performance:

1. Run benchmarks multiple times to account for variance
2. Check system load (other processes affecting performance)
3. Ensure test fixtures are not causing slowdowns
4. Profile the failing operation to identify bottleneck

### Inconsistent Results

Factors affecting benchmark consistency:

- System load from other processes
- CPU frequency scaling
- Background services (antivirus, indexing, etc.)
- Disk I/O contention
- Network activity

For most consistent results:
- Close unnecessary applications
- Run on dedicated test hardware
- Use fixed CPU frequency
- Run multiple iterations and average

## Resources

- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Python Profiling Guide](https://docs.python.org/3/library/profile.html)
