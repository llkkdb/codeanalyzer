# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-17

### Added
- Initial release of CodeAnalyzer
- Modular package structure with separate modules for core, storage, LLM, and CLI
- Custom exception classes for better error handling
- Configuration management with file-based and environment variable support
- Comprehensive logging with rotation support
- Type hints throughout the codebase
- Docstrings for all public APIs
- Pre-commit hooks configuration
- Development tools setup (Black, Ruff, MyPy, Pytest)
- pyproject.toml for modern Python packaging
- CONTRIBUTING.md guidelines
- Unit and integration tests

### Changed
- Refactored monolithic 933-line file into modular structure:
  - `codeanalyzer/core/` - Command execution and parsing
  - `codeanalyzer/storage/` - Session management
  - `codeanalyzer/llm/` - LLM integration
  - `codeanalyzer/cli.py` - Command-line interface
  - `codeanalyzer/config.py` - Configuration management
  - `codeanalyzer/exceptions.py` - Custom exceptions

### Security
- **CRITICAL FIX:** Replaced `shell=True` with `shlex.split()` and `shell=False` in command execution to prevent shell injection attacks
- Added OPENAI_API_KEY validation at system initialization
- Improved command validation with better path sanitization
- Enhanced error handling to prevent information leakage

### Improved
- Better async/sync patterns throughout the codebase
- Enhanced error messages with specific exception types
- Improved logging with rotation and configurable levels
- Better type safety with comprehensive type hints
- More robust file processing with better error handling
- Session management with automatic cleanup
- Configuration file support for runtime customization

### Developer Experience
- Added comprehensive documentation
- Improved code organization for better maintainability
- Added development dependencies in pyproject.toml
- Set up pre-commit hooks for automated quality checks
- Added testing infrastructure

## [Unreleased]

### Planned
- Improved test coverage (target: >80%)
- Performance optimizations
- Additional LLM provider support (Anthropic, local models)
- Enhanced CLI features
- Better documentation with examples
- API documentation generation (Sphinx)

---

[0.1.0]: https://github.com/yourusername/codeanalyzer/releases/tag/v0.1.0
