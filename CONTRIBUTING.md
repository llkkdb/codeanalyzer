# Contributing to CodeAnalyzer

Thank you for your interest in contributing to CodeAnalyzer! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- OpenAI API key (for testing with real API calls)

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/codeanalyzer.git
cd codeanalyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
make dev-setup
# Or: pip install -e ".[dev]"

# Set up environment
export OPENAI_API_KEY="your-api-key"

# Verify installation
make test
```

## Code Style

CodeAnalyzer follows PEP 8 with Black formatting (line length: 100).

### Running Style Checks

```bash
make lint        # Check style
make lint-fix    # Auto-fix issues
make format      # Format code
```

## Testing

### Test Structure

```
tests/
├── unit/              # Fast, isolated tests
├── integration/       # Component interaction tests  
└── performance/       # Performance benchmarks
```

### Running Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-coverage     # With coverage report
make benchmark         # Performance benchmarks
```

### Test Guidelines

1. Use fixtures from `tests/conftest.py`
2. Mark tests appropriately (`@pytest.mark.unit`, `@pytest.mark.integration`)
3. Mock external dependencies
4. Write descriptive test names
5. Keep tests focused

## Submitting Changes

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>
```

**Types:** feat, fix, docs, test, refactor, perf, chore

**Example:**
```
feat(session): Add session export functionality

Add ability to export session data to JSON format.

Closes #123
```

### Pull Request Process

1. Create feature branch
2. Make changes with tests
3. Run `make ci` locally
4. Push and create PR
5. Address review feedback

### PR Checklist

- [ ] Tests pass
- [ ] Code formatted
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] No merge conflicts

## Getting Help

- Issues: GitHub Issues
- Questions: GitHub Discussions
- Docs: `docs/` directory

Thank you for contributing! 🎉
