# Contributing to CodeAnalyzer

Thank you for your interest in contributing to CodeAnalyzer! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/codeanalyzer.git
   cd codeanalyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run formatters and linters:
```bash
black codeanalyzer tests
ruff check codeanalyzer tests
mypy codeanalyzer
```

## Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=codeanalyzer --cov-report=html
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linters:**
   ```bash
   pytest
   black codeanalyzer tests
   ruff check codeanalyzer tests
   mypy codeanalyzer
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Reporting Bugs

Please use GitHub Issues to report bugs. Include:
- A clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)

## Feature Requests

We welcome feature requests! Please use GitHub Issues and include:
- Clear description of the feature
- Use case and rationale
- Proposed implementation (if you have ideas)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
