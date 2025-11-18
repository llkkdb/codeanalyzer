.PHONY: help install test test-unit test-integration test-coverage benchmark lint format clean docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package and dependencies
	pip install -e ".[dev]"

install-prod: ## Install package without dev dependencies
	pip install -e .

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run only unit tests
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	pytest tests/integration/ -v -m integration

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=codeanalyzer --cov-report=term-missing --cov-report=html

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -- -v

benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	pytest tests/performance/ -v -s -m benchmark

benchmark-quick: ## Run quick benchmarks only (exclude slow)
	pytest tests/performance/ -v -s -m "benchmark and not slow"

lint: ## Run linters (ruff, mypy)
	@echo "Running ruff..."
	ruff check codeanalyzer/ tests/
	@echo "Running mypy..."
	mypy codeanalyzer/

lint-fix: ## Auto-fix linting issues
	ruff check --fix codeanalyzer/ tests/

format: ## Format code with black and isort
	@echo "Running black..."
	black codeanalyzer/ tests/
	@echo "Running isort..."
	isort codeanalyzer/ tests/

format-check: ## Check code formatting without making changes
	@echo "Checking with black..."
	black --check codeanalyzer/ tests/
	@echo "Checking with isort..."
	isort --check codeanalyzer/ tests/

security: ## Run security checks
	@echo "Running bandit..."
	bandit -r codeanalyzer/
	@echo "Running safety..."
	safety check

clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ dist/ build/

docker-build: ## Build Docker image
	docker build -t codeanalyzer:latest .

docker-run: ## Run Docker container
	docker run -it --rm \
		-e OPENAI_API_KEY='${OPENAI_API_KEY}' \
		-v $$(pwd)/code-to-analyze:/code:ro \
		-v codeanalyzer-sessions:/app/sessions \
		codeanalyzer:latest

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

docs-serve: ## Serve documentation locally (requires mkdocs)
	@echo "Serving documentation at http://127.0.0.1:8000"
	@echo "Note: Install mkdocs with: pip install mkdocs mkdocs-material"
	mkdocs serve

ci: lint test-coverage ## Run CI checks locally

all: clean install lint test-coverage ## Run all checks

# Development workflows
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	@echo "Installing pre-commit hooks..."
	pre-commit install || echo "Warning: pre-commit not available"
	@echo "Development environment ready!"

quick-check: lint-fix format test-unit ## Quick check before commit

release-check: clean install lint test-coverage security benchmark ## Full release check
