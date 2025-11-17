FROM python:3.11-slim

LABEL org.opencontainers.image.title="CodeAnalyzer"
LABEL org.opencontainers.image.description="An intelligent code understanding system for exploring and analyzing codebases"
LABEL org.opencontainers.image.version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY codeanalyzer/ ./codeanalyzer/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create directories for sessions and logs
RUN mkdir -p /app/sessions /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CODEANALYZER_CONFIG=/app/config/codeanalyzer.json
ENV CODEANALYZER_LOG_DIR=/app/logs
ENV CODEANALYZER_SESSION_DIR=/app/sessions

# Volume for persistent data
VOLUME ["/app/sessions", "/app/logs", "/app/config"]

# Volume for code to analyze
VOLUME ["/code"]

# Set working directory to /code for analyzing mounted code
WORKDIR /code

# Entry point
ENTRYPOINT ["codeanalyzer"]

# Default command
CMD ["--help"]
