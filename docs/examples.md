# CodeAnalyzer Examples

This document provides real-world examples of using CodeAnalyzer to understand and analyze codebases.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Advanced Queries](#advanced-queries)
- [Session Management](#session-management)
- [Configuration Examples](#configuration-examples)
- [Docker Workflows](#docker-workflows)
- [Use Cases](#use-cases)

## Getting Started

### First-Time Setup

```bash
# Install CodeAnalyzer
pip install -e .

# Set your API key
export OPENAI_API_KEY='sk-...'

# Create your first session
codeanalyzer session new --name my-first-session

# Verify it's active
codeanalyzer session list
```

## Basic Usage

### Example 1: Understanding a New Codebase

**Scenario**: You've just joined a project and need to understand how it works.

```bash
# Create a session for the project
codeanalyzer session new --name django-project

# Start with high-level questions
codeanalyzer ask "What is the main purpose of this codebase?"

# Drill into specific areas
codeanalyzer ask "How is the database configured?"
codeanalyzer ask "Where are the API endpoints defined?"
codeanalyzer ask "What authentication method is used?"
```

**Expected Output**:
```
Answer:
The codebase is a Django web application that provides a REST API for...

The database is configured in settings.py using PostgreSQL with the following
parameters...
```

### Example 2: Finding Specific Functionality

**Scenario**: You need to modify the authentication logic.

```bash
# Find authentication code
codeanalyzer ask "Show me all authentication-related code"

# Understand how it works
codeanalyzer ask "Explain the authentication flow step by step"

# Find tests
codeanalyzer ask "Where are the authentication tests located?"
```

### Example 3: Understanding Dependencies

**Scenario**: You want to understand how components interact.

```bash
# Find dependencies
codeanalyzer ask "What external libraries does this project use?"

# Understand usage
codeanalyzer ask "How is the requests library used in this project?"

# Find configuration
codeanalyzer ask "Where are third-party services configured?"
```

## Advanced Queries

### Code Structure Analysis

```bash
# Architecture questions
codeanalyzer ask "What design patterns are used in this codebase?"

# Module organization
codeanalyzer ask "How is the codebase organized? Describe the folder structure."

# Class hierarchies
codeanalyzer ask "Show me the inheritance hierarchy for the User model"
```

### Bug Investigation

```bash
# Find error handling
codeanalyzer ask "How are errors handled in the API layer?"

# Trace execution
codeanalyzer ask "What happens when a user login fails?"

# Find related code
codeanalyzer ask "Find all code related to password reset functionality"
```

### Code Quality Analysis

```bash
# Find code smells
codeanalyzer ask "Are there any TODO or FIXME comments in the code?"

# Security analysis
codeanalyzer ask "How is user input validated?"

# Test coverage
codeanalyzer ask "What parts of the authentication system are tested?"
```

### Refactoring Assistance

```bash
# Identify duplicates
codeanalyzer ask "Find duplicate code or similar functions"

# Understand dependencies before refactoring
codeanalyzer ask "What files depend on the User model?"

# Impact analysis
codeanalyzer ask "If I change the authenticate() function, what else will be affected?"
```

## Session Management

### Working with Multiple Projects

```bash
# Create separate sessions for different projects
codeanalyzer session new --name backend-api
codeanalyzer ask "How does the authentication work?"

codeanalyzer session new --name frontend-app
codeanalyzer ask "How does the login form work?"

# Switch between sessions
codeanalyzer session list
codeanalyzer session switch backend-api

# Continue previous analysis
codeanalyzer ask "What about authorization?"
```

### Session Customization

```bash
# Create session with custom chunk size for large files
codeanalyzer session new --name big-project --chunk-size 2000

# Create session for specific language
codeanalyzer session new --name python-only
codeanalyzer ask "Find all Python files with database queries"
```

### Session Cleanup

```bash
# List all sessions with metadata
codeanalyzer session list

# Clean up old sessions (> 30 days)
codeanalyzer session clean --days 30

# Clean up very old sessions (> 7 days)
codeanalyzer session clean --days 7
```

## Configuration Examples

### Exclude Directories

```bash
# List current exclusions
codeanalyzer config --list-exclude

# Add custom exclusions
codeanalyzer config --add-exclude .cache
codeanalyzer config --add-exclude vendor
codeanalyzer config --add-exclude dist

# Add file extension exclusions
codeanalyzer config --add-exclude .min.js
codeanalyzer config --add-exclude .map
```

### Adjust File Size Limits

```bash
# Set maximum file size to 5MB
codeanalyzer config --max-file-size 5242880

# Set to 1MB for faster processing
codeanalyzer config --max-file-size 1048576
```

### Remove Exclusions

```bash
# Remove an exclusion
codeanalyzer config --remove-exclude vendor

# Verify
codeanalyzer config --list-exclude
```

## Docker Workflows

### Analyzing External Projects

```bash
# Clone a project
git clone https://github.com/django/django.git
cd django

# Run CodeAnalyzer in Docker
docker run -it --rm \
  -e OPENAI_API_KEY='sk-...' \
  -v $(pwd):/code \
  -v django-sessions:/app/sessions \
  codeanalyzer:latest session new --name django-analysis

# Analyze the project
docker run -it --rm \
  -e OPENAI_API_KEY='sk-...' \
  -v $(pwd):/code \
  -v django-sessions:/app/sessions \
  codeanalyzer:latest ask "How does Django's ORM work?"
```

### Docker Compose for Daily Use

**docker-compose.override.yml**:
```yaml
version: '3.8'

services:
  codeanalyzer:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ../my-project:/code:ro  # Your project
      - ./sessions:/app/sessions
      - ./logs:/app/logs
```

**Usage**:
```bash
# Start service
docker-compose up -d

# Create session
docker-compose exec codeanalyzer session new --name my-project

# Ask questions
docker-compose exec codeanalyzer ask "Explain the architecture"

# Interactive mode
docker-compose exec codeanalyzer /bin/bash
```

### CI/CD Integration

```yaml
# .github/workflows/code-analysis.yml
name: Code Analysis

on: [push]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Code Analysis
        run: |
          docker run --rm \
            -e OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }} \
            -v ${{ github.workspace }}:/code \
            codeanalyzer:latest ask "Document the main components" > ANALYSIS.md

      - name: Upload Analysis
        uses: actions/upload-artifact@v3
        with:
          name: code-analysis
          path: ANALYSIS.md
```

## Use Cases

### Use Case 1: Onboarding New Developers

**Goal**: Help new team members understand the codebase quickly.

```bash
# Create onboarding session
codeanalyzer session new --name onboarding

# Generate overview
codeanalyzer ask "Provide a high-level overview of this project's architecture"

# Understand key components
codeanalyzer ask "What are the main components and how do they interact?"

# Find entry points
codeanalyzer ask "Where does the application start? Show me the entry points"

# Understand data flow
codeanalyzer ask "Explain the data flow from user request to database and back"

# Find documentation
codeanalyzer ask "Are there any README or documentation files? Summarize them"
```

### Use Case 2: Security Audit

**Goal**: Identify potential security issues.

```bash
# Create security audit session
codeanalyzer session new --name security-audit

# Input validation
codeanalyzer ask "How is user input validated? Are there any SQL injection risks?"

# Authentication
codeanalyzer ask "How are passwords stored and authenticated?"

# Authorization
codeanalyzer ask "How are permissions and access control implemented?"

# Secrets management
codeanalyzer ask "Are there any hardcoded secrets or API keys?"

# Sensitive data
codeanalyzer ask "How is sensitive user data protected?"
```

### Use Case 3: Debugging Production Issues

**Goal**: Quickly understand and fix a production bug.

```bash
# Create debugging session
codeanalyzer session new --name bug-investigation

# Understand the error
codeanalyzer ask "Find all code related to user registration errors"

# Trace execution
codeanalyzer ask "What happens when a user submits the registration form?"

# Find error handling
codeanalyzer ask "How are registration errors logged and reported?"

# Check recent changes
codeanalyzer ask "Find recent commits or changes to the registration code"
```

### Use Case 4: API Documentation

**Goal**: Generate API documentation from code.

```bash
# Create API docs session
codeanalyzer session new --name api-docs

# Find endpoints
codeanalyzer ask "List all API endpoints with their HTTP methods and purposes"

# Document parameters
codeanalyzer ask "For the /api/users endpoint, what parameters does it accept?"

# Response formats
codeanalyzer ask "What is the response format for successful user creation?"

# Error codes
codeanalyzer ask "What error codes can the API return and what do they mean?"
```

### Use Case 5: Migration Planning

**Goal**: Understand legacy code before migration.

```bash
# Create migration session
codeanalyzer session new --name legacy-migration

# Identify dependencies
codeanalyzer ask "What are all the external dependencies in this project?"

# Find deprecated code
codeanalyzer ask "Are there any deprecated functions or libraries being used?"

# Understand data structures
codeanalyzer ask "What database schema is being used?"

# Identify coupling
codeanalyzer ask "Which components are tightly coupled and need refactoring?"
```

### Use Case 6: Code Review Assistant

**Goal**: Assist in code review process.

```bash
# Review a PR
codeanalyzer session new --name pr-review

# Understand changes
codeanalyzer ask "What functionality does the new authentication middleware add?"

# Check patterns
codeanalyzer ask "Does this follow the existing authentication patterns in the codebase?"

# Find similar code
codeanalyzer ask "Are there similar implementations elsewhere that could be reused?"

# Impact analysis
codeanalyzer ask "What tests need to be updated for this change?"
```

## Best Practices

### Effective Queries

**✅ Good Queries**:
```bash
# Specific and focused
codeanalyzer ask "How does the JWT authentication middleware work?"

# Context-aware
codeanalyzer ask "Show me the error handling in the payment processing module"

# Action-oriented
codeanalyzer ask "What would break if I remove the User.get_permissions() method?"
```

**❌ Poor Queries**:
```bash
# Too vague
codeanalyzer ask "How does this work?"

# Too broad
codeanalyzer ask "Tell me everything about this codebase"

# Not specific enough
codeanalyzer ask "Find code"
```

### Session Organization

```bash
# Use descriptive session names
codeanalyzer session new --name user-auth-feature
codeanalyzer session new --name payment-integration
codeanalyzer session new --name security-review

# Not: session_1, session_2, test
```

### Iterative Analysis

```bash
# Start broad
codeanalyzer ask "What is the purpose of the services module?"

# Then narrow down
codeanalyzer ask "How does the EmailService class work?"

# Finally, specific
codeanalyzer ask "What happens when EmailService.send() fails?"
```

## Troubleshooting

### Issue: No Relevant Results

```bash
# Increase context retrieval
codeanalyzer ask "How does caching work?" --k 10

# Create new session with larger chunks
codeanalyzer session new --name detailed-analysis --chunk-size 2000
```

### Issue: Slow Responses

```bash
# Reduce file size limit
codeanalyzer config --max-file-size 1048576  # 1MB

# Add more exclusions
codeanalyzer config --add-exclude node_modules
codeanalyzer config --add-exclude .min.js
```

### Issue: Out of Context

```bash
# Create focused session for specific area
codeanalyzer session new --name auth-only

# Then ask specific questions
codeanalyzer ask "Find files in the auth/ directory"
codeanalyzer ask "How does the auth module work?"
```

## Next Steps

- Read the [Architecture Documentation](architecture.md)
- Review the [Contributing Guide](../CONTRIBUTING.md)
- Check the [Changelog](../CHANGELOG.md) for latest features
- Report issues on [GitHub](https://github.com/yourusername/codeanalyzer/issues)
