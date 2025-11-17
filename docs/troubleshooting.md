# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with CodeAnalyzer.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Session Management](#session-management)
- [Query Issues](#query-issues)
- [Performance Problems](#performance-problems)
- [Error Messages](#error-messages)
- [Docker Issues](#docker-issues)
- [Development Issues](#development-issues)

## Installation Issues

### Problem: `ModuleNotFoundError` after installation

**Symptoms:**
```
ModuleNotFoundError: No module named 'codeanalyzer'
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep codeanalyzer
   ```

2. **Reinstall in editable mode:**
   ```bash
   pip uninstall codeanalyzer
   pip install -e .
   ```

3. **Check Python path:**
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

4. **Verify virtual environment:**
   ```bash
   which python
   which pip
   # Should point to venv/bin/python
   ```

### Problem: Dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solutions:**

1. **Use a fresh virtual environment:**
   ```bash
   python -m venv fresh_venv
   source fresh_venv/bin/activate
   pip install -e .
   ```

2. **Update pip:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Install specific versions:**
   ```bash
   pip install -e . --force-reinstall
   ```

## Configuration Problems

### Problem: API key not recognized

**Symptoms:**
```
ConfigurationError: OPENAI_API_KEY environment variable is not set
```

**Solutions:**

1. **Set environment variable:**
   ```bash
   export OPENAI_API_KEY='sk-your-key-here'
   ```

2. **Verify it's set:**
   ```bash
   echo $OPENAI_API_KEY
   ```

3. **Add to shell profile (permanent):**
   ```bash
   echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **For Docker:**
   ```bash
   docker run -e OPENAI_API_KEY='sk-your-key-here' ...
   ```

### Problem: Config file not loading

**Symptoms:**
- Changes to config file not taking effect
- Using default values instead of custom config

**Solutions:**

1. **Check config file location:**
   ```bash
   ls ~/.config/codeanalyzer/config.json
   ```

2. **Verify JSON syntax:**
   ```bash
   cat ~/.config/codeanalyzer/config.json | python -m json.tool
   ```

3. **Check environment variable:**
   ```bash
   echo $CODEANALYZER_CONFIG
   ```

4. **Recreate config:**
   ```bash
   codeanalyzer config --max-file-size 5000000
   ```

## Session Management

### Problem: "No active session" error

**Symptoms:**
```
Error: No active session!
Create a new session with:
  codeanalyzer session new --name my-session
```

**Solutions:**

1. **Create a new session:**
   ```bash
   codeanalyzer session new --name my-project
   ```

2. **List existing sessions:**
   ```bash
   codeanalyzer session list
   ```

3. **Switch to existing session:**
   ```bash
   codeanalyzer session switch my-project
   ```

### Problem: Session not persisting

**Symptoms:**
- Session disappears after closing terminal
- Lost context after restart

**Solutions:**

1. **Check session directory:**
   ```bash
   ls ~/.local/share/codeanalyzer/sessions/
   ```

2. **Verify permissions:**
   ```bash
   ls -la ~/.local/share/codeanalyzer/
   ```

3. **Check disk space:**
   ```bash
   df -h ~/.local/share/
   ```

4. **Set custom session directory:**
   ```bash
   export CODEANALYZER_SESSION_DIR=/path/to/sessions
   ```

### Problem: Cannot switch to session

**Symptoms:**
```
Error: Session 'my-session' not found!
```

**Solutions:**

1. **List all sessions:**
   ```bash
   codeanalyzer session list
   ```

2. **Check session directory:**
   ```bash
   ls ~/.local/share/codeanalyzer/sessions/
   ```

3. **Verify session name:**
   - Session names are case-sensitive
   - Check for typos

4. **Load session manually:**
   - Session may be persisted but not loaded
   - Try switching to it anyway - will auto-load

## Query Issues

### Problem: No results found

**Symptoms:**
```
Found 0 relevant files
```

**Solutions:**

1. **Increase context retrieval:**
   ```bash
   codeanalyzer ask "your query" --k 10
   ```

2. **Check excluded directories:**
   ```bash
   codeanalyzer config --list-exclude
   ```

3. **Remove unnecessary exclusions:**
   ```bash
   codeanalyzer config --remove-exclude vendor
   ```

4. **Try more specific queries:**
   ```bash
   # Instead of: "How does this work?"
   # Try: "How does the authentication module work?"
   ```

### Problem: Irrelevant results

**Symptoms:**
- Results don't match the query
- Getting wrong files

**Solutions:**

1. **Create focused session:**
   ```bash
   codeanalyzer session new --name focused
   # Then only query specific areas
   ```

2. **Be more specific in queries:**
   ```bash
   codeanalyzer ask "Find the User model in the models directory"
   ```

3. **Adjust chunk size:**
   ```bash
   codeanalyzer session new --name detailed --chunk-size 2000
   ```

### Problem: Incomplete answers

**Symptoms:**
- Answer cuts off mid-sentence
- Missing important details

**Solutions:**

1. **Increase context:**
   ```bash
   codeanalyzer ask "your query" --k 15
   ```

2. **Ask follow-up questions:**
   ```bash
   codeanalyzer ask "Can you provide more details about X?"
   ```

3. **Break down complex queries:**
   ```bash
   # Instead of one complex query, ask multiple simple ones
   codeanalyzer ask "Where is X defined?"
   codeanalyzer ask "How does X work?"
   codeanalyzer ask "What depends on X?"
   ```

## Performance Problems

### Problem: Slow query responses

**Symptoms:**
- Queries take >30 seconds
- Hanging on "Processing query"

**Solutions:**

1. **Reduce file size limit:**
   ```bash
   codeanalyzer config --max-file-size 1048576  # 1MB
   ```

2. **Add more exclusions:**
   ```bash
   codeanalyzer config --add-exclude node_modules
   codeanalyzer config --add-exclude .min.js
   codeanalyzer config --add-exclude dist
   ```

3. **Use smaller chunk size:**
   ```bash
   codeanalyzer session new --name fast --chunk-size 500
   ```

4. **Reduce context retrieval:**
   ```bash
   codeanalyzer ask "query" --k 3
   ```

### Problem: High memory usage

**Symptoms:**
- System running out of memory
- OOM errors

**Solutions:**

1. **Limit session count:**
   - Maximum 10 sessions by default
   - Old sessions are auto-cleaned after 24 hours

2. **Clean old sessions manually:**
   ```bash
   codeanalyzer session clean --days 7
   ```

3. **Reduce chunk size:**
   ```bash
   codeanalyzer session new --chunk-size 500
   ```

4. **Exclude large files:**
   ```bash
   codeanalyzer config --max-file-size 1048576
   ```

### Problem: API rate limiting

**Symptoms:**
```
LLMError: OpenAI API call failed: Rate limit exceeded
```

**Solutions:**

1. **Wait and retry:**
   - Wait 60 seconds and try again

2. **Upgrade OpenAI plan:**
   - Increase rate limits with higher tier

3. **Reduce query frequency:**
   - Space out queries
   - Be more deliberate with questions

4. **Use caching (future feature):**
   - Currently not implemented
   - Planned for future releases

## Error Messages

### Error: `CommandValidationError`

**Full error:**
```
CommandValidationError: Disallowed flag: -Z
```

**Cause:** Invalid command flag detected

**Solutions:**
- This is a security feature
- Only allowed flags can be used
- Check `ALLOWED_COMMANDS` in config.py

### Error: `CommandExecutionError`

**Full error:**
```
CommandExecutionError: Command execution failed: grep: ...
```

**Cause:** Command failed to execute

**Solutions:**

1. **Check if grep/find/rg is installed:**
   ```bash
   which grep
   which find
   which rg
   ```

2. **Install missing tools:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ripgrep

   # macOS
   brew install ripgrep
   ```

### Error: `FileProcessingError`

**Full error:**
```
FileProcessingError: Failed to process file.txt: ...
```

**Cause:** Cannot read or process file

**Solutions:**

1. **Check file permissions:**
   ```bash
   ls -l file.txt
   ```

2. **Verify file encoding:**
   - Binary files are automatically skipped
   - Check if file is actually text

3. **Check file size:**
   ```bash
   ls -lh file.txt
   # If too large, increase limit or exclude
   ```

### Error: `SessionNotFoundError`

**Full error:**
```
SessionNotFoundError: Session 'xyz' not found
```

**Cause:** Session doesn't exist

**Solutions:**

1. **List available sessions:**
   ```bash
   codeanalyzer session list
   ```

2. **Create the session:**
   ```bash
   codeanalyzer session new --name xyz
   ```

## Docker Issues

### Problem: Container won't start

**Symptoms:**
```
Error: Cannot connect to Docker daemon
```

**Solutions:**

1. **Start Docker:**
   ```bash
   sudo systemctl start docker
   ```

2. **Check Docker status:**
   ```bash
   docker ps
   ```

3. **Verify user permissions:**
   ```bash
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

### Problem: API key not working in Docker

**Symptoms:**
```
ConfigurationError: OPENAI_API_KEY not set
```

**Solutions:**

1. **Pass via environment variable:**
   ```bash
   docker run -e OPENAI_API_KEY='sk-...' ...
   ```

2. **Use docker-compose .env file:**
   ```bash
   echo "OPENAI_API_KEY=sk-..." > .env
   docker-compose up
   ```

3. **Check if variable is set in container:**
   ```bash
   docker run ... /bin/bash
   echo $OPENAI_API_KEY
   ```

### Problem: Volume mount issues

**Symptoms:**
- Code not visible in container
- Sessions not persisting

**Solutions:**

1. **Use absolute paths:**
   ```bash
   docker run -v /absolute/path:/code ...
   ```

2. **Check permissions:**
   ```bash
   ls -la /path/to/code
   ```

3. **Verify volume:**
   ```bash
   docker inspect <container_id>
   # Check Mounts section
   ```

## Development Issues

### Problem: Tests failing

**Symptoms:**
```
FAILED tests/unit/test_cli.py::test_something
```

**Solutions:**

1. **Install dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run specific test:**
   ```bash
   pytest tests/unit/test_cli.py::test_something -v
   ```

3. **Check test output:**
   ```bash
   pytest -v --tb=short
   ```

4. **Update dependencies:**
   ```bash
   pip install --upgrade -e ".[dev]"
   ```

### Problem: Import errors in tests

**Symptoms:**
```
ModuleNotFoundError: No module named 'codeanalyzer'
```

**Solutions:**

1. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

2. **Check PYTHONPATH:**
   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```

3. **Run from project root:**
   ```bash
   cd /path/to/codeanalyzer
   pytest tests/
   ```

### Problem: Pre-commit hooks failing

**Symptoms:**
```
black...........Failed
ruff............Failed
```

**Solutions:**

1. **Install pre-commit:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Run hooks manually:**
   ```bash
   pre-commit run --all-files
   ```

3. **Fix formatting:**
   ```bash
   black codeanalyzer/ tests/
   ruff check --fix codeanalyzer/
   ```

4. **Update hooks:**
   ```bash
   pre-commit autoupdate
   ```

## Getting More Help

### Check Logs

**Log location:**
```bash
ls ~/.local/share/codeanalyzer/logs/
tail -100 ~/.local/share/codeanalyzer/logs/codeanalyzer.log
```

**Increase log verbosity:**
```bash
export CODEANALYZER_LOG_LEVEL=DEBUG
codeanalyzer ask "query"
```

### Debug Mode

**Enable debug logging:**
```python
# In code
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Verbose output:**
```bash
codeanalyzer --help  # Check available options
```

### Report Issues

If you can't resolve the issue:

1. **Check existing issues:**
   - Visit GitHub issues page
   - Search for similar problems

2. **Create new issue with:**
   - Python version: `python --version`
   - CodeAnalyzer version: `pip show codeanalyzer`
   - Operating system
   - Error message (full traceback)
   - Steps to reproduce
   - Log output

3. **Include diagnostics:**
   ```bash
   # System info
   python --version
   pip list | grep -E "langchain|chromadb|click"

   # Config
   codeanalyzer config --list-exclude

   # Logs
   tail -50 ~/.local/share/codeanalyzer/logs/codeanalyzer.log
   ```

## Quick Fixes Checklist

Before reporting an issue, try these:

- [ ] Restart terminal/shell
- [ ] Deactivate and reactivate venv
- [ ] Reinstall package: `pip install -e . --force-reinstall`
- [ ] Clear sessions: `rm -rf ~/.local/share/codeanalyzer/sessions/*`
- [ ] Check API key: `echo $OPENAI_API_KEY`
- [ ] Update dependencies: `pip install --upgrade -e .`
- [ ] Check disk space: `df -h`
- [ ] Check permissions: `ls -la ~/.local/share/codeanalyzer/`
- [ ] Review logs: `tail ~/.local/share/codeanalyzer/logs/codeanalyzer.log`
- [ ] Test with simple query: `codeanalyzer ask "hello"`

## Additional Resources

- [Architecture Documentation](architecture.md)
- [Examples Guide](examples.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Changelog](../CHANGELOG.md)
- [GitHub Issues](https://github.com/yourusername/codeanalyzer/issues)
