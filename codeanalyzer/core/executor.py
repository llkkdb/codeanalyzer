"""
Safe command execution with validation and security controls.
"""
import asyncio
import shlex
import subprocess
import time
import threading
import logging
from pathlib import Path
from typing import List, Tuple

from ..config import Config
from ..exceptions import CommandValidationError, CommandExecutionError
from .parser import CommandOutputParser

logger = logging.getLogger("codeanalyzer.executor")


class SafeCommandExecutor:
    """
    Validate and execute commands safely with better timeout handling.

    Features:
    - Command allowlist validation
    - Flag validation
    - Path restriction enforcement
    - Result caching
    - Timeout protection
    """

    def __init__(self, config: Config = None, timeout: int = None):
        """
        Initialize safe command executor.

        Args:
            config: Configuration object. If None, uses defaults.
            timeout: Command timeout in seconds. If None, uses config default.
        """
        self.config = config or Config()
        self.allow_list = self.config.allowed_commands
        self.timeout = timeout or self.config.command_timeout
        self.parser = CommandOutputParser(self.config)
        self._command_cache = {}  # Cache results for repeated commands
        self._cache_lock = threading.Lock()

    def validate_command(self, command: str) -> str:
        """
        Check if command is allowed with improved security checks.

        Args:
            command: Command string to validate.

        Returns:
            Validated (and possibly modified) command string.

        Raises:
            CommandValidationError: If command fails validation.
        """
        parts = command.split()
        if not parts:
            raise CommandValidationError("Empty command")

        cmd = parts[0]
        if cmd not in self.allow_list:
            raise CommandValidationError(f"Command {cmd} not allowed")

        # Check flags
        allowed_flags = self.allow_list[cmd]["flags"]
        for part in parts[1:]:
            if part.startswith("-"):
                if cmd == "grep" and len(part) > 1 and part[1] != "-":
                    # Split combined flags like -Hn into [-H, -n]
                    flags = [f"-{char}" for char in part[1:]]
                    for flag in flags:
                        if flag not in allowed_flags:
                            raise CommandValidationError(
                                f"Disallowed flag {flag} for {cmd}"
                            )
                elif part not in allowed_flags and not part.startswith("--"):
                    raise CommandValidationError(f"Disallowed flag {part} for {cmd}")

        # Enhanced path and argument validation
        excluded_patterns = [
            "/etc", "/var", "/usr", "/bin", "/sbin",
            "~", "$", "|", ";", ">", "<", "&", "`"
        ]
        for pattern in excluded_patterns:
            if pattern in command:
                raise CommandValidationError(
                    f"Command contains disallowed pattern: {pattern}"
                )

        # Command-specific validation
        if cmd == "find":
            # Check for root path usage
            if any(p.startswith("/") for p in parts[1:] if not p.startswith("-")):
                raise CommandValidationError(
                    "Find command cannot search from root directory"
                )

            # Verify maxdepth parameter
            max_depth = self.allow_list[cmd]["max_depth"]
            if "-maxdepth" in parts:
                idx = parts.index("-maxdepth") + 1
                if idx < len(parts):
                    if not parts[idx].isdigit():
                        raise CommandValidationError("Invalid maxdepth value")
                    if int(parts[idx]) > max_depth:
                        raise CommandValidationError(
                            f"Max depth exceeds {max_depth}"
                        )
                else:
                    raise CommandValidationError("Missing maxdepth value")
            else:
                # Automatically add maxdepth to make command safer
                command += f" -maxdepth {max_depth}"

        return command  # Return potentially modified command

    async def execute_async(self, command: str) -> List[Path]:
        """
        Execute command asynchronously with caching.

        Args:
            command: Command to execute.

        Returns:
            List of Path objects found by the command.

        Raises:
            CommandExecutionError: If command execution fails.
        """
        # Check cache first
        with self._cache_lock:
            cache_key = command
            cache_entry = self._command_cache.get(cache_key)
            if cache_entry:
                cache_time, result = cache_entry
                # Cache valid for 5 minutes
                if time.time() - cache_time < 300:
                    logger.debug(f"Cache hit for command: {command}")
                    return result

        # Validate command
        try:
            validated_command = self.validate_command(command)
        except CommandValidationError as e:
            logger.warning(f"Invalid command: {e}")
            raise

        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            # SECURITY FIX: Use shlex.split() and shell=False to prevent injection
            cmd_args = shlex.split(validated_command)

            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd_args,
                    shell=False,  # FIXED: Use shell=False for security
                    check=False,  # Don't raise exception, handle errors manually
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout
                )
            )

            if result.returncode != 0:
                logger.warning(
                    f"Command failed with code {result.returncode}: {result.stderr}"
                )
                paths = []
            else:
                paths = self.parser.parse(command, result.stdout)

            # Update cache
            with self._cache_lock:
                self._command_cache[cache_key] = (time.time(), paths)

            # Clean old cache entries if cache is too large
            if len(self._command_cache) > 100:
                self._clean_cache()

            return paths

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {self.timeout}s: {command}"
            logger.warning(error_msg)
            raise CommandExecutionError(error_msg)
        except (OSError, ValueError) as e:
            error_msg = f"Command execution error: {e}"
            logger.error(error_msg)
            raise CommandExecutionError(error_msg)

    def execute(self, command: str) -> List[Path]:
        """
        Synchronous wrapper for execute_async.

        Args:
            command: Command to execute.

        Returns:
            List of Path objects found by the command.
        """
        return asyncio.run(self.execute_async(command))

    def _clean_cache(self):
        """Remove old cache entries."""
        now = time.time()
        with self._cache_lock:
            old_keys = [
                k for k, (t, _) in self._command_cache.items()
                if now - t > 300
            ]
            for k in old_keys:
                del self._command_cache[k]
            logger.debug(f"Cleaned {len(old_keys)} cache entries")
