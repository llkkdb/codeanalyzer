"""
Command output parsing utilities.
"""
import re
import logging
from pathlib import Path
from typing import List

from ..config import Config

logger = logging.getLogger("codeanalyzer.parser")


class CommandOutputParser:
    """
    Parse command outputs to extract file paths with improved caching.

    Supports parsing output from find, grep, and ripgrep commands.
    """

    # Compiled regex patterns - do this once at module level for performance
    PATTERNS = {
        "grep": re.compile(r"^(.*?):\d+:"),  # Extract path from grep -Hn output
        "find": None,  # Direct path output
        "rg": re.compile(r"^(.*?)(:\d+){2}")  # Ripgrep format
    }

    def __init__(self, config: Config = None):
        """
        Initialize parser with configuration.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or Config()

    def parse(self, command: str, output: str) -> List[Path]:
        """
        Parse command output based on command type with better error handling.

        Args:
            command: The command that was executed.
            output: The output from the command.

        Returns:
            List of Path objects for files found in the output.
        """
        cmd_type = command.split()[0]
        parser = self.PATTERNS.get(cmd_type)

        paths = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                if parser is None:  # Direct output (like find)
                    path_str = line
                elif isinstance(parser, re.Pattern):
                    match = parser.match(line)
                    if match:
                        path_str = match.group(1)
                    else:
                        continue
                else:  # Fallback for unknown commands
                    path_str = line.split(":")[0]

                path = Path(path_str).resolve()
                if path.exists() and path.is_file() and self._should_include_file(path):
                    paths.append(path)
            except (OSError, ValueError) as e:
                logger.debug(f"Error parsing path from '{line}': {e}")
                continue

        return paths

    def _should_include_file(self, path: Path) -> bool:
        """
        Check if file should be included based on size and extension.

        Args:
            path: Path to check.

        Returns:
            True if file should be included, False otherwise.
        """
        # Skip excluded extensions
        if path.suffix.lower() in self.config.excluded_extensions:
            return False

        # Skip files in excluded directories
        for excluded in self.config.excluded_dirs:
            if excluded in path.parts:
                return False

        # Skip files that are too large
        try:
            if path.stat().st_size > self.config.max_file_size:
                logger.info(f"Skipping file {path}: exceeds size limit")
                return False
        except OSError as e:
            logger.warning(f"Cannot stat file {path}: {e}")
            return False

        return True
