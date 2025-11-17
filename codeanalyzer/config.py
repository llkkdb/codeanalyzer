"""
Configuration management for codeanalyzer.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from logging.handlers import RotatingFileHandler

from .exceptions import ConfigurationError


# Default configuration
ALLOWED_COMMANDS = {
    "find": {"flags": ["-name", "-type", "-path", "-exec", "-maxdepth"], "max_depth": 3},
    "grep": {"flags": ["-r", "-i", "-l", "-H", "-n"], "max_depth": None},
    "rg": {"flags": ["--files", "-g", "-l"], "max_depth": 3}
}

MAX_FILE_SIZE = 1024 * 1024  # 1 MB
EXCLUDED_DIRS = [".git", "node_modules", "__pycache__", "venv", ".idea", ".vscode"]
EXCLUDED_EXTENSIONS = [".jpg", ".png", ".gif", ".bin", ".exe", ".dll", ".so", ".pyc"]
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_COMMAND_TIMEOUT = 10
DEFAULT_MAX_SESSIONS = 10


class Config:
    """
    Configuration manager for codeanalyzer.

    Handles loading configuration from files and environment variables,
    with support for runtime updates.
    """

    def __init__(self, config_path: Path = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        self.config_path = config_path or Path("code_system_config.json")
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except (json.JSONDecodeError, IOError) as e:
                raise ConfigurationError(f"Failed to load config from {self.config_path}: {e}")

        # Return default configuration
        return {
            "allowed_commands": ALLOWED_COMMANDS,
            "max_file_size": MAX_FILE_SIZE,
            "excluded_dirs": EXCLUDED_DIRS.copy(),
            "excluded_extensions": EXCLUDED_EXTENSIONS.copy(),
            "default_chunk_size": DEFAULT_CHUNK_SIZE,
            "command_timeout": DEFAULT_COMMAND_TIMEOUT,
            "max_sessions": DEFAULT_MAX_SESSIONS,
            "log_level": os.getenv("CODE_ANALYZER_LOG_LEVEL", "INFO"),
            "log_dir": os.getenv("CODE_ANALYZER_LOG_DIR", "logs"),
        }

    def save(self):
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(self._config, indent=2))
        except IOError as e:
            raise ConfigurationError(f"Failed to save config to {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    @property
    def allowed_commands(self) -> Dict:
        """Get allowed commands configuration."""
        return self._config.get("allowed_commands", ALLOWED_COMMANDS)

    @property
    def max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return self._config.get("max_file_size", MAX_FILE_SIZE)

    @property
    def excluded_dirs(self) -> List[str]:
        """Get list of excluded directories."""
        return self._config.get("excluded_dirs", EXCLUDED_DIRS)

    @property
    def excluded_extensions(self) -> List[str]:
        """Get list of excluded file extensions."""
        return self._config.get("excluded_extensions", EXCLUDED_EXTENSIONS)

    @property
    def default_chunk_size(self) -> int:
        """Get default chunk size for document processing."""
        return self._config.get("default_chunk_size", DEFAULT_CHUNK_SIZE)

    @property
    def command_timeout(self) -> int:
        """Get command execution timeout in seconds."""
        return self._config.get("command_timeout", DEFAULT_COMMAND_TIMEOUT)

    @property
    def max_sessions(self) -> int:
        """Get maximum number of concurrent sessions."""
        return self._config.get("max_sessions", DEFAULT_MAX_SESSIONS)


def setup_logging(config: Config = None) -> logging.Logger:
    """
    Set up logging with rotation and proper formatting.

    Args:
        config: Configuration object. If None, uses defaults.

    Returns:
        Configured logger instance.
    """
    if config is None:
        config = Config()

    log_level = config.get("log_level", "INFO")
    log_dir = Path(config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("codeanalyzer")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "codeanalyzer.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
