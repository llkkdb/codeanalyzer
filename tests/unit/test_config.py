"""Unit tests for configuration management."""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from codeanalyzer.config import Config, setup_logging, ALLOWED_COMMANDS
from codeanalyzer.exceptions import ConfigurationError


@pytest.fixture
def temp_config_path(tmp_path):
    """Create a temporary config path."""
    return tmp_path / "test_config.json"


@pytest.fixture
def config(temp_config_path):
    """Create a Config instance with temp path."""
    return Config(config_path=temp_config_path)


class TestConfig:
    """Tests for Config class."""

    def test_init_creates_default_config(self, config, temp_config_path):
        """Test that initialization creates default config."""
        assert isinstance(config._config, dict)
        assert "allowed_commands" in config._config
        assert "max_file_size" in config._config

    def test_init_loads_existing_config(self, temp_config_path):
        """Test loading existing config file."""
        # Create config file
        test_config = {"max_file_size": 999999, "custom_key": "custom_value"}
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_config_path.write_text(json.dumps(test_config))

        # Load config
        config = Config(config_path=temp_config_path)

        assert config.get("max_file_size") == 999999
        assert config.get("custom_key") == "custom_value"

    def test_init_handles_invalid_json(self, temp_config_path):
        """Test handling of invalid JSON in config file."""
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_config_path.write_text("invalid json{")

        with pytest.raises(ConfigurationError):
            Config(config_path=temp_config_path)

    def test_save_config(self, config, temp_config_path):
        """Test saving configuration."""
        config.set("test_key", "test_value")
        config.save()

        assert temp_config_path.exists()
        saved_data = json.loads(temp_config_path.read_text())
        assert saved_data["test_key"] == "test_value"

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        config_path = tmp_path / "nested" / "config" / "test.json"
        config = Config(config_path=config_path)
        config.save()

        assert config_path.exists()
        assert config_path.parent.exists()

    def test_save_handles_io_error(self, config):
        """Test save handles IO errors."""
        with patch.object(Path, 'write_text', side_effect=IOError("Write failed")):
            with pytest.raises(ConfigurationError) as exc_info:
                config.save()
            assert "Failed to save config" in str(exc_info.value)

    def test_get_existing_key(self, config):
        """Test getting existing configuration value."""
        config.set("test_key", "test_value")
        assert config.get("test_key") == "test_value"

    def test_get_nonexistent_key_with_default(self, config):
        """Test getting non-existent key returns default."""
        assert config.get("nonexistent", "default") == "default"

    def test_get_nonexistent_key_without_default(self, config):
        """Test getting non-existent key without default returns None."""
        assert config.get("nonexistent") is None

    def test_set_value(self, config):
        """Test setting configuration value."""
        config.set("new_key", 12345)
        assert config._config["new_key"] == 12345

    def test_set_overwrites_existing(self, config):
        """Test that set overwrites existing values."""
        config.set("key", "value1")
        config.set("key", "value2")
        assert config.get("key") == "value2"

    def test_allowed_commands_property(self, config):
        """Test allowed_commands property."""
        commands = config.allowed_commands
        assert isinstance(commands, dict)
        assert "grep" in commands
        assert "find" in commands
        assert "rg" in commands

    def test_allowed_commands_custom(self, config):
        """Test custom allowed_commands in config."""
        custom_commands = {"custom": {"allowed_flags": ["-a"]}}
        config.set("allowed_commands", custom_commands)
        assert config.allowed_commands == custom_commands

    def test_max_file_size_property(self, config):
        """Test max_file_size property."""
        assert isinstance(config.max_file_size, int)
        assert config.max_file_size > 0

    def test_max_file_size_custom(self, config):
        """Test custom max_file_size."""
        config.set("max_file_size", 999999)
        assert config.max_file_size == 999999

    def test_excluded_dirs_property(self, config):
        """Test excluded_dirs property."""
        dirs = config.excluded_dirs
        assert isinstance(dirs, list)
        assert ".git" in dirs
        assert "node_modules" in dirs

    def test_excluded_extensions_property(self, config):
        """Test excluded_extensions property."""
        exts = config.excluded_extensions
        assert isinstance(exts, list)
        assert ".pyc" in exts
        assert ".pyo" in exts

    def test_log_level_property(self, config):
        """Test log_level property."""
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_log_level_custom(self, config):
        """Test custom log_level."""
        config.set("log_level", "DEBUG")
        assert config.log_level == "DEBUG"

    def test_config_persistence(self, temp_config_path):
        """Test that config persists across instances."""
        config1 = Config(config_path=temp_config_path)
        config1.set("persistent_key", "persistent_value")
        config1.save()

        config2 = Config(config_path=temp_config_path)
        assert config2.get("persistent_key") == "persistent_value"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default config."""
        logger = setup_logging()

        assert logger is not None
        assert logger.name == "codeanalyzer"
        assert len(logger.handlers) >= 1

    def test_setup_logging_custom_config(self, temp_config_path):
        """Test setup_logging with custom config."""
        config = Config(config_path=temp_config_path)
        config.set("log_level", "DEBUG")

        logger = setup_logging(config)

        assert logger.level == 10  # DEBUG level

    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that setup_logging creates log directory."""
        import os
        os.environ["CODEANALYZER_LOG_DIR"] = str(tmp_path / "logs")

        logger = setup_logging()

        log_dir = Path(os.environ["CODEANALYZER_LOG_DIR"])
        assert log_dir.exists()

        # Cleanup
        del os.environ["CODEANALYZER_LOG_DIR"]

    def test_setup_logging_file_handler(self):
        """Test that rotating file handler is configured."""
        logger = setup_logging()

        # Find RotatingFileHandler
        from logging.handlers import RotatingFileHandler
        handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]

        assert len(handlers) > 0
        handler = handlers[0]
        assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
        assert handler.backupCount == 5

    def test_setup_logging_console_handler(self):
        """Test that console handler is configured."""
        import logging
        logger = setup_logging()

        # Find StreamHandler
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(handlers) > 0

    def test_setup_logging_with_env_log_level(self):
        """Test setup_logging respects environment variable."""
        import os
        os.environ["CODEANALYZER_LOG_LEVEL"] = "WARNING"

        logger = setup_logging()

        assert logger.level == 30  # WARNING level

        # Cleanup
        del os.environ["CODEANALYZER_LOG_LEVEL"]


class TestAllowedCommands:
    """Tests for ALLOWED_COMMANDS constant."""

    def test_allowed_commands_structure(self):
        """Test ALLOWED_COMMANDS has correct structure."""
        assert isinstance(ALLOWED_COMMANDS, dict)
        assert "grep" in ALLOWED_COMMANDS
        assert "find" in ALLOWED_COMMANDS
        assert "rg" in ALLOWED_COMMANDS

    def test_grep_config(self):
        """Test grep command configuration."""
        grep_config = ALLOWED_COMMANDS["grep"]
        assert "allowed_flags" in grep_config
        assert isinstance(grep_config["allowed_flags"], list)
        assert "-r" in grep_config["allowed_flags"]
        assert "-H" in grep_config["allowed_flags"]

    def test_find_config(self):
        """Test find command configuration."""
        find_config = ALLOWED_COMMANDS["find"]
        assert "allowed_flags" in find_config
        assert "-name" in find_config["allowed_flags"]
        assert "-type" in find_config["allowed_flags"]

    def test_rg_config(self):
        """Test rg (ripgrep) command configuration."""
        rg_config = ALLOWED_COMMANDS["rg"]
        assert "allowed_flags" in rg_config
        assert "--files" in rg_config["allowed_flags"]


class TestConfigEdgeCases:
    """Edge case tests for Config."""

    def test_config_with_none_values(self, config):
        """Test config handles None values."""
        config.set("none_key", None)
        assert config.get("none_key") is None

    def test_config_with_nested_dict(self, config):
        """Test config handles nested dictionaries."""
        nested = {"level1": {"level2": {"level3": "value"}}}
        config.set("nested", nested)
        assert config.get("nested")["level1"]["level2"]["level3"] == "value"

    def test_config_with_list_values(self, config):
        """Test config handles list values."""
        test_list = [1, 2, 3, "four", 5.0]
        config.set("list_key", test_list)
        assert config.get("list_key") == test_list

    def test_config_type_annotations(self, config):
        """Test that get/set methods have proper type annotations."""
        from typing import Any
        import inspect

        get_sig = inspect.signature(config.get)
        assert get_sig.return_annotation == Any

        set_sig = inspect.signature(config.set)
        assert set_sig.return_annotation == type(None) or set_sig.return_annotation == None
