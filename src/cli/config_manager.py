"""
Configuration Management System
Handles hybrid configuration: env vars, YAML files, and CLI arguments
Priority: CLI args > Env vars > User config > Project config > .env > Defaults
"""

from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from src.cli.utils import get_project_root, print_error, print_warning


class ConfigManager:
    """Manages configuration with priority hierarchy."""

    # Default configuration structure
    DEFAULT_CONFIG = {
        "version": "1.0",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_base": None,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_base": None,
        },
        "rag": {
            "mode": "basic",
            "use_docling": False,
            "chunking_strategy": "recursive",
        },
        "persistence": {
            "use_persistent_memory": True,
            "checkpoint_db_path": "knowledges/checkpoints.db",
            "metrics_db_path": "knowledges/metrics.db",
        },
        "observability": {
            "langsmith_tracing": False,
            "langsmith_project": "doc-bases",
        },
        "retrieval": {
            "mode": "dense",
            "k": 10,
            "final_k": 5,
            "rrf_constant": 60,
        },
        "reranker": {
            "provider": None,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }

    # Mapping from config keys to environment variables
    ENV_VAR_MAPPING = {
        "llm.provider": "LLM_PROVIDER",
        "llm.model": "LLM_MODEL",
        "llm.api_base": "LLM_API_BASE",
        "embedding.provider": "EMB_PROVIDER",
        "embedding.model": "EMB_MODEL",
        "embedding.api_base": "EMB_API_BASE",
        "rag.mode": "RAG_MODE",
        "rag.use_docling": "USE_DOCLING",
        "rag.chunking_strategy": "CHUNKING_STRATEGY",
        "persistence.use_persistent_memory": "USE_PERSISTENT_MEMORY",
        "persistence.checkpoint_db_path": "CHECKPOINT_DB_PATH",
        "persistence.metrics_db_path": "METRICS_DB_PATH",
        "observability.langsmith_tracing": "LANGSMITH_TRACING",
        "observability.langsmith_project": "LANGSMITH_PROJECT",
        "retrieval.mode": "RETRIEVAL_MODE",
        "retrieval.k": "RETRIEVAL_K",
        "retrieval.final_k": "RETRIEVAL_FINAL_K",
        "retrieval.rrf_constant": "RRF_CONSTANT",
        "reranker.provider": "RERANKER_PROVIDER",
        "reranker.model": "RERANKER_MODEL",
    }

    def __init__(self, custom_config_path: str | None = None):
        """Initialize ConfigManager.

        Args:
            custom_config_path: Override default config file path
        """
        self.project_root = get_project_root()
        self.custom_config_path = custom_config_path
        self._config: dict[str, Any] = {}
        self._env_vars: dict[str, str] = {}
        self._sources: dict[str, str] = {}  # Track where each value came from
        self.load()

    def load(self) -> None:
        """Load configuration from all sources with priority."""
        # Start with defaults
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        self._sources = dict.fromkeys(self._flatten_config(self._config), "default")

        # Load .env.example (lowest priority for env-based configs)
        self._load_env_vars_from_file(self.project_root / ".env.example")

        # Load .env file (project-level, user secrets)
        self._load_env_vars_from_file(self.project_root / ".env")

        # Load project-level YAML config if it exists
        project_config_path = self.project_root / ".docbases" / "config.yaml"
        if project_config_path.exists():
            self._load_yaml_config(project_config_path, "project")

        # Load user-level YAML config if it exists
        user_config_path = Path.home() / ".docbases" / "config.yaml"
        if user_config_path.exists():
            self._load_yaml_config(user_config_path, "user")

        # Load custom config if specified
        if self.custom_config_path:
            custom_path = Path(self.custom_config_path)
            if custom_path.exists():
                self._load_yaml_config(custom_path, "custom")
            else:
                print_warning(f"Custom config file not found: {custom_path}")

        # Apply environment variables (highest priority for env-based)
        self._apply_env_overrides()

    def _load_env_vars_from_file(self, env_file_path: Path) -> None:
        """Load environment variables from .env file."""
        if not env_file_path.exists():
            return

        env_vars = dotenv_values(env_file_path)
        # Filter out None values for type safety
        self._env_vars.update({k: v for k, v in env_vars.items() if v is not None})

    def _load_yaml_config(self, yaml_path: Path, source: str) -> None:
        """Load and merge YAML configuration."""
        try:
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f) or {}

            # Merge YAML config into current config
            self._deep_merge(self._config, yaml_config)

            # Track sources
            for key in self._flatten_config(yaml_config):
                full_key = key
                self._sources[full_key] = source

        except Exception as e:
            print_error(f"Failed to load YAML config from {yaml_path}: {e}")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        for config_key, env_var_name in self.ENV_VAR_MAPPING.items():
            if env_var_name in self._env_vars:
                value = self._env_vars[env_var_name]
                # Convert string values to appropriate types
                typed_value = self._convert_value(value, config_key)
                self._set_nested(self._config, config_key, typed_value)
                self._sources[config_key] = "environment"

    def _convert_value(self, value: str, config_key: str) -> Any:
        """Convert string environment variable to appropriate type."""
        if isinstance(value, bool):
            return value

        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        try:
            # Try to convert to int
            return int(value)
        except ValueError:
            pass

        try:
            # Try to convert to float
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dot-notation key (e.g., 'llm.provider').

        Args:
            key: Dot-notation key path
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in configuration.

        Args:
            key: Dot-notation key path
            value: Value to set
        """
        self._set_nested(self._config, key, value)
        self._sources[key] = "user"

    def get_all(self) -> dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._deep_copy(self._config)

    def get_all_with_sources(self) -> dict[str, Any]:
        """Get configuration with source information."""
        result = {}
        for key, value in self._sources.items():
            if key not in result:
                result[key] = {"value": self.get(key), "source": value}

        return result

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required LLM settings
        if not self.get("llm.provider"):
            errors.append("llm.provider is required")
        if not self.get("llm.model"):
            errors.append("llm.model is required")

        # Check required embedding settings
        if not self.get("embedding.provider"):
            errors.append("embedding.provider is required")
        if not self.get("embedding.model"):
            errors.append("embedding.model is required")

        # Check RAG mode is valid
        valid_modes = ["basic", "corrective", "adaptive", "multi_agent"]
        if self.get("rag.mode") not in valid_modes:
            errors.append(f"rag.mode must be one of: {', '.join(valid_modes)}")

        # Check chunking strategy is valid
        valid_strategies = ["recursive", "semantic"]
        if self.get("rag.chunking_strategy") not in valid_strategies:
            errors.append(f"rag.chunking_strategy must be one of: {', '.join(valid_strategies)}")

        # Check retrieval mode is valid
        valid_retrieval_modes = ["dense", "hybrid"]
        if self.get("retrieval.mode") not in valid_retrieval_modes:
            errors.append(f"retrieval.mode must be one of: {', '.join(valid_retrieval_modes)}")

        # Check reranker provider is valid (if set)
        reranker_provider = self.get("reranker.provider")
        valid_reranker_providers = [None, "", "cross-encoder", "cohere", "none", "passthrough"]
        if reranker_provider not in valid_reranker_providers:
            errors.append(f"reranker.provider must be one of: {', '.join(str(p) for p in valid_reranker_providers if p)}")

        return errors

    def export_yaml(self) -> str:
        """Export configuration to YAML string."""
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)

    def import_yaml(self, yaml_content: str) -> None:
        """Import configuration from YAML string."""
        try:
            imported_config = yaml.safe_load(yaml_content)
            if imported_config:
                self._deep_merge(self._config, imported_config)
                for key in self._flatten_config(imported_config):
                    self._sources[key] = "imported"
        except yaml.YAMLError as e:
            print_error(f"Failed to parse YAML: {e}")
            raise

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        self._sources = dict.fromkeys(self._flatten_config(self._config), "default")

    @staticmethod
    def _deep_copy(obj: Any) -> Any:
        """Deep copy a dictionary or list."""
        if isinstance(obj, dict):
            return {key: ConfigManager._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ConfigManager._deep_copy(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _deep_merge(target: dict, source: dict) -> None:
        """Merge source dict into target dict (modifies target in place)."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(target[key], value)
            else:
                target[key] = ConfigManager._deep_copy(value)

    @staticmethod
    def _set_nested(config: dict, key: str, value: Any) -> None:
        """Set value at nested key path (creates intermediate dicts as needed)."""
        parts = key.split(".")
        current = config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    @staticmethod
    def _flatten_config(config: dict, prefix: str = "") -> dict[str, Any]:
        """Flatten nested config dict to dot-notation keys."""
        result = {}

        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(ConfigManager._flatten_config(value, full_key))
            else:
                result[full_key] = value

        return result


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config_manager(custom_path: str | None = None) -> ConfigManager:
    """Get or create global config manager instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(custom_path)

    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set configuration value."""
    get_config_manager().set(key, value)
