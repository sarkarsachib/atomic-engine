#!/usr/bin/env python3
"""
Configuration Management for LLM Agent Layer
Reads from config.toml, environment variables, supports runtime switching
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported LLM Providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"
    AZURE = "azure"


class RoutingStrategy(Enum):
    """Routing strategies for provider selection"""
    PRIORITY = "priority"  # Use primary, fallback to others
    ROUND_ROBIN = "round_robin"  # Cycle through providers
    COST_AWARE = "cost_aware"  # Prefer cheaper models for simple tasks
    CAPABILITY_MATCH = "capability_match"  # Match provider capabilities to task
    LOAD_BALANCED = "load_balanced"  # Distribute based on load


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: str
    max_tokens: int = 4096
    max_input_tokens: int = 8192
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_functions: bool = False
    cost_per_input: float = 0.0
    cost_per_output: float = 0.0
    latency_ms: int = 1000
    context_window: int = 8192

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of this dataclass suitable for serialization.
        
        Returns:
            dict: A mapping of field names to their values representing this instance.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """
        Constructs a ModelConfig from a dictionary.
        
        Parameters:
            data (Dict[str, Any]): Dictionary containing fields for ModelConfig (e.g., name, provider, max_tokens, supports_streaming, cost_per_input, etc.).
        
        Returns:
            ModelConfig: An instance populated with values from `data`.
        """
        return cls(**data)


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    name: str
    enabled: bool = True
    priority: int = 1
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    region: Optional[str] = None
    models: List[ModelConfig] = field(default_factory=list)
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 100000  # Tokens per minute
    timeout_seconds: int = 60
    retry_count: int = 3
    retry_backoff: float = 1.0
    health_check_interval: int = 300  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the provider configuration into a dictionary for persistence or inspection, masking sensitive credentials.
        
        Returns:
            dict: Mapping of provider fields to values. The `api_key` value is masked as `"***"` if present, `models` is a list of each model's serialized dictionary.
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.priority,
            "api_key": "***" if self.api_key else None,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "region": self.region,
            "models": [m.to_dict() for m in self.models],
            "rate_limit_rpm": self.rate_limit_rpm,
            "rate_limit_tpm": self.rate_limit_tpm,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "retry_backoff": self.retry_backoff,
            "health_check_interval": self.health_check_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """
        Construct a ProviderConfig from a mapping of configuration values.
        
        Parameters:
            data (Dict[str, Any]): Dictionary containing provider configuration. Expected keys include
                "name", optional "enabled", "priority", "api_key", "api_base", "api_version",
                "region", "models" (list of model dicts), "rate_limit_rpm", "rate_limit_tpm",
                "timeout_seconds", "retry_count", "retry_backoff", and "health_check_interval".
                If "api_key" is not present, the method will attempt to read it from the environment
                using the provider-specific environment key returned by _env_key(name).
        
        Returns:
            ProviderConfig: A ProviderConfig populated from the provided dictionary. Missing fields
            are filled with sensible defaults (e.g., enabled=True, priority=1, rate_limit_rpm=60,
            rate_limit_tpm=100000, timeout_seconds=60, retry_count=3, retry_backoff=1.0,
            health_check_interval=300). Model entries are converted to ModelConfig via ModelConfig.from_dict.
        """
        models = [ModelConfig.from_dict(m) for m in data.get("models", [])]
        return cls(
            name=data["name"],
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1),
            api_key=data.get("api_key") or os.getenv(cls._env_key(data["name"])),
            api_base=data.get("api_base"),
            api_version=data.get("api_version"),
            region=data.get("region"),
            models=models,
            rate_limit_rpm=data.get("rate_limit_rpm", 60),
            rate_limit_tpm=data.get("rate_limit_tpm", 100000),
            timeout_seconds=data.get("timeout_seconds", 60),
            retry_count=data.get("retry_count", 3),
            retry_backoff=data.get("retry_backoff", 1.0),
            health_check_interval=data.get("health_check_interval", 300),
        )

    @staticmethod
    def _env_key(provider_name: str) -> str:
        """
        Return the environment variable name used for a provider's API key, or `None` if the provider does not use an API key.
        
        Parameters:
            provider_name (str): Provider identifier (case-insensitive).
        
        Returns:
            str | None: Environment variable name (e.g., `OPENAI_API_KEY`) for known providers, `None` for providers that do not require an API key (e.g., "ollama"), or a fallback of `{PROVIDER}_API_KEY` using the uppercased provider name.
        """
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "bedrock": "AWS Bedrock",
            "ollama": None,  # No API key needed
            "azure": "AZURE_OPENAI_API_KEY",
        }
        return env_map.get(provider_name.lower(), f"{provider_name.upper()}_API_KEY")


@dataclass
class LLMConfig:
    """Main configuration for LLM Layer"""
    default_provider: str = "openai"
    fallback_providers: List[str] = field(default_factory=lambda: ["anthropic", "bedrock"])
    routing_strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    global_rate_limit_rpm: int = 1000
    global_token_limit_daily: int = 1000000
    enable_streaming: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 10
    enable_health_checks: bool = True
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the LLMConfig into a plain dictionary suitable for persistence or transmission.
        
        Returns:
            dict: A dictionary representation of the configuration. Keys include
            "default_provider", "fallback_providers", "routing_strategy" (the enum value),
            "providers" (a mapping of provider name to each provider's serialized dict),
            and other top-level settings such as rate limits, feature flags, defaults, and
            system_prompt.
        """
        return {
            "default_provider": self.default_provider,
            "fallback_providers": self.fallback_providers,
            "routing_strategy": self.routing_strategy.value,
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "global_rate_limit_rpm": self.global_rate_limit_rpm,
            "global_token_limit_daily": self.global_token_limit_daily,
            "enable_streaming": self.enable_streaming,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_concurrent_requests": self.max_concurrent_requests,
            "enable_health_checks": self.enable_health_checks,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """
        Create an LLMConfig from a dictionary, applying sensible defaults for any missing fields.
        
        Parameters:
            data (Dict[str, Any]): Mapping containing LLM configuration values (top-level keys such as
                "default_provider", "fallback_providers", "routing_strategy", and "providers"). The
                "providers" value, if present, should be a mapping of provider names to provider
                configuration dictionaries and will be converted into ProviderConfig instances.
        
        Returns:
            LLMConfig: An LLMConfig instance populated from `data`; fields not provided in `data`
            are filled with predefined default values.
        """
        providers = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = ProviderConfig.from_dict(pdata)

        return cls(
            default_provider=data.get("default_provider", "openai"),
            fallback_providers=data.get("fallback_providers", ["anthropic", "bedrock"]),
            routing_strategy=RoutingStrategy(data.get("routing_strategy", "priority")),
            providers=providers,
            global_rate_limit_rpm=data.get("global_rate_limit_rpm", 1000),
            global_token_limit_daily=data.get("global_token_limit_daily", 1000000),
            enable_streaming=data.get("enable_streaming", True),
            enable_logging=data.get("enable_logging", True),
            log_level=data.get("log_level", "INFO"),
            enable_metrics=data.get("enable_metrics", True),
            cache_enabled=data.get("cache_enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 3600),
            max_concurrent_requests=data.get("max_concurrent_requests", 10),
            enable_health_checks=data.get("enable_health_checks", True),
            default_temperature=data.get("default_temperature", 0.7),
            default_max_tokens=data.get("default_max_tokens", 4096),
            system_prompt=data.get("system_prompt"),
        )

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name"""
        return self.providers.get(name.lower())

    def get_model(self, provider: str, model_name: str) -> Optional[ModelConfig]:
        """
        Retrieve a model configuration for a given provider and model name.
        
        Parameters:
            provider (str): Provider name (case-insensitive) to search within.
            model_name (str): Exact model name to locate.
        
        Returns:
            ModelConfig if a matching model is found, None otherwise.
        """
        pconfig = self.get_provider(provider)
        if pconfig:
            for model in pconfig.models:
                if model.name == model_name:
                    return model
        return None

    def get_healthy_providers(self) -> List[str]:
        """
        Return the names of providers that are enabled, ordered by priority.
        
        Returns:
            List[str]: Enabled provider names sorted by priority with lower numeric priority values first.
        """
        return [
            name for name, config in sorted(
                self.providers.items(),
                key=lambda x: (not x[1].enabled, x[1].priority)
            )
            if config.enabled
        ]


class ConfigManager:
    """Manages LLM configuration from multiple sources"""

    def __init__(self, config_paths: Optional[List[str]] = None):
        """
        Initialize the ConfigManager with candidate configuration file paths and prepare internal state.
        
        Parameters:
            config_paths: Optional list of file paths to search, in order, for a configuration file. If omitted, a sensible default list of common config locations is used (including user home).
        """
        self.config_paths = config_paths or [
            "config/config.toml",
            "config/llm.toml",
            "llm_config.toml",
            os.path.expanduser("~/.atomic_engine/llm.toml"),
        ]
        self._config: Optional[LLMConfig] = None
        self._watch_handlers: List = []

    def load(self) -> LLMConfig:
        """
        Load configuration from the first readable file in self.config_paths, falling back to defaults.
        
        Tries each configured path in order and uses the first file that can be successfully parsed. If a file is found and loaded, stores it on self._config and returns it. If no valid file is found, creates, stores, and returns a default LLMConfig.
        Returns:
            LLMConfig: The loaded configuration or a newly created default configuration.
        """
        for path_str in self.config_paths:
            path = Path(path_str)
            if path.exists():
                try:
                    config = self._load_from_file(path)
                    logger.info(f"Loaded configuration from {path}")
                    self._config = config
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")

        # Return default configuration if no config file found
        logger.info("No config file found, using defaults")
        self._config = self._create_default_config()
        return self._config

    def _load_from_file(self, path: Path) -> LLMConfig:
        """
        Load a configuration file and return an LLMConfig constructed from its contents.
        
        Supports TOML ('.toml'), JSON ('.json'), and YAML ('.yaml' or '.yml') formats.
        
        Returns:
            LLMConfig: Configuration parsed from the provided file.
        
        Raises:
            ValueError: If the file extension is not one of the supported formats.
        """
        suffix = path.suffix.lower()

        if suffix == '.toml':
            return self._load_toml(path)
        elif suffix == '.json':
            return self._load_json(path)
        elif suffix == '.yaml' or suffix == '.yml':
            return self._load_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    def _load_toml(self, path: Path) -> LLMConfig:
        """
        Load configuration from a TOML file and return an LLMConfig.
        
        Returns:
            LLMConfig: Parsed configuration built from the TOML file contents.
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return self._parse_config_data(data)

    def _load_json(self, path: Path) -> LLMConfig:
        """
        Load a JSON configuration file and construct an LLMConfig from its contents.
        
        Parameters:
            path (Path): Filesystem path to the JSON configuration file.
        
        Returns:
            LLMConfig: Configuration built from the parsed JSON data.
        """
        with open(path) as f:
            data = json.load(f)
        return self._parse_config_data(data)

    def _load_yaml(self, path: Path) -> LLMConfig:
        """
        Load and parse a YAML configuration file into an LLMConfig.
        
        Returns:
            LLMConfig: An LLMConfig built from the YAML file contents.
        
        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files")

        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse_config_data(data)

    def _parse_config_data(self, data: Dict[str, Any]) -> LLMConfig:
        """
        Builds an LLMConfig from raw configuration data, applying environment-variable overrides and provider defaults.
        
        Parameters:
            data (Dict[str, Any]): Parsed configuration mapping (expected keys include "providers" and top-level settings like "default_provider", "routing_strategy", and rate/feature flags).
        
        Returns:
            LLMConfig: Configuration object with providers normalized (lowercased keys), provider models populated with defaults when absent, and top-level fields set to provided values or sensible defaults.
        """
        # Apply environment variable overrides
        data = self._apply_env_overrides(data)

        # Build provider configurations with default models
        providers = {}
        for name, pdata in data.get("providers", {}).items():
            pconfig = ProviderConfig.from_dict(pdata)
            providers[name.lower()] = pconfig

        # Add default models if none specified
        for name in providers:
            if not providers[name].models:
                providers[name].models = self._get_default_models(name)

        return LLMConfig(
            default_provider=data.get("default_provider", "openai"),
            fallback_providers=data.get("fallback_providers", ["anthropic", "bedrock"]),
            routing_strategy=RoutingStrategy(data.get("routing_strategy", "priority")),
            providers=providers,
            global_rate_limit_rpm=data.get("global_rate_limit_rpm", 1000),
            global_token_limit_daily=data.get("global_token_limit_daily", 1000000),
            enable_streaming=data.get("enable_streaming", True),
            enable_logging=data.get("enable_logging", True),
            log_level=data.get("log_level", "INFO"),
            enable_metrics=data.get("enable_metrics", True),
            cache_enabled=data.get("cache_enabled", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 3600),
            max_concurrent_requests=data.get("max_concurrent_requests", 10),
            enable_health_checks=data.get("enable_health_checks", True),
            default_temperature=data.get("default_temperature", 0.7),
            default_max_tokens=data.get("default_max_tokens", 4096),
            system_prompt=data.get("system_prompt"),
        )

    def _apply_env_overrides(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject environment variable values into the parsed configuration dictionary for known provider-specific keys.
        
        Parameters:
            data (Dict[str, Any]): Parsed configuration data (mutable) to apply environment overrides to.
        
        Returns:
            Dict[str, Any]: The same configuration dictionary with environment-provided values set for matching provider keys (e.g., API keys, endpoints).
        """
        # Provider API keys
        env_mappings = {
            "OPENAI_API_KEY": ("providers", "openai", "api_key"),
            "ANTHROPIC_API_KEY": ("providers", "anthropic", "api_key"),
            "AWS_ACCESS_KEY_ID": ("providers", "bedrock", "aws_access_key_id"),
            "AWS_SECRET_ACCESS_KEY": ("providers", "bedrock", "aws_secret_access_key"),
            "AZURE_OPENAI_API_KEY": ("providers", "azure", "api_key"),
            "AZURE_OPENAI_ENDPOINT": ("providers", "azure", "api_base"),
        }

        for env_key, path in env_mappings.items():
            value = os.getenv(env_key)
            if value:
                current = data
                for key in path[:-1]:
                    current = current.setdefault(key, {})
                current[path[-1]] = value

        return data

    def _get_default_models(self, provider: str) -> List[ModelConfig]:
        """
        Return default model configurations for the specified provider.
        
        Parameters:
            provider (str): Provider name (case-insensitive). Supported values include "openai", "anthropic", "bedrock", "ollama", and "azure".
        
        Returns:
            List[ModelConfig]: A list of default ModelConfig instances for the provider; an empty list if the provider is not recognized.
        """
        defaults = {
            "openai": [
                ModelConfig(name="gpt-4o", provider="openai", max_tokens=16384,
                           max_input_tokens=128000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=5.0, cost_per_output=15.0,
                           latency_ms=500, context_window=128000),
                ModelConfig(name="gpt-4o-mini", provider="openai", max_tokens=16384,
                           max_input_tokens=128000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=0.15, cost_per_output=0.6,
                           latency_ms=300, context_window=128000),
                ModelConfig(name="o1", provider="openai", max_tokens=65536,
                           max_input_tokens=200000, supports_streaming=False, supports_vision=True,
                           supports_functions=False, cost_per_input=15.0, cost_per_output=60.0,
                           latency_ms=5000, context_window=200000),
                ModelConfig(name="gpt-4-turbo", provider="openai", max_tokens=4096,
                           max_input_tokens=128000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=10.0, cost_per_output=30.0,
                           latency_ms=800, context_window=128000),
            ],
            "anthropic": [
                ModelConfig(name="claude-sonnet-4-20250514", provider="anthropic", max_tokens=8192,
                           max_input_tokens=200000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=3.0, cost_per_output=15.0,
                           latency_ms=600, context_window=200000),
                ModelConfig(name="claude-opus-4-20250514", provider="anthropic", max_tokens=8192,
                           max_input_tokens=200000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=15.0, cost_per_output=75.0,
                           latency_ms=800, context_window=200000),
                ModelConfig(name="claude-haiku-3-20250514", provider="anthropic", max_tokens=8192,
                           max_input_tokens=200000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=0.25, cost_per_output=1.25,
                           latency_ms=400, context_window=200000),
            ],
            "bedrock": [
                ModelConfig(name="anthropic.claude-3-5-sonnet-20241022", provider="bedrock",
                           max_tokens=8192, max_input_tokens=200000, supports_streaming=True,
                           supports_vision=True, supports_functions=True, cost_per_input=3.0,
                           cost_per_output=15.0, latency_ms=700, context_window=200000),
                ModelConfig(name="anthropic.claude-3-haiku-20240307", provider="bedrock",
                           max_tokens=8192, max_input_tokens=200000, supports_streaming=True,
                           supports_vision=True, supports_functions=True, cost_per_input=0.25,
                           cost_per_output=1.25, latency_ms=500, context_window=200000),
                ModelConfig(name="amazon.titan-text-premier-v1:0", provider="bedrock",
                           max_tokens=4096, max_input_tokens=300000, supports_streaming=True,
                           supports_vision=False, supports_functions=False, cost_per_input=0.0005,
                           cost_per_output=0.0015, latency_ms=300, context_window=300000),
            ],
            "ollama": [
                ModelConfig(name="llama3.2", provider="ollama", max_tokens=8192,
                           max_input_tokens=131072, supports_streaming=True, supports_vision=False,
                           supports_functions=False, cost_per_input=0.0, cost_per_output=0.0,
                           latency_ms=100, context_window=131072),
                ModelConfig(name="qwen2.5", provider="ollama", max_tokens=8192,
                           max_input_tokens=131072, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=0.0, cost_per_output=0.0,
                           latency_ms=80, context_window=131072),
                ModelConfig(name="mistral", provider="ollama", max_tokens=8192,
                           max_input_tokens=32768, supports_streaming=True, supports_vision=False,
                           supports_functions=False, cost_per_input=0.0, cost_per_output=0.0,
                           latency_ms=100, context_window=32768),
                ModelConfig(name="deepseek-r1", provider="ollama", max_tokens=8192,
                           max_input_tokens=131072, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=0.0, cost_per_output=0.0,
                           latency_ms=150, context_window=131072),
            ],
            "azure": [
                ModelConfig(name="gpt-4o", provider="azure", max_tokens=16384,
                           max_input_tokens=128000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=5.0, cost_per_output=15.0,
                           latency_ms=600, context_window=128000),
                ModelConfig(name="gpt-4-turbo", provider="azure", max_tokens=4096,
                           max_input_tokens=128000, supports_streaming=True, supports_vision=True,
                           supports_functions=True, cost_per_input=10.0, cost_per_output=30.0,
                           latency_ms=800, context_window=128000),
            ],
        }
        return defaults.get(provider.lower(), [])

    def _create_default_config(self) -> LLMConfig:
        """
        Builds a default LLMConfig with a predefined set of providers and their default models.
        
        Returns:
            LLMConfig: Configuration containing enabled provider entries for "openai", "anthropic", "bedrock", "ollama", and "azure"; each provider is assigned a priority and populated with its default ModelConfig list.
        """
        config = LLMConfig()
        default_providers = ["openai", "anthropic", "bedrock", "ollama", "azure"]

        for name in default_providers:
            config.providers[name] = ProviderConfig(
                name=name,
                enabled=True,
                priority=default_providers.index(name) + 1,
                models=self._get_default_models(name),
            )

        return config

    def get_config(self) -> LLMConfig:
        """
        Get the current LLM configuration, loading it if not already loaded.
        
        Returns:
            LLMConfig: The active configuration object.
        """
        if self._config is None:
            self.load()
        return self._config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Apply runtime updates to the loaded LLM configuration.
        
        Ensures a configuration is loaded, then sets top-level attributes on the in-memory LLMConfig for any keys present in the provided mapping. Unknown keys are ignored; updates are applied in-place.
        Parameters:
            updates (Dict[str, Any]): Mapping of LLMConfig attribute names to new values to apply.
        """
        if self._config is None:
            self.load()

        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        logger.info("Configuration updated at runtime")

    def switch_provider(self, provider_name: str) -> bool:
        """
        Set the LLM configuration's default provider to the given provider name.
        
        If the configuration is not yet loaded, it will be loaded first. The lookup is case-insensitive; when successful, the manager's `default_provider` is updated to the provider name in lowercase and `True` is returned. If no matching provider is found, the configuration is unchanged and `False` is returned.
        
        Parameters:
            provider_name (str): The provider name to set as default (case-insensitive).
        
        Returns:
            bool: `True` if the default provider was changed to `provider_name`, `False` otherwise.
        """
        if self._config is None:
            self.load()

        if provider_name.lower() in self._config.providers:
            self._config.default_provider = provider_name.lower()
            logger.info(f"Switched default provider to {provider_name}")
            return True
        else:
            logger.warning(f"Provider {provider_name} not found")
            return False

    def save(self, path: str) -> None:
        """
        Persist the current LLM configuration to disk using the file format inferred from the given path.
        
        If the manager has not yet loaded a configuration, the configuration is loaded before saving.
        
        Parameters:
            path (str): Filesystem path where the configuration will be written. Supported file extensions are `.toml` and `.json`.
        
        Raises:
            ValueError: If the file extension is not `.toml` or `.json`.
        """
        if self._config is None:
            self.load()

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == '.toml':
            self._save_toml(path)
        elif suffix == '.json':
            self._save_json(path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    def _save_toml(self, path: Path) -> None:
        """
        Write the manager's current configuration to the given file path in TOML format, replacing any existing file.
        
        Parameters:
            path (Path): Destination file path where the TOML configuration will be written.
        """
        import tomllib

        with open(path, 'wb') as f:
            tomllib.dump(self._config.to_dict(), f)

    def _save_json(self, path: Path) -> None:
        """
        Write the current LLM configuration to the given path in JSON format, pretty-printed with 2-space indentation.
        
        Parameters:
            path (Path): Filesystem path to write the JSON configuration to; an existing file will be overwritten.
        """
        with open(path, 'w') as f:
            json.dump(self._config.to_dict(), f, indent=2)


def load_config(config_paths: Optional[List[str]] = None) -> LLMConfig:
    """Convenience function to load configuration"""
    manager = ConfigManager(config_paths)
    return manager.load()


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Return the singleton global ConfigManager used for managing LLM configuration.
    
    Returns:
        ConfigManager: The cached global ConfigManager instance; one is created and stored on first call.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> LLMConfig:
    """
    Retrieve the global LLM configuration used by the application.
    
    Returns:
        The active LLMConfig instance.
    """
    return get_config_manager().get_config()