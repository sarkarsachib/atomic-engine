#!/usr/bin/env python3
"""
Manus LLM Agent Layer for Atomic Engine
Multi-Provider LLM Abstraction with Intelligent Routing, Streaming, and Token Accounting
"""

__version__ = "1.0.0"
__author__ = "Atomic Engine"

from .client import LLMAgent, create_agent, stream_response
from .config import LLMConfig, ConfigManager, load_config
from .router import Router, RoutingStrategy, CostAwareRouter
from .streaming import StreamHandler, SSEStreamHandler
from .token_accounting import TokenTracker, RateLimiter
from .exceptions import (
    LLMError,
    ProviderError,
    RateLimitError,
    TokenLimitError,
    ConfigurationError,
    StreamingError,
)

# Lazy load providers to avoid import errors when SDKs aren't installed
def __getattr__(name):
    if name in ("BaseProvider", "OpenAIProvider", "AnthropicProvider", 
                "BedrockProvider", "OllamaProvider", "AzureProvider"):
        from . import providers
        if name == "BaseProvider":
            return providers.BaseProvider
        elif name == "OpenAIProvider":
            from .providers.openai import OpenAIProvider
            return OpenAIProvider
        elif name == "AnthropicProvider":
            from .providers.anthropic import AnthropicProvider
            return AnthropicProvider
        elif name == "BedrockProvider":
            from .providers.bedrock import BedrockProvider
            return BedrockProvider
        elif name == "OllamaProvider":
            from .providers.ollama import OllamaProvider
            return OllamaProvider
        elif name == "AzureProvider":
            from .providers.azure import AzureProvider
            return AzureProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main client
    "LLMAgent",
    "create_agent",
    "stream_response",
    # Configuration
    "LLMConfig",
    "ConfigManager",
    "load_config",
    # Providers (lazy loaded)
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "BedrockProvider",
    "OllamaProvider",
    "AzureProvider",
    # Routing
    "Router",
    "RoutingStrategy",
    "CostAwareRouter",
    # Streaming
    "StreamHandler",
    "SSEStreamHandler",
    # Token accounting
    "TokenTracker",
    "RateLimiter",
    # Exceptions
    "LLMError",
    "ProviderError",
    "RateLimitError",
    "TokenLimitError",
    "ConfigurationError",
    "StreamingError",
]
