#!/usr/bin/env python3
"""
Base Provider Abstract Class
Defines the interface for all LLM providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, Optional, List
from datetime import datetime
import logging

from ..config import ModelConfig
from ..exceptions import (
    LLMError,
    ProviderError,
    RateLimitError,
    TokenLimitError,
    AuthenticationError,
    ConnectionError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Standardized LLM request format"""
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    system_prompt: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[str] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "system_prompt": self.system_prompt,
            "functions": self.functions,
            "function_call": self.function_call,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "user": self.user,
            "metadata": self.metadata,
        }


@dataclass
class TokenUsage:
    """Token usage information"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: Optional[int] = None

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens or self.input_tokens,
            "completion_tokens": self.completion_tokens or self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_tokens=self.prompt_tokens + (other.prompt_tokens or 0),
            completion_tokens=self.completion_tokens + (other.completion_tokens or 0),
            reasoning_tokens=self.reasoning_tokens + (other.reasoning_tokens or 0) if self.reasoning_tokens and other.reasoning_tokens else None,
        )


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    provider: str
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "provider": self.provider,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "raw_response": self.raw_response,
            "function_call": self.function_call,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """A chunk of a streaming response"""
    content: str
    delta: str
    model: str
    provider: str
    chunk_index: int
    is_final: bool
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_chunk: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "delta": self.delta,
            "model": self.model,
            "provider": self.provider,
            "chunk_index": self.chunk_index,
            "is_final": self.is_final,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "timestamp": self.timestamp.isoformat(),
            "raw_chunk": self.raw_chunk,
        }


class ProviderHealthStatus:
    """Provider health status tracking"""

    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
        self.last_check: Optional[datetime] = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_error: Optional[str] = None
        self.latency_ms: Optional[float] = None
        self.request_count = 0
        self.error_count = 0

    def record_success(self, latency_ms: float) -> None:
        self.is_healthy = True
        self.last_check = datetime.now()
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.latency_ms = latency_ms
        self.request_count += 1

    def record_failure(self, error: str) -> None:
        self.is_healthy = False
        self.last_check = datetime.now()
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_error = error
        self.error_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "request_count": self.request_count,
            "error_count": self.error_count,
        }


class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""

    provider_name: str = "base"
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_functions: bool = False

    def __init__(
        self,
        config: Optional["ProviderConfig"] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.config = config
        self.api_key = api_key or (config.api_key if config else None)
        self.health_status = ProviderHealthStatus(self.provider_name)
        self._client = None

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (e.g., create client connections)"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up provider resources"""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a non-streaming response"""
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response"""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens for a given text"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model for this provider"""
        pass

    def _get_model_config(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """Get model configuration"""
        if not self.config:
            return None
        model_name = model_name or self.default_model
        for model in self.config.models:
            if model.name == model_name:
                return model
        return None

    def _apply_default_params(
        self,
        request: LLMRequest,
        model_config: Optional[ModelConfig] = None,
    ) -> LLMRequest:
        """Apply default parameters from model config"""
        if model_config:
            if request.max_tokens is None:
                request.max_tokens = model_config.max_tokens

        return request

    def _estimate_cost(self, usage: TokenUsage, model_config: Optional[ModelConfig] = None) -> float:
        """Estimate cost for token usage"""
        if not model_config:
            return 0.0

        return (
            usage.input_tokens * model_config.cost_per_input / 1_000_000 +
            usage.output_tokens * model_config.cost_per_output / 1_000_000
        )

    async def _with_retry(
        self,
        operation,
        max_retries: Optional[int] = None,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        """Execute operation with retry logic"""
        import random
        import asyncio

        max_retries = max_retries or (self.config.retry_count if self.config else 3)
        base_delay = base_delay or (self.config.retry_backoff if self.config else 1.0)

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await operation(attempt=attempt)
            except (RateLimitError, TimeoutError, ConnectionError) as e:
                last_error = e

                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    delay = min(delay, max_delay)

                    if jitter:
                        delay *= (0.5 + random.random())

                    logger.warning(
                        f"Provider {self.provider_name} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Provider {self.provider_name} failed after {max_retries + 1} attempts: {e}"
                    )
                    raise
            except AuthenticationError as e:
                # Don't retry auth errors
                logger.error(f"Authentication failed for {self.provider_name}: {e}")
                raise

        raise last_error

    def _create_error(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        is_retryable: bool = True,
    ) -> ProviderError:
        """Create a provider error with context"""
        return ProviderError(
            message=message,
            provider=self.provider_name,
            status_code=status_code,
            response_body=response_body,
            recovery_hint=self._get_recovery_hint(status_code),
            is_retryable=is_retryable,
        )

    def _get_recovery_hint(self, status_code: Optional[int]) -> Optional[str]:
        """Get recovery hint based on status code"""
        hints = {
            401: "Check your API key and ensure it has the correct permissions",
            403: "Verify your account has access to this model/feature",
            404: "Check if the model name is correct and available",
            429: "Wait before retrying or reduce request frequency",
            500: "Retry the request; this may be a temporary server issue",
            503: "The service is temporarily unavailable; try again later",
        }
        return hints.get(status_code) if status_code else None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name} healthy={self.health_status.is_healthy}>"

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self._client:
                import asyncio
                asyncio.get_event_loop().run_until_complete(self.shutdown())
        except Exception:
            pass
