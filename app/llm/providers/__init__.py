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
        """
        Serialize the LLMRequest into a dictionary of its public fields.
        
        Returns:
            request_dict (Dict[str, Any]): Dictionary containing the request fields: "messages", "model", "temperature", "max_tokens", "stream", "system_prompt", "functions", "function_call", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user", and "metadata".
        """
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
        """
        Ensure total_tokens is populated after initialization.
        
        If `total_tokens` is zero, set it to the sum of `input_tokens` and `output_tokens` to provide a sensible default.
        """
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the token usage metrics.
        
        The mapping includes `input_tokens`, `output_tokens`, `total_tokens`, `prompt_tokens` (falls back to `input_tokens` if unset), `completion_tokens` (falls back to `output_tokens` if unset), and `reasoning_tokens`.
        
        Returns:
            dict: A dictionary with keys `"input_tokens"`, `"output_tokens"`, `"total_tokens"`, `"prompt_tokens"`, `"completion_tokens"`, and `"reasoning_tokens"` mapping to their respective values.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens or self.input_tokens,
            "completion_tokens": self.completion_tokens or self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """
        Combine two TokenUsage instances by summing corresponding token counts.
        
        Returns:
            TokenUsage: A new TokenUsage whose numeric fields are the sum of the operands' fields. `reasoning_tokens` is set to the sum only if both operands have a value; otherwise it is `None`.
        """
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
        """
        Serialize the LLMResponse into a dictionary suitable for JSON serialization.
        
        Returns:
            dict: A mapping of response fields to serializable values:
                - `content`: response text or payload
                - `model`: model identifier
                - `usage`: token usage as a dictionary
                - `finish_reason`: reason the generation finished
                - `provider`: provider name
                - `request_id`: optional provider request identifier
                - `timestamp`: ISO 8601 formatted timestamp string
                - `raw_response`: original provider response payload
                - `function_call`: optional function call metadata
                - `metadata`: additional user-defined metadata
        """
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
        """
        Return a dictionary representation of the StreamChunk.
        
        @returns A dict with the following keys:
            - "content" (str): Full accumulated content for the chunk.
            - "delta" (str): Incremental content delivered in this chunk.
            - "model" (str): Model name that produced the chunk.
            - "provider" (str): Provider identifier.
            - "chunk_index" (int): Sequential index of this chunk in the stream.
            - "is_final" (bool): Whether this chunk is the final chunk for the stream.
            - "finish_reason" (Optional[str]): Reason the stream finished, if any.
            - "usage" (dict | None): Token usage metrics as a dict, or `None` if unavailable.
            - "timestamp" (str): ISO 8601 formatted timestamp for the chunk.
            - "raw_chunk" (Any): Original raw chunk payload from the provider.
        """
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
        """
        Initialize ProviderHealthStatus for a provider.
        
        Parameters:
            name (str): The provider's identifier used in health records and logs.
        """
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
        """
        Mark the provider as healthy and update health metrics after a successful request.
        
        Parameters:
            latency_ms (float): Observed request latency in milliseconds; used to record the latest latency and increment success metrics.
        """
        self.is_healthy = True
        self.last_check = datetime.now()
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.latency_ms = latency_ms
        self.request_count += 1

    def record_failure(self, error: str) -> None:
        """
        Mark the provider as having experienced a failure and update its health metrics.
        
        Parameters:
            error (str): Human-readable error message or exception summary associated with the failure.
        
        Description:
            Sets `is_healthy` to False, updates `last_check` to the current time, increments `consecutive_failures`
            and `error_count`, resets `consecutive_successes` to 0, and records `last_error`.
        """
        self.is_healthy = False
        self.last_check = datetime.now()
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_error = error
        self.error_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the provider's health metrics into a dictionary.
        
        Returns:
            dict: Mapping of health metric names to their values:
                - name (str): Provider name.
                - is_healthy (bool): Whether the provider is considered healthy.
                - last_check (str | None): ISO 8601 timestamp of the last health check, or None if not set.
                - consecutive_failures (int): Number of consecutive failed checks.
                - consecutive_successes (int): Number of consecutive successful checks.
                - last_error (str | None): Last error message, if any.
                - latency_ms (float | None): Last observed latency in milliseconds, if recorded.
                - request_count (int): Total number of requests observed.
                - error_count (int): Total number of errors observed.
        """
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
        """
        Initialize the provider with optional configuration and API key, and set up provider health tracking.
        
        Parameters:
            config (Optional[ProviderConfig]): Optional provider configuration; used to derive defaults such as api_key when not provided explicitly.
            api_key (Optional[str]): Optional API key to override the one in `config`.
            **kwargs: Additional attributes to apply to the instance. For each key present in `kwargs`, if the instance already has an attribute with that name, its value will be set to the provided value.
        
        Side effects:
            - Sets `self.config`, `self.api_key`, `self.health_status` (a ProviderHealthStatus initialized with the provider's name), and `self._client` (initialized to None).
        """
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
        """
        Prepare and allocate any resources required for the provider to operate.
        
        Implementations should establish clients, connections, or background tasks needed before handling requests.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Release and close any resources held by the provider, making it safe to dispose.
        
        Implementations should ensure network clients, open connections, and background tasks are cleanly closed and any pending work is finalized.
        """
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a non-streaming LLM response for the given request.
        
        Parameters:
            request (LLMRequest): The request payload describing messages, model choice, and optional parameters.
        
        Returns:
            LLMResponse: The provider's completed response, including content, token usage, model info, and any function call or raw response metadata.
        """
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream chunks of a model response for the given LLM request.
        
        Yields incremental StreamChunk objects representing partial outputs from the provider. The final chunk will indicate completion via its `is_final` flag and may include `finish_reason` and aggregated `usage`.
        
        Parameters:
            request (LLMRequest): The request payload describing messages, model selection, streaming flag, and any model-specific parameters.
        
        Returns:
            AsyncIterator[StreamChunk]: An iterator that produces StreamChunk instances representing the streaming response.
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Return the number of tokenizer tokens in the provided text.
        
        Parameters:
            text (str): The input string whose tokens should be counted.
        
        Returns:
            int: The count of tokens produced from `text` according to the provider's tokenization.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Return the names of models that this provider can currently serve.
        
        Returns:
            A list of model name strings available from the provider.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform a health check for the provider.
        
        Returns:
            True if the provider is healthy, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """
        Get the provider's default model name.
        
        Returns:
            The default model name for this provider.
        """
        pass

    def _get_model_config(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """
        Retrieve the ModelConfig for a given model name or for the provider's default model.
        
        Parameters:
            model_name (Optional[str]): Name of the model to look up. If omitted, the provider's default model name is used.
        
        Returns:
            Optional[ModelConfig]: The matching ModelConfig if found, otherwise `None`.
        """
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
        """
        Apply model-configured default parameters to an LLMRequest.
        
        If a ModelConfig is provided and the request's `max_tokens` is unset, this sets
        `request.max_tokens` to `model_config.max_tokens`.
        
        Parameters:
            request (LLMRequest): The request to update.
            model_config (Optional[ModelConfig]): Model configuration whose defaults may be applied.
        
        Returns:
            LLMRequest: The same request object, possibly updated with defaults.
        """
        if model_config:
            if request.max_tokens is None:
                request.max_tokens = model_config.max_tokens

        return request

    def _estimate_cost(self, usage: TokenUsage, model_config: Optional[ModelConfig] = None) -> float:
        """
        Estimate the monetary cost for the given token usage using the model's pricing.
        
        Parameters:
        	usage (TokenUsage): Token usage totals to price.
        	model_config (Optional[ModelConfig]): Model pricing configuration; its `cost_per_input` and `cost_per_output` are interpreted as cost per 1,000,000 tokens. If omitted, no cost is estimated.
        
        Returns:
        	estimated_cost (float): Estimated cost in the same currency/unit as the model_config prices.
        """
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
        """
        Retry an operation using exponential backoff with optional jitter.
        
        Parameters:
            operation (callable): A callable that accepts an `attempt` keyword argument (int) and performs the operation. The callable's return value is returned by this method.
            max_retries (Optional[int]): Maximum number of retry attempts (not counting the initial try). If omitted, uses provider config retry_count or 3.
            base_delay (float): Base delay in seconds for exponential backoff. If omitted, uses provider config retry_backoff or 1.0.
            max_delay (float): Upper bound in seconds for any backoff delay.
            jitter (bool): If true, apply random jitter to the computed backoff delay.
        
        Returns:
            The successful result returned by `operation`.
        
        Raises:
            AuthenticationError: If the operation fails due to authentication (not retried).
            Exception: The last exception raised by `operation` after all retries are exhausted.
        """
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
        """
        Constructs a ProviderError containing provider-specific context and an optional recovery hint.
        
        Parameters:
            message (str): Human-readable error message.
            status_code (Optional[int]): HTTP or provider status code used to produce a recovery hint, if available.
            response_body (Optional[str]): Raw response body from the provider to include for debugging.
            is_retryable (bool): Whether the error should be considered retryable.
        
        Returns:
            ProviderError: An error populated with provider name, status code, response body, a recovery_hint derived from the status code, and the retryable flag.
        """
        return ProviderError(
            message=message,
            provider=self.provider_name,
            status_code=status_code,
            response_body=response_body,
            recovery_hint=self._get_recovery_hint(status_code),
            is_retryable=is_retryable,
        )

    def _get_recovery_hint(self, status_code: Optional[int]) -> Optional[str]:
        """
        Return a user-facing recovery hint for common HTTP status codes.
        
        Parameters:
            status_code (Optional[int]): HTTP status code to look up.
        
        Returns:
            Optional[str]: A brief recovery hint for the given status code, or `None` if no hint is available.
        """
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
        """
        Provide a compact string describing the provider instance and its health.
        
        Returns:
            str: A representation containing the class name, `provider_name`, and the health status (`True` if healthy, `False` otherwise).
        """
        return f"<{self.__class__.__name__} provider={self.provider_name} healthy={self.health_status.is_healthy}>"

    def __del__(self):
        """
        Attempt to gracefully shut down provider resources when the instance is garbage-collected.
        
        If an internal client exists, runs the provider's asynchronous shutdown routine on the current event loop and suppresses any exceptions raised during the cleanup.
        """
        try:
            if self._client:
                import asyncio
                asyncio.get_event_loop().run_until_complete(self.shutdown())
        except Exception:
            pass