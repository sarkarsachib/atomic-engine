#!/usr/bin/env python3
"""
Custom Exceptions for LLM Agent Layer
Comprehensive error handling with detailed context and recovery hints
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import traceback
import json


@dataclass
class ErrorContext:
    """Context information for errors"""
    provider: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    attempt_number: int = 1
    timestamp: str = field(default_factory=lambda: str(__import__('datetime').datetime.now()))
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Produce a dictionary representation of the error context.
        
        Returns:
            dict: Mapping with keys "provider", "model", "request_id", "attempt_number", "timestamp", and "additional_info" containing the corresponding context values.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp,
            "additional_info": self.additional_info,
        }


class LLMError(Exception):
    """Base exception for all LLM-related errors"""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        recovery_hint: Optional[str] = None,
        is_retryable: bool = True,
    ):
        """
        Initialize the base LLMError with contextual metadata, optional recovery guidance, and retryability.
        
        Parameters:
            message (str): Human-readable error message describing the failure.
            context (Optional[ErrorContext]): Contextual information about the error (provider, model, request id, timestamps, etc.). If omitted, a default empty ErrorContext is created.
            recovery_hint (Optional[str]): Suggested action to recover from the error, suitable for display to operators or automated recovery logic.
            is_retryable (bool): Indicates whether the operation that caused the error may be retried.
        
        Notes:
            Captures the full traceback at the time of construction for inclusion in serialized error representations.
        """
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.recovery_hint = recovery_hint
        self.is_retryable = is_retryable
        self._full_traceback = traceback.format_exc()

    def __str__(self):
        """
        Return a human-readable description of the error including its class and message, and append the provider if available.
        
        Returns:
            str: Formatted error description, e.g. "[ClassName] message (provider: provider_name)" or "[ClassName] message".
        """
        base = f"[{self.__class__.__name__}] {self.message}"
        if self.context.provider:
            base += f" (provider: {self.context.provider})"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """
        Produce a serializable dictionary representing the exception, including its type, message, context, recovery guidance, retryability, and captured traceback.
        
        Returns:
            error_dict (Dict[str, Any]): Dictionary with keys:
                - `error_type`: Exception class name.
                - `message`: Human-readable error message.
                - `context`: Result of `self.context.to_dict()` with contextual metadata.
                - `recovery_hint`: Suggested action to recover from the error (may be None).
                - `is_retryable`: `True` if the operation may be retried, `False` otherwise.
                - `traceback`: Captured full traceback string for diagnostics.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context.to_dict(),
            "recovery_hint": self.recovery_hint,
            "is_retryable": self.is_retryable,
            "traceback": self._full_traceback,
        }

    def to_json(self) -> str:
        """
        Return a JSON-formatted string representation of the error, including its type, message, context, recovery hint, retryability flag, and traceback.
        
        Returns:
            json_str (str): JSON string containing keys `error_type`, `message`, `context`, `recovery_hint`, `is_retryable`, and `traceback`.
        """
        return json.dumps(self.to_dict())

    def with_context(self, **kwargs) -> "LLMError":
        """
        Create a new error instance preserving the original message, recovery hint, and retryability while merging additional context.
        
        Parameters:
            **kwargs: Additional context fields to merge into the error's ErrorContext. Keys may match ErrorContext attributes (e.g., provider, model, request_id, attempt_number, timestamp) or provide extra entries for `additional_info`.
        
        Returns:
            LLMError: A new instance of the same error class with an updated ErrorContext that merges the original context and the provided kwargs.
        """
        new_context = ErrorContext(**{**self.context.__dict__, **kwargs})
        return self.__class__(
            message=self.message,
            context=new_context,
            recovery_hint=self.recovery_hint,
            is_retryable=self.is_retryable,
        )


class ProviderError(LLMError):
    """Exception for provider-specific errors"""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recovery_hint: Optional[str] = None,
        is_retryable: bool = True,
    ):
        """
        Initialize a ProviderError with provider-specific context and optional HTTP response details.
        
        Parameters:
            message (str): Human-readable error message.
            provider (str): Name of the provider associated with the error; set on the error context.
            status_code (Optional[int]): HTTP status code related to the error; stored in context.additional_info["status_code"].
            response_body (Optional[str]): HTTP response body related to the error; stored in context.additional_info["response_body"].
            context (Optional[ErrorContext]): Existing ErrorContext to augment; if omitted, a new ErrorContext is created and populated with provider and additional info.
            recovery_hint (Optional[str]): Optional guidance for recovering from the error.
            is_retryable (bool): Whether the error is considered retryable.
        """
        error_context = context or ErrorContext()
        error_context.provider = provider
        error_context.additional_info["status_code"] = status_code
        error_context.additional_info["response_body"] = response_body

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint=recovery_hint,
            is_retryable=is_retryable,
        )
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(ProviderError):
    """Exception for authentication failures"""

    def __init__(
        self,
        provider: str,
        auth_type: str = "API key",
        context: Optional[ErrorContext] = None,
    ):
        """
        Create an AuthenticationError for the given provider indicating invalid or missing credentials.
        
        Parameters:
            provider (str): The provider whose authentication failed.
            auth_type (str): Type of authentication used (e.g., "API key"). Defaults to "API key".
            context (Optional[ErrorContext]): Additional contextual information to attach to the error.
        
        Notes:
            The instance's status_code is set to 401, is_retryable is set to False, and recovery_hint advises checking the specified credentials and permissions for the provider.
        """
        super().__init__(
            message=f"Authentication failed for {provider}: Invalid or missing {auth_type}",
            provider=provider,
            status_code=401,
            context=context,
            recovery_hint=f"Check your {auth_type} for {provider} and ensure it's valid and has sufficient permissions",
            is_retryable=False,
        )
        self.auth_type = auth_type


class AuthorizationError(ProviderError):
    """Exception for authorization/permission errors"""

    def __init__(
        self,
        provider: str,
        required_permission: str,
        context: Optional[ErrorContext] = None,
    ):
        """
        Create an AuthorizationError indicating that the given provider request lacked a required permission.
        
        Parameters:
            provider (str): The provider where authorization failed (e.g., "openai").
            required_permission (str): The specific permission that is missing.
            context (Optional[ErrorContext]): Optional contextual information to attach to the error.
        """
        super().__init__(
            message=f"Authorization failed for {provider}: Missing required permission '{required_permission}'",
            provider=provider,
            status_code=403,
            context=context,
            recovery_hint=f"Ensure your API key has the '{required_permission}' permission",
            is_retryable=False,
        )
        self.required_permission = required_permission


class RateLimitError(LLMError):
    """Exception for rate limit violations"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        limit_type: str = "requests",
        current_usage: Optional[int] = None,
        limit_value: Optional[int] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a RateLimitError carrying provider and rate-limit metadata.
        
        Records provided values into the error's context (under `additional_info`), sets a recovery hint that advises waiting `retry_after_seconds` seconds when given (otherwise suggests exponential backoff), and marks the error as retryable.
        
        Parameters:
            message (str): Human-readable error message.
            provider (Optional[str]): Optional provider identifier to attach to the error context.
            retry_after_seconds (Optional[float]): Recommended wait time before retrying; used to build the recovery hint.
            limit_type (str): Type of limit that was hit (e.g., "requests", "tokens").
            current_usage (Optional[int]): Current usage count relevant to the limit.
            limit_value (Optional[int]): Configured limit value relevant to the limit.
            context (Optional[ErrorContext]): Existing ErrorContext to extend; a new context is created if omitted.
        """
        error_context = context or ErrorContext()
        if provider:
            error_context.provider = provider
        error_context.additional_info["retry_after_seconds"] = retry_after_seconds
        error_context.additional_info["limit_type"] = limit_type
        error_context.additional_info["current_usage"] = current_usage
        error_context.additional_info["limit_value"] = limit_value

        recovery_hint = f"Wait {retry_after_seconds:.1f} seconds before retrying" if retry_after_seconds else "Implement exponential backoff"

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint=recovery_hint,
            is_retryable=True,
        )
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type
        self.current_usage = current_usage
        self.limit_value = limit_value

    @classmethod
    def from_headers(
        cls,
        provider: str,
        headers: Dict[str, str],
        limit_type: str = "requests",
    ) -> "RateLimitError":
        """
        Constructs a RateLimitError from HTTP rate-limit headers for a given provider.
        
        Parameters:
            provider (str): Name of the provider that returned the headers.
            headers (Dict[str, str]): HTTP response headers potentially containing
                'Retry-After', 'X-RateLimit-Limit', and 'X-RateLimit-Remaining'.
                Numeric header values that cannot be parsed are ignored.
            limit_type (str): Human-readable label for the limit type (default: "requests").
        
        Returns:
            RateLimitError: An instance populated with provider, retry_after_seconds
            (float or None), limit_type, current_usage (int or None), and limit_value (int or None).
            The error message includes remaining/limit details when both are available.
        """
        retry_after = None
        limit = None
        remaining = None

        if "Retry-After" in headers:
            try:
                retry_after = float(headers["Retry-After"])
            except ValueError:
                pass

        if "X-RateLimit-Limit" in headers:
            try:
                limit = int(headers["X-RateLimit-Limit"])
            except ValueError:
                pass

        if "X-RateLimit-Remaining" in headers:
            try:
                remaining = int(headers["X-RateLimit-Remaining"])
            except ValueError:
                pass

        message = f"Rate limit exceeded for {provider}"
        if remaining is not None and limit is not None:
            message += f" ({remaining}/{limit} {limit_type})"

        return cls(
            message=message,
            provider=provider,
            retry_after_seconds=retry_after,
            limit_type=limit_type,
            current_usage=remaining,
            limit_value=limit,
        )


class TokenLimitError(LLMError):
    """Exception for token limit violations"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        input_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a TokenLimitError with contextual information about token usage and model/provider.
        
        Parameters:
            message (str): Human-readable error message describing the token limit condition.
            provider (Optional[str]): Provider identifier to attach to the error context.
            model (Optional[str]): Model identifier to attach to the error context.
            input_tokens (Optional[int]): Number of tokens present in the input that triggered the error.
            max_tokens (Optional[int]): Maximum allowed tokens for the model or request.
            context (Optional[ErrorContext]): Existing ErrorContext to extend; if omitted, a new context is created.
        
        Notes:
            - The error populates the provided or new ErrorContext with provider, model, and `input_tokens`/`max_tokens` in `additional_info`.
            - The error includes a recovery hint recommending reducing input size or using a model with a larger context window.
            - The error is marked as retryable.
        """
        error_context = context or ErrorContext()
        if provider:
            error_context.provider = provider
        error_context.model = model
        error_context.additional_info["input_tokens"] = input_tokens
        error_context.additional_info["max_tokens"] = max_tokens

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint="Reduce input size or use a model with larger context window",
            is_retryable=True,
        )
        self.provider = provider
        self.model = model
        self.input_tokens = input_tokens
        self.max_tokens = max_tokens


class ContextLengthError(TokenLimitError):
    """Exception for context length exceeded"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context_length: Optional[int] = None,
        requested_length: Optional[int] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Indicates that a request's context length exceeded the model's maximum context window.
        
        Parameters:
            provider (Optional[str]): Provider identifier to include in the error context.
            model (Optional[str]): Model identifier to include in the error context.
            context_length (Optional[int]): Maximum context/window size (in tokens) supported by the model.
            requested_length (Optional[int]): Number of tokens requested/used that exceeded the context length.
            context (Optional[ErrorContext]): Existing error context to be extended for this error.
        """
        message = "Context length exceeded"
        if context_length and requested_length:
            message += f": Requested {requested_length} tokens, max context is {context_length}"

        super().__init__(
            message=message,
            provider=provider,
            model=model,
            input_tokens=requested_length,
            max_tokens=context_length,
            context=context,
        )
        self.context_length = context_length
        self.requested_length = requested_length


class OutputLengthError(TokenLimitError):
    """Exception for output length exceeded"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        requested_tokens: Optional[int] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Indicates the model's generated output would exceed the configured maximum output tokens.
        
        Parameters:
            provider: Optional provider identifier to attach to the error context.
            model: Optional model identifier to attach to the error context.
            max_output_tokens: The configured maximum number of output tokens allowed.
            requested_tokens: The number of tokens that were requested/generated.
            context: Optional ErrorContext to attach additional contextual information.
        """
        message = "Output length exceeded maximum"
        if max_output_tokens and requested_tokens:
            message += f": Requested {requested_tokens}, max is {max_output_tokens}"

        super().__init__(
            message=message,
            provider=provider,
            model=model,
            input_tokens=requested_tokens,
            max_tokens=max_output_tokens,
            context=context,
        )
        self.max_output_tokens = max_output_tokens
        self.requested_tokens = requested_tokens


class ConfigurationError(LLMError):
    """Exception for configuration errors"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a ConfigurationError representing a problem with application configuration.
        
        Parameters:
            message (str): Human-readable description of the configuration error.
            config_key (Optional[str]): The configuration key associated with the error, if known.
            config_value (Optional[Any]): The value found for the configuration key; stored as a string in the error context.
            context (Optional[ErrorContext]): Existing error context to augment; a new context is created if none is provided.
        
        Notes:
            Records `config_key` and `config_value` in the error's context additional_info, sets a recovery hint suggesting to check configuration, marks the error as not retryable, and stores `config_key` and `config_value` on the instance.
        """
        error_context = context or ErrorContext()
        error_context.additional_info["config_key"] = config_key
        error_context.additional_info["config_value"] = str(config_value)

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint="Check your configuration file and ensure all required values are set correctly",
            is_retryable=False,
        )
        self.config_key = config_key
        self.config_value = config_value


class ModelNotFoundError(ProviderError):
    """Exception when requested model is not available"""

    def __init__(
        self,
        provider: str,
        model: str,
        available_models: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a ModelNotFoundError for a provider when the requested model is unavailable.
        
        Parameters:
            provider (str): The provider where the model was requested.
            model (str): The name of the unavailable model.
            available_models (Optional[List[str]]): A list of models available from the provider; used to enrich the error message, recovery hint, and context.
            context (Optional[ErrorContext]): Existing ErrorContext to augment; if omitted a new ErrorContext is created.
        
        """
        message = f"Model '{model}' not available for provider '{provider}'"
        error_context = context or ErrorContext()
        error_context.model = model
        error_context.additional_info["available_models"] = available_models

        if available_models:
            message += f". Available models: {', '.join(available_models[:10])}"
            if len(available_models) > 10:
                message += f" and {len(available_models) - 10} more"

        super().__init__(
            message=message,
            provider=provider,
            status_code=404,
            context=error_context,
            recovery_hint=f"Use one of the available models: {', '.join(available_models[:5]) if available_models else 'Check provider documentation'}",
            is_retryable=True,
        )
        self.model = model
        self.available_models = available_models


class StreamingError(LLMError):
    """Exception for streaming-related errors"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        stream_id: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recovery_hint: Optional[str] = None,
        is_retryable: bool = True,
    ):
        """
        Create a StreamingError with contextual provider and stream identifier information.
        
        Parameters:
        	message (str): Human-readable error message describing the streaming failure.
        	provider (Optional[str]): Name of the provider associated with the stream; stored on the error context and the instance.
        	stream_id (Optional[str]): Identifier for the streaming session; added to context.additional_info under "stream_id" and stored on the instance.
        	context (Optional[ErrorContext]): Existing ErrorContext to augment; a new ErrorContext is created if omitted.
        	recovery_hint (Optional[str]): Custom recovery guidance; defaults to "Check your streaming connection and try again" when not provided.
        	is_retryable (bool): Whether the error should be considered retryable.
        """
        error_context = context or ErrorContext()
        if provider:
            error_context.provider = provider
        error_context.additional_info["stream_id"] = stream_id

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint=recovery_hint or "Check your streaming connection and try again",
            is_retryable=is_retryable,
        )
        self.provider = provider
        self.stream_id = stream_id


class ConnectionError(LLMError):
    """Exception for network/connection errors"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a ConnectionError with provider, endpoint, timeout details, and a default recovery hint.
        
        Parameters:
            message (str): Human-readable error message.
            provider (Optional[str]): Name of the provider or service involved.
            endpoint (Optional[str]): Endpoint or URL related to the connection failure.
            timeout_seconds (Optional[float]): Timeout threshold (in seconds) associated with the failure.
            context (Optional[ErrorContext]): Existing error context to enrich; a new context is created if omitted.
        """
        error_context = context or ErrorContext()
        if provider:
            error_context.provider = provider
        error_context.additional_info["endpoint"] = endpoint
        error_context.additional_info["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint="Check your network connection and API endpoint configuration",
            is_retryable=True,
        )
        self.provider = provider
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds


class TimeoutError(LLMError):
    """Exception for request timeouts"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a TimeoutError with optional provider and timing context.
        
        Parameters:
            message (str): Human-readable error message describing the timeout.
            provider (Optional[str]): Name of the provider associated with the timeout, if any.
            timeout_seconds (Optional[float]): Configured timeout threshold in seconds.
            elapsed_seconds (Optional[float]): Measured elapsed time in seconds when the timeout occurred.
            context (Optional[ErrorContext]): Existing ErrorContext to augment; a new context is created if omitted.
        
        Notes:
            The exception includes a recovery hint recommending a shorter request or increased timeout and is marked retryable.
        """
        error_context = context or ErrorContext()
        if provider:
            error_context.provider = provider
        error_context.additional_info["timeout_seconds"] = timeout_seconds
        error_context.additional_info["elapsed_seconds"] = elapsed_seconds

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint="Try again with a shorter request or increase timeout",
            is_retryable=True,
        )
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class HealthCheckError(LLMError):
    """Exception for provider health check failures"""

    def __init__(
        self,
        provider: str,
        reason: str,
        is_critical: bool = False,
        context: Optional[ErrorContext] = None,
    ):
        """
        Initialize a HealthCheckError for a failed provider health check.
        
        Parameters:
            provider (str): The name of the provider that failed its health check.
            reason (str): A concise explanation of why the health check failed.
            is_critical (bool): Whether the health check failure is critical. Stored on the exception.
            context (Optional[ErrorContext]): Optional contextual information to attach; if omitted, a new ErrorContext with the provider is created.
        
        Notes:
            The exception includes a recovery hint advising to check the provider status page and is marked as retryable.
        """
        message = f"Health check failed for {provider}: {reason}"

        super().__init__(
            message=message,
            context=context or ErrorContext(provider=provider),
            recovery_hint="Check provider status page and your configuration",
            is_retryable=True,
        )
        self.provider = provider
        self.reason = reason
        self.is_critical = is_critical


class FallbackError(LLMError):
    """Exception when all providers fail"""

    def __init__(
        self,
        providers: List[str],
        last_error: Optional[LLMError] = None,
        context: Optional[ErrorContext] = None,
    ):
        """
        Create a FallbackError representing that all listed providers failed.
        
        Parameters:
            providers (List[str]): Ordered list of provider identifiers that were attempted.
            last_error (Optional[LLMError]): The most recent error from a provider; included in context.additional_info as a dict if provided.
            context (Optional[ErrorContext]): Base error context to extend; if omitted a new ErrorContext is created.
        
        Notes:
            - Adds `failed_providers` and `last_error` (as a dict or None) to `context.additional_info`.
            - Sets a recovery hint advising to check API keys, network connection, and provider status.
            - Marks the error as not retryable and stores `providers` and `last_error` on the instance.
        """
        message = f"All providers failed: {' -> '.join(providers)}"

        error_context = context or ErrorContext()
        error_context.additional_info["failed_providers"] = providers
        error_context.additional_info["last_error"] = last_error.to_dict() if last_error else None

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint="Check your API keys, network connection, and provider status",
            is_retryable=False,
        )
        self.providers = providers
        self.last_error = last_error


class RoutingError(LLMError):
    """Exception for routing errors"""

    def __init__(
        self,
        message: str,
        strategy: str,
        context: Optional[ErrorContext] = None,
    ):
        """
        Create a RoutingError that records the failing routing strategy and attaches it to the error context.
        
        Parameters:
            message (str): Human-readable explanation of the routing failure.
            strategy (str): Identifier of the routing strategy involved (e.g., "round_robin", "failover").
            context (Optional[ErrorContext]): Existing error context to enrich; if omitted, a new context is created and the strategy is recorded in its additional_info.
        """
        error_context = context or ErrorContext()
        error_context.additional_info["strategy"] = strategy

        super().__init__(
            message=message,
            context=error_context,
            recovery_hint="Check your routing configuration and provider availability",
            is_retryable=True,
        )
        self.strategy = strategy


# Exception mapping for HTTP status codes
HTTP_STATUS_TO_EXCEPTION = {
    400: lambda p, m, h, b: ProviderError(f"Bad request: {m}", p, 400, b),
    401: lambda p, m, h, b: AuthenticationError(p),
    403: lambda p, m, h, b: AuthorizationError(p, "unknown"),
    404: lambda p, m, h, b: ProviderError(f"Resource not found: {m}", p, 404, b),
    429: lambda p, m, h, b: RateLimitError.from_headers(p, h),
    500: lambda p, m, h, b: ProviderError(f"Internal server error: {m}", p, 500, b),
    502: lambda p, m, h, b: ProviderError(f"Bad gateway: {m}", p, 502, b),
    503: lambda p, m, h, b: ProviderError(f"Service unavailable: {m}", p, 503, b),
    504: lambda p, m, h, b: TimeoutError(f"Gateway timeout: {m}", p),
}