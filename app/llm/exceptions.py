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
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.recovery_hint = recovery_hint
        self.is_retryable = is_retryable
        self._full_traceback = traceback.format_exc()

    def __str__(self):
        base = f"[{self.__class__.__name__}] {self.message}"
        if self.context.provider:
            base += f" (provider: {self.context.provider})"
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context.to_dict(),
            "recovery_hint": self.recovery_hint,
            "is_retryable": self.is_retryable,
            "traceback": self._full_traceback,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def with_context(self, **kwargs) -> "LLMError":
        """Create a new error with additional context"""
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
        """Create RateLimitError from HTTP headers"""
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
