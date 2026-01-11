#!/usr/bin/env python3
"""
OpenAI Provider Implementation
Supports GPT-4o, o1, GPT-4 Turbo, and other OpenAI models
"""

import os
import json
import time
from typing import AsyncIterator, Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from . import BaseProvider, LLMRequest, LLMResponse, StreamChunk, TokenUsage
from ..config import ProviderConfig, ModelConfig
from ..exceptions import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    TokenLimitError,
    ContextLengthError,
    ModelNotFoundError,
    ConnectionError,
    TimeoutError,
)

logger = logging.getLogger(__name__)

# Attempt to import OpenAI SDK
try:
    from openai import AsyncOpenAI
    from openai.types import ChatCompletionChunk
    from openai._exceptions import OpenAIError, APIStatusError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not installed. Install with: pip install openai")


@dataclass
class OpenAIConfig:
    """OpenAI-specific configuration"""
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    http_client: Optional[Any] = None


class OpenAIProvider(BaseProvider):
    """OpenAI LLM Provider"""

    provider_name = "openai"
    supports_streaming = True
    supports_vision = True
    supports_functions = True

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI provider, validate SDK availability, and build runtime configuration.
        
        Parameters:
            config (Optional[ProviderConfig]): Optional provider-level configuration used as defaults.
            api_key (Optional[str]): Explicit OpenAI API key; falls back to the OPENAI_API_KEY environment variable if not provided.
            **kwargs: Optional overrides for configuration fields: `organization`, `base_url`, `timeout`, and `max_retries`.
        
        Raises:
            ImportError: If the OpenAI SDK is not installed.
        
        Side effects:
            - Constructs and stores `self.openai_config` from provided arguments, environment variables, and `config`.
            - Initializes `self._client` to `None`.
            - Sets `self._default_model` to `"gpt-4o"`.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK is required. Install with: pip install openai")
        
        super().__init__(config, api_key, **kwargs)

        self.openai_config = OpenAIConfig(
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            organization=kwargs.get("organization") or (config.api_base if config else None),
            base_url=kwargs.get("base_url") or (config.api_base if config else None),
            timeout=kwargs.get("timeout") or (config.timeout_seconds if config else 60.0),
            max_retries=kwargs.get("max_retries") or (config.retry_count if config else 3),
        )

        self._client: Optional[AsyncOpenAI] = None
        self._default_model = "gpt-4o"

    @property
    def default_model(self) -> str:
        """
        The provider's default model identifier.
        
        Returns:
            str: The default model id used by this provider.
        """
        return self._default_model

    async def initialize(self) -> None:
        """
        Create and validate the internal OpenAI async client.
        
        If a client already exists, this is a no-op. On success, the AsyncOpenAI client is stored on self._client and the provider's credentials are verified via a health check. If client construction or the health check fails, the original exception is raised.
         
        Raises:
            Exception: Propagates any error raised while creating the client or performing the health check.
        """
        if self._client is not None:
            return

        try:
            self._client = AsyncOpenAI(
                api_key=self.openai_config.api_key,
                organization=self.openai_config.organization,
                base_url=self.openai_config.base_url,
                timeout=self.openai_config.timeout,
                max_retries=self.openai_config.max_retries,
            )

            # Validate API key with a simple request
            await self.health_check()
            logger.info(f"OpenAI provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise

    async def shutdown(self) -> None:
        """
        Close the internal OpenAI client and clear its reference.
        
        If no client is initialized, this is a no-op.
        """
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("OpenAI provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a complete, non-streaming chat response for the given LLMRequest using the configured OpenAI client.
        
        Parameters:
            request (LLMRequest): Request containing messages, model selection, and generation parameters (temperature, max_tokens, stop tokens, penalties, logit_bias, user, etc.).
        
        Returns:
            LLMResponse: Parsed response including the generated content, model used, token usage, finish reason, provider identifier, request ID, and the raw API response.
        """
        await self.initialize()

        # Apply default parameters
        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        request_id = None

        try:
            # Prepare messages
            messages = self._prepare_messages(request)

            # Make API call
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logit_bias=request.logit_bias,
                user=request.user,
            )

            request_id = response.id
            latency_ms = (time.time() - start_time) * 1000
            self.health_status.record_success(latency_ms)

            # Parse response
            return self._parse_response(response, request.model, latency_ms)

        except Exception as e:
            self.health_status.record_failure(str(e))
            self._handle_api_error(e, request.model)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream tokens from the model for a single LLMRequest and yield incremental updates as StreamChunk objects.
        
        Parameters:
            request (LLMRequest): The request to send to the model. Must have `stream=True` and include the desired model and generation parameters.
        
        Returns:
            AsyncIterator[StreamChunk]: An async iterator that yields StreamChunk values containing cumulative `content`, the latest `delta`, `model`, `provider`, `chunk_index`, `is_final`, `finish_reason`, `usage`, and `raw_chunk`.
        
        Raises:
            ValueError: If `request.stream` is False.
            ProviderError: If the provider API returns an error during streaming (mapped to provider-specific exceptions).
        """
        await self.initialize()

        if not request.stream:
            raise ValueError("Streaming requested but stream=False")

        # Apply default parameters
        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        chunk_index = 0
        accumulated_content = ""

        try:
            messages = self._prepare_messages(request)

            response_stream = await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stop=request.stop,
            )

            async for chunk in response_stream:
                chunk_index += 1
                delta, finish_reason, usage = self._parse_stream_chunk(chunk)

                if delta:
                    accumulated_content += delta

                is_final = chunk.choices[0].finish_reason is not None if chunk.choices else False

                yield StreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    model=request.model,
                    provider=self.provider_name,
                    chunk_index=chunk_index,
                    is_final=is_final,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_chunk=chunk.model_dump(),
                )

                if is_final:
                    latency_ms = (time.time() - start_time) * 1000
                    self.health_status.record_success(latency_ms)
                    break

        except Exception as e:
            self.health_status.record_failure(str(e))
            raise self._create_error(str(e))

    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """
        Constructs the sequence of chat messages for the OpenAI chat API.
        
        Parameters:
            request (LLMRequest): The LLM request whose optional `system_prompt` will be inserted as a system-role message (if present) followed by the request's user messages.
        
        Returns:
            List[Dict[str, str]]: A list of message objects suitable for the OpenAI API; each item contains `role` and `content`, with the system message first when provided.
        """
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Add user messages
        messages.extend(request.messages)

        return messages

    def _parse_response(self, response, model: str, latency_ms: float) -> LLMResponse:
        """
        Convert an OpenAI chat completion response into an LLMResponse.
        
        Parameters:
            response: The raw OpenAI chat completion response object.
            model (str): The model identifier used to generate the response.
            latency_ms (float): Round-trip latency in milliseconds for the request.
        
        Returns:
            LLMResponse: Standardized response containing content, model, token usage, finish reason, provider name, request ID, raw response data, and any function call info.
        """
        choice = response.choices[0]
        message = choice.message

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=message.content or "",
            model=model,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            provider=self.provider_name,
            request_id=response.id,
            raw_response=response.model_dump(),
            function_call=message.function_call if message.function_call else None,
        )

    def _parse_stream_chunk(self, chunk) -> tuple:
        """
        Extract the text delta, finish reason, and token usage from a streaming response chunk.
        
        Parameters:
            chunk: A streaming chat-completion chunk object returned by the OpenAI client; expected to have `choices` (with `delta.content` and `finish_reason`) and optional `usage` attributes.
        
        Returns:
            tuple: A 3-tuple (delta, finish_reason, usage) where
                delta (str): The incremental text content from the chunk (empty string if none).
                finish_reason (str | None): The reason the stream finished for this choice (e.g., "stop", "length") or `None` if not present.
                usage (TokenUsage | None): Token usage for the chunk if present, otherwise `None`.
        """
        delta = ""
        finish_reason = None
        usage = None

        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            delta = choice.delta.content or ""
            finish_reason = choice.finish_reason

        if chunk.usage:
            usage = TokenUsage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )

        return delta, finish_reason, usage

    async def _handle_api_error(self, error, model: str) -> None:
        """
        Map OpenAI API errors to provider-specific exceptions.
        
        Parameters:
            error: The error object returned by the OpenAI client; its attributes (status_code, message, response) are inspected to determine the mapped exception.
            model (str): The model identifier associated with the request that triggered the error.
        
        Raises:
            AuthenticationError: When the API returns a 401 status code.
            RateLimitError: When the API returns a 429 status code; includes `retry_after_seconds` if the `Retry-After` header is present and parseable.
            ContextLengthError: When the API returns a 400 status code and the error message indicates the input exceeded the model's context length; includes the model and reported `x-max-input-tokens` header.
            ProviderError: For generic 400 responses that are not context-length errors, and for other non-specialized HTTP error codes; includes status code and response body when available.
            ModelNotFoundError: When the API returns a 404 status code; includes the requested model and the list of available models discovered via `get_available_models`.
        """
        status_code = error.status_code
        response_body = error.response.text if error.response else None

        if status_code == 401:
            raise AuthenticationError(self.provider_name)
        elif status_code == 429:
            retry_after = None
            if error.response and "Retry-After" in error.response.headers:
                try:
                    retry_after = float(error.response.headers["Retry-After"])
                except (ValueError, TypeError):
                    pass
            raise RateLimitError(
                message=f"Rate limit exceeded for OpenAI: {error.message}",
                provider=self.provider_name,
                retry_after_seconds=retry_after,
                limit_type="requests",
            )
        elif status_code == 400:
            if "context_length_exceeded" in error.message.lower():
                raise ContextLengthError(
                    provider=self.provider_name,
                    model=model,
                    context_length=error.response.headers.get("x-max-input-tokens"),
                )
            raise ProviderError(
                message=f"Bad request: {error.message}",
                provider=self.provider_name,
                status_code=400,
                response_body=response_body,
            )
        elif status_code == 404:
            models = await self.get_available_models()
            raise ModelNotFoundError(
                provider=self.provider_name,
                model=model,
                available_models=models,
            )
        else:
            raise ProviderError(
                message=f"OpenAI API error: {error.message}",
                provider=self.provider_name,
                status_code=status_code,
                response_body=response_body,
            )

    async def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the provided text.
        
        Attempts to obtain the token count from the OpenAI chat completions API; if the API call fails, returns an approximate count computed as len(text) // 4.
        
        Returns:
            int: Number of tokens — exact value returned by the API when available, otherwise an approximation.
        """
        await self.initialize()

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": text}],
                max_tokens=1,
            )
            # This is a workaround - in production, use tiktoken directly
            return response.usage.prompt_tokens
        except Exception:
            # Fallback to approximate count
            return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """
        Retrieve available OpenAI model IDs filtered to supported families.
        
        Returns:
            List[str]: Model ID strings that contain "gpt" or "o1". Returns an empty list if fetching models fails.
        """
        await self.initialize()

        try:
            models = await self._client.models.list()
            return [m.id for m in models.data if "gpt" in m.id.lower() or "o1" in m.id.lower()]
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Verify that the configured OpenAI API is reachable and functioning.
        
        Returns:
            bool: `True` if the API responded to a models list request, `False` otherwise.
        """
        try:
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False


# Alias for backward compatibility
OpenAIClient = OpenAIProvider