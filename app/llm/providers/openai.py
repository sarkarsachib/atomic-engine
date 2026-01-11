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
        """Default model for OpenAI"""
        return self._default_model

    async def initialize(self) -> None:
        """Initialize OpenAI client"""
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
        """Shutdown OpenAI client"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("OpenAI provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate non-streaming response"""
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
        """Generate streaming response"""
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
        """Prepare messages for OpenAI API"""
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Add user messages
        messages.extend(request.messages)

        return messages

    def _parse_response(self, response, model: str, latency_ms: float) -> LLMResponse:
        """Parse OpenAI response into standard format"""
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
        """Parse streaming chunk and return delta, finish_reason, and usage"""
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
        """Handle OpenAI API errors and raise appropriate exception"""
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
        """Count tokens using OpenAI's tokenizer"""
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
        """Get list of available OpenAI models"""
        await self.initialize()

        try:
            models = await self._client.models.list()
            return [m.id for m in models.data if "gpt" in m.id.lower() or "o1" in m.id.lower()]
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            return []

    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False


# Alias for backward compatibility
OpenAIClient = OpenAIProvider
