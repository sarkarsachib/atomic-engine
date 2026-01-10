#!/usr/bin/env python3
"""
Azure OpenAI Provider Implementation
Supports Azure OpenAI deployments
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

try:
    from openai import AsyncAzureOpenAI
    from openai.types import ChatCompletionChunk
    from openai._exceptions import OpenAIError, APIStatusError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("OpenAI SDK is required for Azure. Install with: pip install openai")


@dataclass
class AzureConfig:
    """Azure OpenAI-specific configuration"""
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    azure_deployment: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


class AzureProvider(BaseProvider):
    """Azure OpenAI LLM Provider"""

    provider_name = "azure"
    supports_streaming = True
    supports_vision = True
    supports_functions = True

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config, api_key, **kwargs)

        if not AZURE_AVAILABLE:
            raise ImportError("OpenAI SDK is required. Install with: pip install openai")

        self.azure_config = AzureConfig(
            api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=kwargs.get("azure_endpoint") or (config.api_base if config else None) or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=kwargs.get("api_version") or (config.api_version if config else "2024-02-15-preview"),
            azure_deployment=kwargs.get("azure_deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            timeout=kwargs.get("timeout") or (config.timeout_seconds if config else 60.0),
            max_retries=kwargs.get("max_retries") or (config.retry_count if config else 3),
        )

        self._client: Optional[AsyncAzureOpenAI] = None
        self._default_model = "gpt-4o"

    @property
    def default_model(self) -> str:
        """Default model for Azure"""
        return self._default_model

    async def initialize(self) -> None:
        """Initialize Azure OpenAI client"""
        if self._client is not None:
            return

        try:
            self._client = AsyncAzureOpenAI(
                api_key=self.azure_config.api_key,
                azure_endpoint=self.azure_config.azure_endpoint,
                api_version=self.azure_config.api_version,
                azure_deployment=self.azure_config.azure_deployment,
                timeout=self.azure_config.timeout,
                max_retries=self.azure_config.max_retries,
            )

            # Validate credentials
            await self.health_check()
            logger.info(f"Azure provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Azure OpenAI client"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Azure provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate non-streaming response"""
        await self.initialize()

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        request_id = None

        try:
            messages = self._prepare_messages(request)

            # Use azure_deployment if set, otherwise use model name
            deployment = self.azure_config.azure_deployment or request.model

            response = await self._client.chat.completions.create(
                model=deployment,
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

            return self._parse_response(response, request.model, latency_ms)

        except APIStatusError as e:
            self.health_status.record_failure(str(e))
            self._handle_api_error(e, request.model)
        except Exception as e:
            self.health_status.record_failure(str(e))
            raise self._create_error(str(e))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Generate streaming response"""
        await self.initialize()

        if not request.stream:
            raise ValueError("Streaming requested but stream=False")

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        chunk_index = 0
        accumulated_content = ""

        try:
            messages = self._prepare_messages(request)
            deployment = self.azure_config.azure_deployment or request.model

            response_stream = await self._client.chat.completions.create(
                model=deployment,
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
        """Prepare messages for Azure API (same as OpenAI)"""
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.extend(request.messages)

        return messages

    def _parse_response(self, response, model: str, latency_ms: float) -> LLMResponse:
        """Parse Azure response into standard format"""
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
        """Parse streaming chunk (same as OpenAI)"""
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

    def _handle_api_error(self, error, model: str) -> None:
        """Handle Azure API errors"""
        status_code = error.status_code
        response_body = error.response.text if error.response else None

        if status_code == 401:
            raise AuthenticationError(self.provider_name, "Azure API key")
        elif status_code == 429:
            retry_after = None
            if error.response and "Retry-After" in error.response.headers:
                try:
                    retry_after = float(error.response.headers["Retry-After"])
                except (ValueError, TypeError):
                    pass
            raise RateLimitError(
                message=f"Rate limit exceeded for Azure: {error.message}",
                provider=self.provider_name,
                retry_after_seconds=retry_after,
                limit_type="requests",
            )
        elif status_code == 400:
            if "context_length" in error.message.lower():
                raise ContextLengthError(
                    provider=self.provider_name,
                    model=model,
                )
            raise ProviderError(
                message=f"Bad request: {error.message}",
                provider=self.provider_name,
                status_code=400,
                response_body=response_body,
            )
        else:
            raise ProviderError(
                message=f"Azure API error: {error.message}",
                provider=self.provider_name,
                status_code=status_code,
                response_body=response_body,
            )

    async def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer"""
        await self.initialize()

        try:
            response = await self._client.chat.completions.create(
                model=self._default_model,
                messages=[{"role": "user", "content": text}],
                max_tokens=1,
            )
            return response.usage.prompt_tokens
        except Exception:
            return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """Get list of available Azure deployments"""
        # Azure doesn't have a direct model listing API
        # Return common deployments
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-35-turbo"]

    async def health_check(self) -> bool:
        """Check Azure API health"""
        try:
            deployment = self.azure_config.azure_deployment or self._default_model
            await self._client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning(f"Azure health check failed: {e}")
            return False


# Alias for backward compatibility
AzureOpenAI = AzureProvider
