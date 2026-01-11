#!/usr/bin/env python3
"""
Anthropic Claude Provider Implementation
Supports Claude 3.5 Sonnet, Opus, and Haiku models
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
    from anthropic import AsyncAnthropic
    from anthropic.types import TextBlock, ToolUseBlock
    from anthropic._exceptions import APIError, APIStatusError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not installed. Install with: pip install anthropic")


@dataclass
class AnthropicConfig:
    """Anthropic-specific configuration"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


class AnthropicProvider(BaseProvider):
    """Anthropic Claude LLM Provider"""

    provider_name = "anthropic"
    supports_streaming = True
    supports_vision = True
    supports_functions = True  # Via tool use

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config, api_key, **kwargs)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK is required. Install with: pip install anthropic")

        self.anthropic_config = AnthropicConfig(
            api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=kwargs.get("base_url") or (config.api_base if config else None),
            timeout=kwargs.get("timeout") or (config.timeout_seconds if config else 60.0),
            max_retries=kwargs.get("max_retries") or (config.retry_count if config else 3),
        )

        self._client: Optional[AsyncAnthropic] = None
        self._default_model = "claude-sonnet-4-20250514"

    @property
    def default_model(self) -> str:
        """Default model for Anthropic"""
        return self._default_model

    async def initialize(self) -> None:
        """Initialize Anthropic client"""
        if self._client is not None:
            return

        try:
            self._client = AsyncAnthropic(
                api_key=self.anthropic_config.api_key,
                base_url=self.anthropic_config.base_url,
                timeout=self.anthropic_config.timeout,
                max_retries=self.anthropic_config.max_retries,
            )

            # Validate API key with a simple request
            await self.health_check()
            logger.info(f"Anthropic provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Anthropic client"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Anthropic provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate non-streaming response"""
        await self.initialize()

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        request_id = None

        try:
            # Prepare messages in Anthropic format
            messages = self._prepare_messages(request)

            # Convert functions to tools if needed
            tools = self._prepare_tools(request)

            # Make API call
            response = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens or 4096,
                messages=messages,
                temperature=request.temperature,
                stop_sequences=request.stop,
                tools=tools if tools else None,
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

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        chunk_index = 0
        accumulated_content = ""

        try:
            messages = self._prepare_messages(request)
            tools = self._prepare_tools(request)

            async with self._client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens or 4096,
                messages=messages,
                temperature=request.temperature,
                stop_sequences=request.stop,
                tools=tools if tools else None,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta.text or ""
                        accumulated_content += delta

                        yield StreamChunk(
                            content=accumulated_content,
                            delta=delta,
                            model=request.model,
                            provider=self.provider_name,
                            chunk_index=chunk_index,
                            is_final=False,
                            raw_chunk=event.model_dump(),
                        )
                        chunk_index += 1

                    elif event.type == "message_delta":
                        yield StreamChunk(
                            content=accumulated_content,
                            delta="",
                            model=request.model,
                            provider=self.provider_name,
                            chunk_index=chunk_index,
                            is_final=True,
                            finish_reason=event.delta.stop_reason,
                            usage=self._parse_usage(event.usage) if event.usage else None,
                            raw_chunk=event.model_dump(),
                        )

        except Exception as e:
            self.health_status.record_failure(str(e))
            raise self._create_error(str(e))

    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Prepare messages for Anthropic API"""
        messages = []

        # Add system message first
        if request.system_prompt:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"SYSTEM: {request.system_prompt}"}],
            })

        # Convert messages to Anthropic format
        for msg in request.messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            messages.append({
                "role": msg["role"],
                "content": content,
            })

        return messages

    def _prepare_tools(self, request: LLMRequest) -> Optional[List[Dict[str, Any]]]:
        """Prepare tools for Anthropic API"""
        if not request.functions:
            return None

        # Convert OpenAI-style functions to Anthropic tools
        tools = []
        for func in request.functions:
            tool = {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            }
            tools.append(tool)

        return tools

    def _parse_response(self, response, model: str, latency_ms: float) -> LLMResponse:
        """Parse Anthropic response into standard format"""
        content_blocks = response.content
        content = ""
        function_calls = []

        for block in content_blocks:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                function_calls.append({
                    "name": block.name,
                    "input": block.input,
                })
                content += f"[Tool call: {block.name}]"

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            content=content.strip(),
            model=model,
            usage=usage,
            finish_reason=response.stop_reason or "stop",
            provider=self.provider_name,
            request_id=response.id,
            raw_response=response.model_dump(),
            function_call=function_calls[0] if function_calls else None,
        )

    def _parse_usage(self, usage) -> TokenUsage:
        """Parse usage from streaming response"""
        return TokenUsage(
            input_tokens=getattr(usage, 'input_tokens', 0),
            output_tokens=getattr(usage, 'output_tokens', 0),
            total_tokens=getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0),
        )

    def _handle_api_error(self, error, model: str) -> None:
        """Handle Anthropic API errors"""
        status_code = error.status_code
        response_body = error.response.text if error.response else None

        if status_code == 401:
            raise AuthenticationError(self.provider_name)
        elif status_code == 429:
            retry_after = None
            if error.response and "retry-after" in error.response.headers:
                try:
                    retry_after = float(error.response.headers["retry-after"])
                except (ValueError, TypeError):
                    pass
            raise RateLimitError(
                message=f"Rate limit exceeded for Anthropic: {error.message}",
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
                message=f"Anthropic API error: {error.message}",
                provider=self.provider_name,
                status_code=status_code,
                response_body=response_body,
            )

    async def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's tokenizer"""
        # Anthropic doesn't have a public token counter
        # Fallback to approximate count
        return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models"""
        # Return known models since API doesn't list them
        return [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-3-20250514",
        ]

    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            await self._client.messages.create(
                model=self._default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False


# Alias for backward compatibility
AnthropicClaude = AnthropicProvider
