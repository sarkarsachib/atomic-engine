#!/usr/bin/env python3
"""
Ollama Provider Implementation
Supports local LLM models via Ollama API
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
    RateLimitError,
    TokenLimitError,
    ModelNotFoundError,
    ConnectionError,
    TimeoutError,
)

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx is required for Ollama. Install with: pip install httpx")


@dataclass
class OllamaConfig:
    """Ollama-specific configuration"""
    base_url: str = "http://localhost:11434"
    timeout: float = 300.0  # Longer timeout for local models
    max_retries: int = 0  # Local, usually no retries needed

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")),
        )


class OllamaProvider(BaseProvider):
    """Ollama Local LLM Provider"""

    provider_name = "ollama"
    supports_streaming = True
    supports_vision = True  # Depends on model
    supports_functions = False  # Depends on model

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config, api_key, **kwargs)

        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for Ollama. Install with: pip install httpx")

        self.ollama_config = OllamaConfig(
            base_url=kwargs.get("base_url") or (config.api_base if config else "http://localhost:11434"),
            timeout=kwargs.get("timeout") or (config.timeout_seconds if config else 300.0),
            max_retries=kwargs.get("max_retries") or 0,
        )

        self._client: Optional[httpx.AsyncClient] = None
        self._default_model = "llama3.2"

        # Cache for available models
        self._cached_models: Optional[List[str]] = None
        self._model_cache_time: float = 0
        self._model_cache_ttl: float = 300  # 5 minutes

    @property
    def default_model(self) -> str:
        """Default model for Ollama"""
        return self._default_model

    async def initialize(self) -> None:
        """Initialize Ollama client"""
        if self._client is not None:
            return

        try:
            self._client = httpx.AsyncClient(
                base_url=self.ollama_config.base_url,
                timeout=self.ollama_config.timeout,
                follow_redirects=True,
            )

            # Validate connection
            await self.health_check()
            logger.info(f"Ollama provider initialized successfully at {self.ollama_config.base_url}")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise ConnectionError(
                message=f"Failed to connect to Ollama at {self.ollama_config.base_url}: {e}",
                provider=self.provider_name,
                endpoint=self.ollama_config.base_url,
            )

    async def shutdown(self) -> None:
        """Shutdown Ollama client"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Ollama provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate non-streaming response"""
        await self.initialize()

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()

        try:
            # Prepare request body
            body = self._prepare_request_body(request)

            # Make API call
            response = await self._client.post("/api/generate", json=body)
            response.raise_for_status()

            response_body = response.json()

            latency_ms = (time.time() - start_time) * 1000
            self.health_status.record_success(latency_ms)

            return self._parse_response(response_body, request.model, latency_ms)

        except httpx.HTTPStatusError as e:
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
            body = self._prepare_request_body(request, stream=True)

            async with self._client.stream("POST", "/api/generate", json=body) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    chunk_index += 1

                    # Get the response text
                    delta = chunk.get("response", "")
                    accumulated_content += delta

                    is_final = chunk.get("done", False)
                    usage = None

                    if is_final and "eval_count" in chunk:
                        usage = TokenUsage(
                            input_tokens=chunk.get("prompt_eval_count", 0),
                            output_tokens=chunk.get("eval_count", 0),
                            total_tokens=chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0),
                        )
                        latency_ms = (time.time() - start_time) * 1000
                        self.health_status.record_success(latency_ms)

                    yield StreamChunk(
                        content=accumulated_content,
                        delta=delta,
                        model=request.model,
                        provider=self.provider_name,
                        chunk_index=chunk_index,
                        is_final=is_final,
                        finish_reason="stop" if is_final else None,
                        usage=usage,
                        raw_chunk=chunk,
                    )

                    if is_final:
                        break

        except Exception as e:
            self.health_status.record_failure(str(e))
            raise self._create_error(str(e))

    def _prepare_request_body(
        self,
        request: LLMRequest,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Prepare request body for Ollama API"""
        # Build prompt from messages
        prompt = self._build_prompt(request)

        body = {
            "model": request.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "stop": request.stop,
            },
        }

        # Remove None values
        body["options"] = {k: v for k, v in body["options"].items() if v is not None}

        return body

    def _build_prompt(self, request: LLMRequest) -> str:
        """Build prompt from messages"""
        prompt_parts = []

        # Add system prompt
        if request.system_prompt:
            prompt_parts.append(f"System: {request.system_prompt}\n")

        # Add messages
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "\n".join(text_parts)

            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            else:
                prompt_parts.append(f"{role}: {content}\n")

        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    def _parse_response(
        self,
        response_body: Dict[str, Any],
        model: str,
        latency_ms: float,
    ) -> LLMResponse:
        """Parse Ollama response into standard format"""
        content = response_body.get("response", "")

        usage = TokenUsage(
            input_tokens=response_body.get("prompt_eval_count", 0),
            output_tokens=response_body.get("eval_count", 0),
            total_tokens=response_body.get("prompt_eval_count", 0) + response_body.get("eval_count", 0),
        )

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason="stop",
            provider=self.provider_name,
            raw_response=response_body,
        )

    async def _handle_api_error(self, error, model: str) -> None:
        """Handle Ollama API errors"""
        status_code = error.response.status_code

        if status_code == 404:
            models = await self.get_available_models()
            raise ModelNotFoundError(
                provider=self.provider_name,
                model=model,
                available_models=models,
            )
        elif status_code == 500:
            raise ProviderError(
                message=f"Ollama server error: {str(error)}",
                provider=self.provider_name,
                status_code=500,
            )
        else:
            raise ProviderError(
                message=f"Ollama API error: {str(error)}",
                provider=self.provider_name,
                status_code=status_code,
            )

    async def count_tokens(self, text: str) -> int:
        """Count tokens (approximate for local models)"""
        # Ollama doesn't expose token counting directly
        # Use approximate count based on words
        return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        # Use cache
        current_time = time.time()
        if self._cached_models and (current_time - self._model_cache_time) < self._model_cache_ttl:
            return self._cached_models

        await self.initialize()

        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            self._cached_models = models
            self._model_cache_time = current_time

            return models

        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            # Return common models as fallback
            return [
                "llama3.2",
                "llama3.1",
                "llama3",
                "mistral",
                "mixtral",
                "qwen2.5",
                "qwen2",
                "deepseek-r1",
                "deepseek-coder",
                "codellama",
                "phi",
                "gemma",
                "gemma2",
            ]

    async def health_check(self) -> bool:
        """Check Ollama API health"""
        if not self._client:
            await self.initialize()

        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def pull_model(self, model_name: str) -> AsyncIterator[Dict[str, Any]]:
        """Pull a model from Ollama library"""
        await self.initialize()

        async with self._client.stream("POST", "/api/pull", json={"name": model_name}) as response:
            async for line in response.aiter_lines():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama"""
        await self.initialize()

        try:
            response = await self._client.delete("/api/delete", json={"name": model_name})
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False


# Alias for backward compatibility
OllamaClient = OllamaProvider
