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
        """
        Create an OllamaConfig populated from environment variables.
        
        Reads OLLAMA_BASE_URL (default "http://localhost:11434") for the provider base URL and OLLAMA_TIMEOUT (default "300") parsed as a float for the request timeout in seconds.
        
        Returns:
            OllamaConfig: Configuration with `base_url` and `timeout` set from the environment (defaults applied if unset).
        """
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
        """
        Initialize the OllamaProvider with configuration, validate HTTP client dependency, and prepare internal state.
        
        Parameters:
            config (Optional[ProviderConfig]): Optional provider-level configuration; used for base URL and timeout when corresponding kwargs are not provided.
            api_key (Optional[str]): Optional API key (not used by Ollama HTTP API but accepted for interface compatibility).
            **kwargs: Optional overrides:
                base_url (str): Ollama server base URL (default: from `config.api_base` or "http://localhost:11434").
                timeout (float): Request timeout seconds (default: from `config.timeout_seconds` or 300.0).
                max_retries (int): Number of request retries (default: 0).
        
        Raises:
            ImportError: If the `httpx` library is not available.
        
        Behavior:
            - Constructs an OllamaConfig from provided kwargs or `config`.
            - Initializes internal AsyncClient reference to None and sets the default model to "llama3.2".
            - Initializes model discovery cache and cache TTL (300 seconds).
        """
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
        """
        Default model identifier used when no model is specified for requests.
        
        Returns:
            The default model name.
        """
        return self._default_model

    async def initialize(self) -> None:
        """
        Initialize the internal HTTP client and verify connectivity to the Ollama API.
        
        Creates and configures the provider's internal httpx AsyncClient if one does not
        already exist, then performs a health check against the configured Ollama base
        URL. On failure, raises a ConnectionError containing provider and endpoint
        details.
         
        Raises:
            ConnectionError: If the client cannot be created or the health check fails.
        """
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
        """
        Close the provider's internal HTTP client and clear the client reference.
        
        If an internal httpx.AsyncClient exists, await its close and set the internal client attribute to None.
        """
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Ollama provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a single non-streaming completion from the configured Ollama model.
        
        Prepare and send a generation request using the provided LLMRequest, record request latency in health metrics, and return the parsed LLMResponse containing the generated content, token usage, model, and raw provider response.
        
        Parameters:
            request (LLMRequest): Input request containing messages, target model, and generation parameters.
        
        Returns:
            LLMResponse: The provider's parsed response including generated text, token usage, model identifier, and raw response payload.
        """
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
        """
        Stream model output for the given LLMRequest as incremental chunks.
        
        Yields:
            StreamChunk: incremental generation chunks in order. Each chunk contains:
                - content: accumulated text produced so far
                - delta: newly produced text in this chunk
                - model: model name used from the request
                - provider: provider identifier
                - chunk_index: 1-based index of the chunk
                - is_final: `True` for the final chunk, `False` otherwise
                - finish_reason: "stop" when `is_final` is `True`, otherwise `None`
                - usage: TokenUsage present only on the final chunk when evaluation counts are provided
                - raw_chunk: original decoded JSON chunk from the provider
        
        Behavior:
            - Initializes the HTTP client if needed.
            - Streams lines from the provider's /api/generate endpoint and parses each JSON line into a chunk.
            - Accumulates deltas into `content` and yields a StreamChunk for each parsed line.
            - On the final chunk (when `done` is true) constructs a TokenUsage from `prompt_eval_count` and `eval_count`, records request latency into health status, yields the final StreamChunk, and stops iteration.
        
        Raises:
            ProviderError: on underlying errors while streaming (wrapped via the provider's internal error creation).
        """
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
        """
        Construct the JSON body for the Ollama /api/generate request.
        
        Builds a payload containing the target model, a textual prompt (assembled from the request's system prompt and messages), the stream flag, and an options object derived from request fields. Options include temperature, num_predict (mapped from request.max_tokens), and stop; any option with value None is omitted.
        
        Parameters:
            request (LLMRequest): Request object whose fields are used to populate model, prompt, temperature, max_tokens, stop, and any message/system_prompt content.
            stream (bool): Whether the Ollama API should stream responses.
        
        Returns:
            dict: A dictionary with keys:
                - "model": model name string
                - "prompt": assembled prompt string
                - "stream": boolean stream flag
                - "options": dict of provided options (temperature, num_predict, stop) with None values removed
        """
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
        """
        Assembles a textual prompt for Ollama from the request's system prompt and messages.
        
        The prompt includes an optional leading "System: <system_prompt>" line, followed by each message rendered as:
        - "User: <content>" for messages with role "user",
        - "Assistant: <content>" for messages with role "assistant",
        - "<role>: <content>" for any other role.
        If a message's content is a list (multimodal), only items with type "text" are extracted and joined with newlines. The returned prompt always ends with the suffix "Assistant: " as a continuation cue.
        
        Parameters:
            request (LLMRequest): Request object containing an optional `system_prompt` and `messages` where each message is a dict with keys like `role` and `content`.
        
        Returns:
            str: The assembled prompt string ready to send to the Ollama API.
        """
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
        """
        Convert an Ollama API response into an LLMResponse with token usage.
        
        The returned LLMResponse's `content` is taken from response_body["response"] (or empty string if missing),
        and `usage` is built from `prompt_eval_count` and `eval_count` fields where:
        - input_tokens = prompt_eval_count (default 0)
        - output_tokens = eval_count (default 0)
        - total_tokens = input_tokens + output_tokens
        
        Returns:
            LLMResponse: Parsed response including content, model, TokenUsage, finish_reason "stop",
                         provider name, and the original raw_response dict.
        """
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
        """
        Map HTTP error responses from Ollama into provider-specific exceptions.
        
        Parameters:
            error: The HTTP error object from the Ollama request; must provide a response with a status code.
            model (str): The model name that was requested.
        
        Raises:
            ModelNotFoundError: If the response status code is 404; includes available models.
            ProviderError: For 500 or other non-404 status codes; includes the original error message and the status code.
        """
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
        """
        Estimate the number of tokens for a local model using a simple heuristic.
        
        Parameters:
            text (str): The input text to estimate token count for.
        
        Returns:
            int: Estimated token count (approximate; computed as len(text) // 4).
        """
        # Ollama doesn't expose token counting directly
        # Use approximate count based on words
        return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """
        Retrieve available Ollama model names, using an internal cache.
        
        Checks the provider's local cache and returns cached model names if still valid.
        If the cache is stale or missing, ensures the HTTP client is initialized and requests
        the model list from the Ollama `/api/tags` endpoint. On success the result is cached.
        If the request fails, returns a predefined fallback list of common model names.
        
        Returns:
            List[str]: A list of available model names (or a fallback list if the API request fails).
        """
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
        """
        Verify connectivity to the Ollama API.
        
        Ensures the provider's internal HTTP client is initialized, then performs a lightweight request to confirm the service is reachable.
        
        Returns:
            True if the Ollama API responds successfully, False otherwise.
        """
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
        """
        Stream pull progress events for a model download from the Ollama server.
        
        Parameters:
            model_name (str): Name of the model to pull.
        
        Returns:
            AsyncIterator[Dict[str, Any]]: Yields parsed JSON objects for each line of the server's streaming response; lines that are not valid JSON are ignored.
        """
        await self.initialize()

        async with self._client.stream("POST", "/api/pull", json={"name": model_name}) as response:
            async for line in response.aiter_lines():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from the local Ollama instance.
        
        Parameters:
            model_name (str): The name of the model to delete.
        
        Returns:
            bool: True if the model was deleted successfully, False otherwise.
        """
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