#!/usr/bin/env python3
"""
AWS Bedrock Provider Implementation
Supports Claude models via AWS Bedrock
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
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    logger.warning("boto3 is required for Bedrock. Install with: pip install boto3")


@dataclass
class BedrockConfig:
    """Bedrock-specific configuration"""
    region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    endpoint_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


class BedrockProvider(BaseProvider):
    """AWS Bedrock LLM Provider"""

    provider_name = "bedrock"
    supports_streaming = True
    supports_vision = True
    supports_functions = False  # Depends on model

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the BedrockProvider with configuration, credentials, and built-in model ID mappings.
        
        Constructs a BedrockConfig from provided kwargs, environment variables, or the given ProviderConfig; validates that boto3 is available; and sets up internal client placeholders, a default model ID, and a mapping of short model names to full Bedrock model IDs.
        
        Parameters:
            config (Optional[ProviderConfig]): Optional shared provider configuration to fall back to for region, api_base (endpoint), timeout_seconds, and retry_count.
            api_key (Optional[str]): Optional API key (kept for compatibility; Bedrock primarily uses AWS credentials).
            **kwargs: Provider-specific overrides. Recognized keys include:
                - region: AWS region to use (overrides config.region).
                - aws_access_key_id, aws_secret_access_key, aws_session_token: explicit AWS credentials (fallbacks to environment variables).
                - endpoint_url: custom Bedrock endpoint (overrides config.api_base).
                - timeout: request timeout in seconds (overrides config.timeout_seconds).
                - max_retries: maximum retry attempts (overrides config.retry_count).
        
        Raises:
            ImportError: If boto3 is not installed or available.
        """
        super().__init__(config, api_key, **kwargs)

        if not BEDROCK_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")

        self.bedrock_config = BedrockConfig(
            region=kwargs.get("region") or (config.region if config else "us-east-1"),
            aws_access_key_id=kwargs.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=kwargs.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=kwargs.get("aws_session_token") or os.getenv("AWS_SESSION_TOKEN"),
            endpoint_url=kwargs.get("endpoint_url") or (config.api_base if config else None),
            timeout=kwargs.get("timeout") or (config.timeout_seconds if config else 60.0),
            max_retries=kwargs.get("max_retries") or (config.retry_count if config else 3),
        )

        self._client = None
        self._runtime_client = None
        self._default_model = "anthropic.claude-3-5-sonnet-20241022"

        # Model ID mappings
        self._model_ids = {
            "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307",
            "claude-3-opus": "anthropic.claude-3-opus-20240307",
            "claude-2-1": "anthropic.claude-v2:1",
            "claude-2-0": "anthropic.claude-v2",
            "titan-text-premier": "amazon.titan-text-premier-v1:0",
            "titan-text-express": "amazon.titan-text-express-v1",
            "titan-text-lite": "amazon.titan-text-lite-v1",
            "llama-3-1-405b": "meta.llama3-1-405b-instruct-v1:0",
            "llama-3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
            "llama-3-1-8b": "meta.llama3-1-8b-instruct-v1:0",
            "llama-3-2-90b": "meta.llama3-2-90b-instruct-v1:0",
            "llama-3-2-11b": "meta.llama3-2-11b-instruct-v1:0",
            "mistral-large": "mistral.mistral-large-2407-v1:0",
            "mistral-small": "mistral.mistral-small-2407-v1:0",
            "mistral-7b": "mistral.mistral-7b-instruct-v0:1",
        }

    @property
    def default_model(self) -> str:
        """
        Get the provider's default model identifier.
        
        Returns:
            default_model (str): The default Bedrock model identifier used when none is specified.
        """
        return self._default_model

    def _get_model_id(self, model_name: str) -> str:
        """
        Resolve a short model name to its full Bedrock model identifier.
        
        Parameters:
            model_name (str): Short or full model name to resolve.
        
        Returns:
            model_id (str): The full Bedrock model ID if a mapping exists; otherwise returns the original `model_name`.
        """
        if model_name in self._model_ids:
            return self._model_ids[model_name]
        return model_name

    async def initialize(self) -> None:
        """
        Initialize the Bedrock boto3 clients and validate connectivity.
        
        Creates and assigns the internal Bedrock `bedrock` and `bedrock-runtime` boto3 clients and performs a health check; no action is taken if clients are already initialized.
        """
        if self._client is not None:
            return

        try:
            # Create boto3 clients
            self._client = boto3.client(
                service_name="bedrock",
                region_name=self.bedrock_config.region,
                aws_access_key_id=self.bedrock_config.aws_access_key_id,
                aws_secret_access_key=self.bedrock_config.aws_secret_access_key,
                aws_session_token=self.bedrock_config.aws_session_token,
                endpoint_url=self.bedrock_config.endpoint_url,
            )

            self._runtime_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.bedrock_config.region,
                aws_access_key_id=self.bedrock_config.aws_access_key_id,
                aws_secret_access_key=self.bedrock_config.aws_secret_access_key,
                aws_session_token=self.bedrock_config.aws_session_token,
                endpoint_url=self.bedrock_config.endpoint_url,
            )

            # Validate credentials
            await self.health_check()
            logger.info(f"Bedrock provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Bedrock provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Bedrock clients"""
        self._client = None
        self._runtime_client = None
        logger.info("Bedrock provider shut down")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Perform a single (non-streaming) inference call to Bedrock and return the parsed response.
        
        Parameters:
            request (LLMRequest): The request containing the model selection, messages, and any model-specific parameters used to build the Bedrock payload.
        
        Returns:
            LLMResponse: The provider-normalized response including generated text, token usage, finish reason, and latency metadata.
        """
        await self.initialize()

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        model_id = self._get_model_id(request.model)

        try:
            # Prepare body based on model provider
            body = self._prepare_request_body(request)

            # Make API call
            response = self._runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())

            latency_ms = (time.time() - start_time) * 1000
            self.health_status.record_success(latency_ms)

            return self._parse_response(response_body, request.model, latency_ms)

        except ClientError as e:
            self.health_status.record_failure(str(e))
            self._handle_api_error(e, request.model)
        except Exception as e:
            self.health_status.record_failure(str(e))
            raise self._create_error(str(e))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream model-generated text as incremental StreamChunk objects.
        
        Yields progressive chunks containing the latest delta and the accumulated content so far; a final chunk with is_final True and finish_reason "stop" is emitted when the model signals completion.
        
        Parameters:
            request (LLMRequest): Request specifying messages, model, and generation parameters; the request's model determines which Bedrock model and payload format are used.
        
        Returns:
            AsyncIterator[StreamChunk]: An async iterator that yields intermediate StreamChunk instances (is_final=False) for each delta and a final StreamChunk (is_final=True) when generation stops.
        """
        await self.initialize()

        model_config = self._get_model_config(request.model)
        request = self._apply_default_params(request, model_config)

        start_time = time.time()
        chunk_index = 0
        accumulated_content = ""
        model_id = self._get_model_id(request.model)

        try:
            body = self._prepare_request_body(request, stream=True)

            response = self._runtime_client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
            )

            for event in response.get("completionStream", []):
                chunk_index += 1

                if "delta" in event:
                    delta = event["delta"].get("text", "")
                    accumulated_content += delta

                    yield StreamChunk(
                        content=accumulated_content,
                        delta=delta,
                        model=request.model,
                        provider=self.provider_name,
                        chunk_index=chunk_index,
                        is_final=False,
                        raw_chunk=event,
                    )

                if event.get("stop"):
                    latency_ms = (time.time() - start_time) * 1000
                    self.health_status.record_success(latency_ms)

                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        model=request.model,
                        provider=self.provider_name,
                        chunk_index=chunk_index,
                        is_final=True,
                        finish_reason="stop",
                        raw_chunk=event,
                    )
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
        Builds a Bedrock-compatible request payload for the resolved model based on the given LLMRequest.
        
        Constructs provider-specific request bodies for Anthropic, Meta Llama, Amazon Titan, and Mistral families using fields from `request`. When `stream` is True, includes streaming flags where supported (Anthropic). Raises a ValueError if the resolved model ID is not supported.
        
        Parameters:
            request (LLMRequest): The normalized LLM request containing model, messages, and generation options.
            stream (bool): Whether the payload should enable streaming where supported.
        
        Returns:
            Dict[str, Any]: A dictionary formatted for the Bedrock API for the resolved model.
        """
        model_id = self._get_model_id(request.model)

        # Anthropic models
        if "anthropic" in model_id:
            messages = self._prepare_anthropic_messages(request)

            body = {
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
                "top_p": 0.999,
                "top_k": 250,
                "stop_sequences": request.stop or [],
            }

            if stream:
                body["stream"] = True

            return body

        # Meta Llama models
        elif "meta.llama" in model_id:
            prompt = self._prepare_llama_prompt(request)

            body = {
                "prompt": prompt,
                "max_gen_len": request.max_tokens or 4096,
                "temperature": request.temperature,
                "top_p": 0.9,
            }

            return body

        # Amazon Titan models
        elif "amazon.titan" in model_id:
            prompt = request.messages[0]["content"] if request.messages else ""

            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": request.max_tokens or 4096,
                    "temperature": request.temperature,
                    "topP": 0.9,
                    "stopSequences": request.stop or [],
                },
            }

            return body

        # Mistral models
        elif "mistral" in model_id:
            prompt = self._prepare_mistral_prompt(request)

            body = {
                "prompt": prompt,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
                "top_p": 0.9,
                "stop": request.stop or [],
            }

            return body

        raise ValueError(f"Unsupported model: {model_id}")

    def _prepare_anthropic_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Build a list of message dictionaries formatted for Anthropic-compatible Bedrock requests.
        
        The returned list includes a system prompt (when present) inserted as a user-role message whose `content` is a list containing a single text block. Each message from `request.messages` is appended with its `role` preserved; if a message's `content` is a string it is converted to a list of text blocks of the form `{"type": "text", "text": <text>}`.
        
        Parameters:
            request (LLMRequest): Request object containing `system_prompt` (optional) and `messages` (iterable of message dicts with `role` and `content`).
        
        Returns:
            List[Dict[str, Any]]: Messages ready to send to Anthropic models. Each dict contains `role` and `content` where `content` is a list of text block dicts.
        """
        messages = []

        # Add system message
        if request.system_prompt:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"{request.system_prompt}"}],
            })

        # Add user messages
        for msg in request.messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            messages.append({
                "role": msg["role"],
                "content": content,
            })

        return messages

    def _prepare_llama_prompt(self, request: LLMRequest) -> str:
        """
        Builds a prompt string formatted for Meta Llama models from the given LLMRequest.
        
        Encodes the optional system prompt and the sequence of messages into the tokenized form expected by Llama models:
        - system prompt is wrapped as `<|system|>... </s>`
        - user messages are wrapped as `<|user|>... </s>`
        - assistant messages are wrapped as `<|assistant|>... </s>`
        A trailing `<|assistant|>` token is appended to indicate where the model should continue.
        
        Parameters:
        	request (LLMRequest): Request containing an optional `system_prompt` and an ordered list of `messages` where each message is a mapping with `role` (`"user"` or `"assistant"`) and `content` (string).
        
        Returns:
        	prompt (str): The concatenated prompt string formatted for Meta Llama models.
        """
        prompt_parts = []

        if request.system_prompt:
            prompt_parts.append(f"<|system|>{request.system_prompt}</s>")

        for msg in request.messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<|user|>{msg['content']}</s>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|assistant|>{msg['content']}</s>")

        prompt_parts.append("<|assistant|>")
        return "".join(prompt_parts)

    def _prepare_mistral_prompt(self, request: LLMRequest) -> str:
        """
        Builds a single input prompt string formatted for Mistral-compatible models.
        
        Parameters:
            request (LLMRequest): The request containing optional system_prompt and a sequence of messages with "role" and "content".
        
        Returns:
            str: The assembled prompt where the system prompt and user messages are wrapped with "<s>[INST] ... [/INST]</s>" and assistant messages are appended followed by "</s>".
        """
        prompt_parts = []

        if request.system_prompt:
            prompt_parts.append(f"<s>[INST] {request.system_prompt} [/INST]</s>")

        for msg in request.messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<s>[INST] {msg['content']} [/INST]</s>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"{msg['content']}</s>")

        return "".join(prompt_parts)

    def _parse_response(
        self,
        response_body: Dict[str, Any],
        model: str,
        latency_ms: float,
    ) -> LLMResponse:
        """
        Convert a Bedrock model response into a standardized LLMResponse.
        
        Parameters:
            response_body (Dict[str, Any]): The raw response payload returned by Bedrock for the invoked model.
            model (str): The provider model name or alias used to resolve the Bedrock model ID.
            latency_ms (float): Measured request latency in milliseconds (provided for context).
        
        Returns:
            LLMResponse: A normalized response containing the aggregated text content, model, token usage, finish reason, provider name, and the original raw response.
        
        Raises:
            ValueError: If the response format does not match any supported model family.
        """
        model_id = self._get_model_id(model)

        # Anthropic response parsing
        if "anthropic" in model_id:
            content = ""
            for block in response_body.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")

            usage = TokenUsage(
                input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                output_tokens=response_body.get("usage", {}).get("output_tokens", 0),
                total_tokens=response_body.get("usage", {}).get("total_tokens", 0),
            )

            return LLMResponse(
                content=content.strip(),
                model=model,
                usage=usage,
                finish_reason=response_body.get("stop_reason") or "stop",
                provider=self.provider_name,
                raw_response=response_body,
            )

        # Llama response parsing
        elif "meta.llama" in model_id:
            content = response_body.get("generation", "")

            return LLMResponse(
                content=content.strip(),
                model=model,
                usage=TokenUsage(),
                finish_reason="stop",
                provider=self.provider_name,
                raw_response=response_body,
            )

        # Titan response parsing
        elif "amazon.titan" in model_id:
            content = response_body.get("results", [{}])[0].get("outputText", "")

            return LLMResponse(
                content=content.strip(),
                model=model,
                usage=TokenUsage(),
                finish_reason="stop",
                provider=self.provider_name,
                raw_response=response_body,
            )

        # Mistral response parsing
        elif "mistral" in model_id:
            content = response_body.get("outputs", [{}])[0].get("text", "")

            return LLMResponse(
                content=content.strip(),
                model=model,
                usage=TokenUsage(),
                finish_reason="stop",
                provider=self.provider_name,
                raw_response=response_body,
            )

        raise ValueError(f"Unknown model response format: {model_id}")

    def _handle_api_error(self, error, model: str) -> None:
        """
        Map Bedrock API error responses to provider-specific exceptions.
        
        Parameters:
            error: The caught boto3/botocore client error containing an AWS-style error response.
            model (str): The resolved Bedrock model identifier associated with the request.
        
        Raises:
            AuthenticationError: If the error code indicates access or credential issues.
            RateLimitError: If the error code indicates throttling or rate limits were exceeded.
            ProviderError: For validation errors (bad request) or other unspecified Bedrock API errors.
            ModelNotFoundError: If the specified model cannot be found.
        """
        status_code = error.response.get("Error", {}).get("Code", "")

        if status_code == "AccessDeniedException":
            raise AuthenticationError(
                self.provider_name,
                "AWS credentials",
            )
        elif status_code == "ThrottlingException":
            raise RateLimitError(
                message=f"Rate limit exceeded for Bedrock: {str(error)}",
                provider=self.provider_name,
                limit_type="requests",
            )
        elif status_code == "ValidationException":
            raise ProviderError(
                message=f"Bad request: {str(error)}",
                provider=self.provider_name,
                status_code=400,
            )
        elif status_code == "ResourceNotFoundException":
            raise ModelNotFoundError(
                provider=self.provider_name,
                model=model,
            )
        else:
            raise ProviderError(
                message=f"Bedrock API error: {str(error)}",
                provider=self.provider_name,
            )

    async def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text for Bedrock models using a simple heuristic.
        
        Returns:
            int: Estimated token count (approximate). This uses a rough heuristic of len(text) // 4.
        """
        # Bedrock doesn't expose token counting
        return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """
        Return the short names of Bedrock models known to this provider.
        
        Returns:
            List[str]: List of model short-name identifiers available from the provider.
        """
        # Return known models
        return list(self._model_ids.keys())

    async def health_check(self) -> bool:
        """
        Verify that the configured Bedrock client can respond to a foundation-models listing request.
        
        Attempts to call the Bedrock client's list_foundation_models method and returns whether the call succeeded.
        
        Returns:
            `true` if the Bedrock client responded successfully to a foundation-models listing request, `false` otherwise.
        """
        try:
            # Try to list available models
            self._client.list_foundation_models()
            return True
        except Exception as e:
            logger.warning(f"Bedrock health check failed: {e}")
            return False


# Alias for backward compatibility
BedrockClient = BedrockProvider