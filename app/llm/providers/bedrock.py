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
        """Default model for Bedrock"""
        return self._default_model

    def _get_model_id(self, model_name: str) -> str:
        """Get full Bedrock model ID from short name"""
        if model_name in self._model_ids:
            return self._model_ids[model_name]
        return model_name

    async def initialize(self) -> None:
        """Initialize Bedrock clients"""
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
        """Generate non-streaming response"""
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
        """Generate streaming response"""
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
        """Prepare request body for Bedrock API"""
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
        """Prepare messages for Anthropic models"""
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
        """Prepare prompt for Llama models"""
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
        """Prepare prompt for Mistral models"""
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
        """Parse Bedrock response into standard format"""
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
        """Handle Bedrock API errors"""
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
        """Count tokens (approximate)"""
        # Bedrock doesn't expose token counting
        return len(text) // 4

    async def get_available_models(self) -> List[str]:
        """Get list of available Bedrock models"""
        # Return known models
        return list(self._model_ids.keys())

    async def health_check(self) -> bool:
        """Check Bedrock API health"""
        try:
            # Try to list available models
            self._client.list_foundation_models()
            return True
        except Exception as e:
            logger.warning(f"Bedrock health check failed: {e}")
            return False


# Alias for backward compatibility
BedrockClient = BedrockProvider
