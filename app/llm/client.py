#!/usr/bin/env python3
"""
Main LLM Agent Client
High-level interface for the Manus LLM Layer
Ready to be called from C++ via subprocess/IPC
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .config import LLMConfig, ConfigManager, load_config, RoutingStrategy
from .providers import LLMRequest, LLMResponse, StreamChunk, TokenUsage
from .router import Router, RoutingStrategy as RouterStrategy, RoutingCriteria, FallbackChain
from .streaming import StreamHandler, SSEStreamHandler, StreamContext
from .token_accounting import TokenTracker, RateLimiter
from .exceptions import LLMError, FallbackError, ConfigurationError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lazy load providers for internal use
def _get_provider_class(provider_name: str):
    """
    Get the provider class for a given provider name by lazily importing its module.
    
    Parameters:
        provider_name (str): Provider identifier, e.g. "openai", "anthropic", "bedrock", "ollama", or "azure".
    
    Returns:
        The provider class corresponding to the given provider name.
    
    Raises:
        ValueError: If the provider_name is not one of the supported providers.
    """
    if provider_name == "openai":
        from .providers.openai import OpenAIProvider
        return OpenAIProvider
    elif provider_name == "anthropic":
        from .providers.anthropic import AnthropicProvider
        return AnthropicProvider
    elif provider_name == "bedrock":
        from .providers.bedrock import BedrockProvider
        return BedrockProvider
    elif provider_name == "ollama":
        from .providers.ollama import OllamaProvider
        return OllamaProvider
    elif provider_name == "azure":
        from .providers.azure import AzureProvider
        return AzureProvider
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


@dataclass
class AgentConfig:
    """Configuration for an LLM agent"""
    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    streaming: bool = True
    routing_strategy: str = "priority"
    fallback_providers: List[str] = field(default_factory=list)
    timeout: int = 60
    enable_fallback: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """
        Create an AgentConfig from a dictionary of configuration values.
        
        Parameters:
            data (Dict[str, Any]): Mapping containing agent configuration. Recognized keys and defaults:
                - "provider": provider name (default: "openai")
                - "model": model name (default: None)
                - "temperature": sampling temperature (default: 0.7)
                - "max_tokens": maximum tokens (default: 4096)
                - "system_prompt": system prompt text (default: None)
                - "streaming": enable streaming (default: True)
                - "routing_strategy": routing strategy name (default: "priority")
                - "fallback_providers": list of fallback provider names (default: [])
                - "timeout": request timeout seconds (default: 60)
                - "enable_fallback": enable fallback behavior (default: True)
        
        Returns:
            AgentConfig: Configured AgentConfig instance populated from `data`.
        """
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            system_prompt=data.get("system_prompt"),
            streaming=data.get("streaming", True),
            routing_strategy=data.get("routing_strategy", "priority"),
            fallback_providers=data.get("fallback_providers", []),
            timeout=data.get("timeout", 60),
            enable_fallback=data.get("enable_fallback", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the AgentConfig into a plain dictionary suitable for storage or transmission.
        
        Returns:
            dict: Mapping containing the agent configuration fields: `provider`, `model`,
            `temperature`, `max_tokens`, `system_prompt`, `streaming`, `routing_strategy`,
            `fallback_providers`, `timeout`, and `enable_fallback`.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "streaming": self.streaming,
            "routing_strategy": self.routing_strategy,
            "fallback_providers": self.fallback_providers,
            "timeout": self.timeout,
            "enable_fallback": self.enable_fallback,
        }


@dataclass
class Message:
    """Chat message"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the Message to a dictionary suitable for JSON encoding.
        
        Returns:
            Dict[str, Any]: Dictionary with keys "role", "content", and "timestamp" where "timestamp" is an ISO 8601 string.
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create a Message from a dictionary representation.
        
        Parameters:
            data (Dict[str, Any]): Dictionary with keys "role" and "content". May include "timestamp" as an ISO 8601 string.
        
        Returns:
            Message: A Message instance with the parsed timestamp (uses current time if "timestamp" is not provided).
        
        Raises:
            KeyError: If "role" or "content" is missing from `data`.
            ValueError: If "timestamp" is provided but is not a valid ISO 8601 string.
        """
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )


@dataclass
class Conversation:
    """Conversation context"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> Message:
        """
        Append a new message with the given role and content to this conversation.
        
        Returns:
            The appended Message instance (with its timestamp set).
        """
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the conversation into a JSON-serializable dictionary.
        
        Returns:
            A dictionary with the following keys:
            - `id` (str): Conversation identifier.
            - `messages` (List[Dict[str, Any]]): Ordered list of messages as dictionaries.
            - `system_prompt` (Optional[str]): Conversation-level system prompt, or `None`.
            - `created_at` (str): ISO 8601 formatted creation timestamp.
            - `metadata` (Dict[str, Any]): Arbitrary metadata associated with the conversation.
        """
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "system_prompt": self.system_prompt,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """
        Constructs a Conversation instance from a dictionary representation.
        
        Parameters:
            data (Dict[str, Any]): Dictionary containing conversation fields. Expected keys:
                - "id" (str, optional): Conversation id; a new UUID is generated if missing.
                - "messages" (List[Dict], optional): List of message dicts parsed with Message.from_dict.
                - "system_prompt" (str, optional): Optional system prompt for the conversation.
                - "created_at" (str, optional): ISO-8601 timestamp; parsed into a datetime. If missing, current time is used.
                - "metadata" (Dict[str, Any], optional): Arbitrary metadata for the conversation.
        
        Returns:
            Conversation: A Conversation populated from the provided dictionary, with sensible defaults for missing fields.
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            system_prompt=data.get("system_prompt"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )

    def get_message_history(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Return ordered message history suitable for API requests.
        
        Parameters:
        	include_system (bool): If True and a system prompt exists, include a system message at the start of the history.
        
        Returns:
        	List[Dict[str, str]]: Ordered list of messages where each entry has keys `"role"` and `"content"`. When included, the system message appears first followed by conversation messages in chronological order.
        """
        messages = []
        if include_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend([{"role": m.role, "content": m.content} for m in self.messages])
        return messages


class LLMAgent:
    """
    Main LLM Agent class for Atomic Engine

    Usage:
        agent = LLMAgent()
        response = agent.generate("Hello, how are you?")
        async for chunk in agent.stream("Tell me a story"):
            print(chunk.delta)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize a new LLMAgent instance and set up its configuration and empty runtime state.
        
        Parameters:
            config (Optional[AgentConfig]): Agent configuration to use; when None, a default AgentConfig is created.
            llm_config (Optional[LLMConfig]): Optional low-level LLM configuration (provider/model-specific settings).
            config_path (Optional[str]): Optional filesystem path to the agent configuration file.
        """
        self.agent_config = config or AgentConfig()
        self._llm_config = llm_config
        self._config_path = config_path

        # Initialize components
        self._providers: Dict[str, BaseProvider] = {}
        self._router: Optional[Router] = None
        self._stream_handler: Optional[StreamHandler] = None
        self._token_tracker: Optional[TokenTracker] = None
        self._fallback_chain: Optional[FallbackChain] = None

        # Conversation state
        self._conversations: Dict[str, Conversation] = {}

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the agent's configuration, providers, and runtime components.
        
        This loads the LLM configuration if missing, creates the token tracker and rate limiter, initializes providers, and constructs the router, stream handler, and fallback chain. On success the agent is marked initialized.
        
        Raises:
            ConfigurationError: If initialization fails for any reason.
        """
        if self._initialized:
            return

        try:
            # Load configuration
            if self._llm_config is None:
                config_manager = ConfigManager(
                    [self._config_path] if self._config_path else None
                )
                self._llm_config = config_manager.load()

            # Initialize token tracker
            self._token_tracker = TokenTracker()

            # Initialize rate limiter
            self._rate_limiter = RateLimiter(
                requests_per_minute=self._llm_config.global_rate_limit_rpm,
            )

            # Initialize providers
            await self._initialize_providers()

            # Initialize router
            routing_strategy = RouterStrategy(self.agent_config.routing_strategy)
            self._router = Router(
                config=self._llm_config,
                providers=self._providers,
                token_tracker=self._token_tracker,
                rate_limiter=self._rate_limiter,
            )
            self._router.config.routing_strategy = routing_strategy

            # Initialize stream handler
            self._stream_handler = StreamHandler()

            # Initialize fallback chain
            self._fallback_chain = FallbackChain(self._router)

            self._initialized = True
            logger.info("LLM Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LLM Agent: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")

    async def _initialize_providers(self) -> None:
        """Initialize all configured providers"""
        for name, config in self._llm_config.providers.items():
            if not config.enabled:
                continue

            try:
                provider_class = _get_provider_class(name)
                provider = provider_class(config=config)

                await provider.initialize()
                self._providers[name] = provider
                logger.info(f"Initialized provider: {name}")

            except Exception as e:
                logger.warning(f"Failed to initialize provider {name}: {e}")
                continue

    async def shutdown(self) -> None:
        """
        Shuts down all configured providers and cleans up agent state.
        
        Shuts down each loaded provider, stops the token tracker if present, and marks the agent as uninitialized.
        """
        for provider in self._providers.values():
            try:
                await provider.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down {provider.provider_name}: {e}")

        if self._token_tracker:
            self._token_tracker.shutdown()

        self._initialized = False
        logger.info("LLM Agent shut down")

    # ==================== Generation Methods ====================

    async def generate(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate an LLM response for the given prompt, using the conversation context when a conversation_id is provided.
        
        Parameters:
            prompt: The user prompt to send to the model.
            conversation_id: Optional conversation identifier whose message history will be included in the request.
            **kwargs: Optional request overrides (e.g., `model`, `temperature`, `max_tokens`, `system_prompt`, and metadata).
        
        Returns:
            LLMResponse containing the model output content, response metadata, and usage information.
        """
        await self.initialize()

        # Build request
        request = await self._build_request(prompt, conversation_id, **kwargs)

        # Route and generate
        if self.agent_config.enable_fallback:
            try:
                response = await self._router.route_request(
                    request,
                    fallback_chain=self.agent_config.fallback_providers,
                )
            except FallbackError:
                # Try primary provider directly
                response = await self._generate_with_provider(request)
        else:
            response = await self._generate_with_provider(request)

        return response

    async def _generate_with_provider(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response using the provider specified in the request or the agent's default provider.
        
        Parameters:
            request (LLMRequest): The prepared LLM request containing prompt, model overrides, and metadata.
        
        Returns:
            LLMResponse: The response produced by the selected provider.
        
        Raises:
            ConfigurationError: If the resolved provider is not loaded or available.
        """
        provider_name = request.metadata.get("provider", self.agent_config.provider)
        provider = self._providers.get(provider_name)

        if not provider:
            raise ConfigurationError(f"Provider {provider_name} not available")

        model = request.model or provider.default_model

        # Update provider config
        self._llm_config.providers[provider_name].enabled = True

        return await provider.generate(request)

    async def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a response to a prompt

        Args:
            prompt: The user prompt
            conversation_id: Optional conversation ID for context
            **kwargs: Additional arguments

        Yields:
            StreamChunk with content deltas
        """
        await self.initialize()

        request = await self._build_request(prompt, conversation_id, **kwargs)

        # Route to provider
        provider_name = request.metadata.get("provider", self.agent_config.provider)
        provider = self._providers.get(provider_name)

        if not provider:
            raise ConfigurationError(f"Provider {provider_name} not available")

        # Create stream context
        context = await self._stream_handler.create_stream(
            request=request,
            provider=provider_name,
            model=request.model or provider.default_model,
        )

        # Process stream
        async for chunk in provider.stream(request):
            yield chunk

            if chunk.is_final:
                break

    async def stream_sse(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream the model's response as Server-Sent Events (SSE).
        
        Parameters:
            prompt (str): The user prompt to send to the model.
            conversation_id (Optional[str]): Optional conversation ID used to provide context for the request.
            **kwargs: Additional request overrides (e.g., model, temperature, max_tokens, provider, metadata).
        
        Yields:
            str: SSE-formatted event strings representing incremental stream events from the provider.
        """
        await self.initialize()

        sse_handler = SSEStreamHandler()

        request = await self._build_request(prompt, conversation_id, **kwargs)

        provider_name = request.metadata.get("provider", self.agent_config.provider)
        provider = self._providers.get(provider_name)

        if not provider:
            raise ConfigurationError(f"Provider {provider_name} not available")

        context = await sse_handler.create_stream(
            request=request,
            provider=provider_name,
            model=request.model or provider.default_model,
        )

        async for event in sse_handler.stream_to_sse(provider, request, context):
            yield event

    # ==================== Conversation Management ====================

    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new conversation, store it in the agent's conversation map, and return its identifier.
        
        If `system_prompt` is not provided, the agent's configured system prompt is used. `metadata`
        is attached to the conversation and defaults to an empty dict.
        
        Parameters:
            system_prompt (Optional[str]): Optional system prompt for the conversation.
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with the conversation.
        
        Returns:
            str: The newly created conversation's unique identifier.
        """
        conversation = Conversation(
            system_prompt=system_prompt or self.agent_config.system_prompt,
            metadata=metadata or {},
        )
        self._conversations[conversation.id] = conversation
        return conversation.id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by its identifier.
        
        Returns:
            The Conversation with the given id, or `None` if no conversation exists for that id.
        """
        return self._conversations.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Remove a conversation with the given ID from the agent's conversation store.
        
        Returns:
            `true` if the conversation existed and was deleted, `false` otherwise.
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        create_if_missing: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a message to a conversation, append the user's message and the assistant's reply to the conversation, and return the assistant's response.
        
        If no conversation_id is provided or the specified conversation is missing, a new conversation is created when create_if_missing is True; otherwise a ValueError is raised.
        
        Parameters:
            conversation_id (Optional[str]): ID of the conversation to use; if omitted or not found and create_if_missing is True, a new conversation is created.
            create_if_missing (bool): When True, create a new conversation if the given conversation_id does not exist.
        
        Returns:
            LLMResponse: The assistant's response including content and associated metadata.
        """
        await self.initialize()

        if not conversation_id:
            conversation_id = self.create_conversation()

        conversation = self._conversations.get(conversation_id)
        if not conversation:
            if create_if_missing:
                conversation_id = self.create_conversation()
                conversation = self._conversations[conversation_id]
            else:
                raise ValueError(f"Conversation {conversation_id} not found")

        # Add user message
        conversation.add_message("user", message)

        # Generate response
        response = await self.generate(
            prompt=message,
            conversation_id=conversation_id,
            **kwargs,
        )

        # Add assistant message
        conversation.add_message("assistant", response.content)

        return response

    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        create_if_missing: bool = True,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Start a streaming chat exchange: ensure the agent is initialized, add the user's message to the conversation, and yield incremental response chunks from the model.
        
        Parameters:
            message (str): The user message to send.
            conversation_id (Optional[str]): ID of the conversation to use; if omitted a new conversation is created.
            create_if_missing (bool): If True, create a new conversation when the provided ID does not exist; otherwise raise.
            **kwargs: Provider/model overrides and request options forwarded to the streaming call.
        
        Returns:
            AsyncIterator[StreamChunk]: An async iterator that yields incremental stream chunks from the model; the final chunk indicates completion.
        
        Raises:
            ValueError: If a conversation_id is provided but not found and create_if_missing is False.
        """
        await self.initialize()

        if not conversation_id:
            conversation_id = self.create_conversation()

        conversation = self._conversations.get(conversation_id)
        if not conversation:
            if create_if_missing:
                conversation_id = self.create_conversation()
                conversation = self._conversations[conversation_id]
            else:
                raise ValueError(f"Conversation {conversation_id} not found")

        conversation.add_message("user", message)

        async for chunk in self.stream(message, conversation_id, **kwargs):
            yield chunk

    # ==================== Helper Methods ====================

    async def _build_request(
        self,
        prompt: str,
        conversation_id: Optional[str],
        **kwargs,
    ) -> LLMRequest:
        """
        Constructs an LLMRequest using the given prompt, optional conversation context, and override options.
        
        If a conversation_id matches a stored conversation, the conversation's message history is used and its last user message is replaced with the provided prompt. Options passed via kwargs override corresponding agent defaults.
        
        Parameters:
            prompt (str): The user's prompt to include as the latest user message.
            conversation_id (Optional[str]): ID of a conversation whose history should be used as context, if present.
            **kwargs: Optional overrides and request-specific fields. Recognized keys include:
                - model (str): Model name to use.
                - temperature (float)
                - max_tokens (int)
                - streaming (bool)
                - system_prompt (str)
                - functions, function_call, stop
                - presence_penalty, frequency_penalty, user
                - provider (str)
                - metadata (dict): extra metadata to merge into the request metadata.
        
        Returns:
            LLMRequest: Request populated with messages, model, sampling parameters, system prompt, function settings, and metadata (including conversation_id and provider).
        """
        # Get conversation context
        messages = [{"role": "user", "content": prompt}]

        if conversation_id:
            conversation = self._conversations.get(conversation_id)
            if conversation:
                messages = conversation.get_message_history()
                messages[-1] = {"role": "user", "content": prompt}  # Update last message

        # Override with kwargs
        temperature = kwargs.get("temperature", self.agent_config.temperature)
        max_tokens = kwargs.get("max_tokens", self.agent_config.max_tokens)
        system_prompt = kwargs.get("system_prompt", self.agent_config.system_prompt)

        request = LLMRequest(
            messages=messages,
            model=kwargs.get("model") or self.agent_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=kwargs.get("streaming", self.agent_config.streaming),
            system_prompt=system_prompt,
            functions=kwargs.get("functions"),
            function_call=kwargs.get("function_call"),
            stop=kwargs.get("stop"),
            presence_penalty=kwargs.get("presence_penalty"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            user=kwargs.get("user"),
            metadata={
                "conversation_id": conversation_id,
                "provider": kwargs.get("provider", self.agent_config.provider),
                **kwargs.get("metadata", {}),
            },
        )

        return request

    # ==================== Provider Management ====================

    async def switch_provider(self, provider: str, model: Optional[str] = None) -> bool:
        """
        Change the agent's active provider and optionally set its default model.
        
        Parameters:
            model (Optional[str]): Optional model identifier to set as the agent's default for the new provider.
        
        Returns:
            `True` if the provider was available and the agent was switched, `False` otherwise.
        """
        if provider not in self._providers:
            logger.warning(f"Provider {provider} not available")
            return False

        self.agent_config.provider = provider
        if model:
            self.agent_config.model = model

        logger.info(f"Switched to provider: {provider}")
        return True

    def get_available_providers(self) -> List[str]:
        """
        Return the names of currently loaded providers.
        
        Returns:
            List[str]: A list of provider names that have been initialized and are available.
        """
        return list(self._providers.keys())

    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Collects the health status for each loaded provider.
        
        @returns Dict[str, Dict[str, Any]]: A mapping from provider name to its health status as a dictionary.
        """
        status = {}
        for name, provider in self._providers.items():
            status[name] = provider.health_status.to_dict()
        return status

    # ==================== Usage & Metrics ====================

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Retrieve current token usage statistics tracked by the agent.
        
        Returns:
            stats (Dict[str, Any]): A dictionary of usage statistics from the TokenTracker, or an empty dict if no token tracker is configured.
        """
        if self._token_tracker:
            return self._token_tracker.get_stats()
        return {}

    def get_usage_summary(self, days: int = 1) -> Dict[str, Any]:
        """
        Return a summary of token usage over the past `days` days.
        
        Parameters:
            days (int): Number of days to include in the summary (default 1).
        
        Returns:
            Dict[str, Any]: Aggregated usage metrics for the requested period, or an empty dict if no token tracker is configured.
        """
        if self._token_tracker:
            return self._token_tracker.get_summary(days).to_dict()
        return {}

    def get_routing_metrics(self) -> Dict[str, Any]:
        """
        Retrieve routing metrics collected by the agent's router.
        
        If no router is configured, returns an empty dict.
        
        Returns:
            Dict[str, Any]: Mapping of metric names to their values; empty if router is unavailable.
        """
        if self._router:
            return self._router.get_metrics()
        return {}

    # ==================== IPC Interface ====================

    def to_ipc_message(self, response: LLMResponse) -> str:
        """
        Serialize an LLMResponse into an IPC-friendly JSON string.
        
        The produced JSON contains the following keys: `type` (set to "response"), `content`, `model`, `provider`, `usage` (as a dict), `finish_reason`, `request_id`, and `timestamp` (ISO 8601).
        
        Returns:
            A JSON-formatted string containing the response fields for IPC consumption.
        """
        return json.dumps({
            "type": "response",
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage.to_dict(),
            "finish_reason": response.finish_reason,
            "request_id": response.request_id,
            "timestamp": response.timestamp.isoformat(),
        })

    @classmethod
    def from_ipc_message(cls, message: str) -> Dict[str, Any]:
        """
        Parse an IPC-formatted JSON string into a Python dictionary.
        
        Parameters:
            message (str): IPC message as a JSON-encoded string.
        
        Returns:
            dict: Dictionary representation of the parsed IPC message.
        
        Raises:
            json.JSONDecodeError: If the message is not valid JSON.
        """
        return json.loads(message)


async def create_agent(
    config: Optional[AgentConfig] = None,
    config_path: Optional[str] = None,
) -> LLMAgent:
    """
    Create and initialize an LLMAgent instance.
    
    Parameters:
        config (Optional[AgentConfig]): Optional agent configuration to use. If omitted, configuration may be loaded from `config_path` or defaults.
        config_path (Optional[str]): Optional filesystem path to a configuration file to load if no `config` is provided.
    
    Returns:
        LLMAgent: An initialized LLMAgent ready to handle requests.
    """
    agent = LLMAgent(config=config, config_path=config_path)
    await agent.initialize()
    return agent


async def stream_response(
    prompt: str,
    config: Optional[AgentConfig] = None,
    callback: Optional[callable] = None,
) -> str:
    """
    Stream a response to a prompt using a temporary LLMAgent and return the final accumulated content.
    
    Parameters:
        prompt (str): The user prompt to send to the agent.
        config (Optional[AgentConfig]): Optional agent configuration used to create the temporary agent.
        callback (Optional[callable]): Optional callable invoked with each streamed chunk's `delta` as it arrives.
    
    Returns:
        str: The final response content produced by the agent.
    """
    agent = await create_agent(config)
    full_response = ""

    async for chunk in agent.stream(prompt):
        if callback:
            callback(chunk.delta)
        full_response = chunk.content

    return full_response


# ==================== CLI / IPC Entry Point ====================

async def run_cli():
    """Run the agent as an interactive CLI"""
    print("Manus LLM Agent - Atomic Engine")
    print("Type 'quit' to exit, 'help' for commands\n")

    agent = await create_agent()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break

        if prompt.lower() == "help":
            print("\nCommands:")
            print("  quit - Exit the agent")
            print("  stats - Show usage statistics")
            print("  providers - Show available providers")
            print("  switch <provider> - Switch provider")
            continue

        if prompt.lower() == "stats":
            print(f"\nUsage: {agent.get_usage_stats()}")
            continue

        if prompt.lower() == "providers":
            print(f"\nProviders: {agent.get_available_providers()}")
            continue

        if prompt.lower().startswith("switch "):
            provider = prompt.split()[1]
            await agent.switch_provider(provider)
            print(f"Switched to {provider}")
            continue

        print("\nAgent: ", end="", flush=True)

        full_response = ""
        async for chunk in agent.stream(prompt):
            print(chunk.delta, end="", flush=True)
            full_response = chunk.content

        print("\n")

    await agent.shutdown()


async def handle_ipc_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatches IPC commands from a C++ orchestrator to an LLMAgent and returns a JSON-serializable result.
    
    Parameters:
        data (Dict[str, Any]): IPC payload containing at least a "command" string and an optional
            "params" dictionary. Supported commands: "generate", "stream", "chat", "providers",
            "switch", "stats", "health".
    
    Returns:
        Dict[str, Any]: A JSON-serializable dictionary with a "status" key ("success" or "error")
        and command-specific fields:
          - "response": serialized LLM response for "generate" and "chat"
          - "chunks": list of serialized stream chunks for "stream"
          - "providers": list of provider names for "providers" and "health"
          - "stats": usage statistics for "stats"
          - "message": error description when status is "error"
    """
    try:
        command = data.get("command")
        params = data.get("params", {})

        agent = await create_agent()

        if command == "generate":
            response = await agent.generate(**params)
            return {"status": "success", "response": response.to_dict()}

        elif command == "stream":
            chunks = []
            async for chunk in agent.stream(**params):
                chunks.append(chunk.to_dict())
            return {"status": "success", "chunks": chunks}

        elif command == "chat":
            response = await agent.chat(**params)
            return {"status": "success", "response": response.to_dict()}

        elif command == "providers":
            return {"status": "success", "providers": agent.get_available_providers()}

        elif command == "switch":
            success = await agent.switch_provider(params.get("provider"), params.get("model"))
            return {"status": "success" if success else "error"}

        elif command == "stats":
            return {"status": "success", "stats": agent.get_usage_stats()}

        elif command == "health":
            return {"status": "success", "providers": await agent.get_provider_status()}

        else:
            return {"status": "error", "message": f"Unknown command: {command}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    """
    Start the program in either IPC or interactive CLI mode.
    
    If the `--ipc` command-line flag is present, enter IPC mode: read JSON lines from standard input, pass each parsed object to `handle_ipc_request`, and write the handler's JSON-serializable result to standard output. If an input line is not valid JSON, write {"status": "error", "message": "Invalid JSON"} to standard output. Without `--ipc`, run the interactive CLI via `run_cli`.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--ipc":
        # IPC mode - read from stdin, write to stdout
        import asyncio

        async def ipc_loop():
            """
            Run a continuous IPC loop that reads JSON requests from stdin and writes JSON responses to stdout.
            
            Reads each line from standard input, parses it as JSON, passes the parsed object to the IPC request handler, and prints the handler's JSON-serializable result to standard output. If a line is not valid JSON, writes {"status": "error", "message": "Invalid JSON"} to stdout. Flushes stdout after each output.
            """
            for line in sys.stdin:
                try:
                    data = json.loads(line.strip())
                    result = await handle_ipc_request(data)
                    print(json.dumps(result))
                    sys.stdout.flush()
                except json.JSONDecodeError:
                    print(json.dumps({"status": "error", "message": "Invalid JSON"}))
                    sys.stdout.flush()

        asyncio.run(ipc_loop())
    else:
        # Interactive CLI mode
        asyncio.run(run_cli())


if __name__ == "__main__":
    main()