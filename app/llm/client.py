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
    """Get provider class by name (lazy loaded)"""
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
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
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
        """Add a message to the conversation"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "system_prompt": self.system_prompt,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            system_prompt=data.get("system_prompt"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )

    def get_message_history(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get message history for API calls"""
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
        """Initialize the agent and all providers"""
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
        """Shutdown all providers and cleanup"""
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
        Generate a response to a prompt

        Args:
            prompt: The user prompt
            conversation_id: Optional conversation ID for context
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with content and usage
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
        """Generate with specific provider"""
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
        Stream response as SSE events

        Args:
            prompt: The user prompt
            conversation_id: Optional conversation ID for context
            **kwargs: Additional arguments

        Yields:
            SSE formatted event strings
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
        """Create a new conversation and return its ID"""
        conversation = Conversation(
            system_prompt=system_prompt or self.agent_config.system_prompt,
            metadata=metadata or {},
        )
        self._conversations[conversation.id] = conversation
        return conversation.id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self._conversations.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
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
        Send a message in a conversation

        Args:
            message: The user message
            conversation_id: Conversation ID (creates new if not found)
            create_if_missing: Create conversation if not found
            **kwargs: Additional arguments

        Returns:
            LLMResponse
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
        """Stream a chat response"""
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
        """Build an LLMRequest from parameters"""
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
        """Switch the active provider at runtime"""
        if provider not in self._providers:
            logger.warning(f"Provider {provider} not available")
            return False

        self.agent_config.provider = provider
        if model:
            self.agent_config.model = model

        logger.info(f"Switched to provider: {provider}")
        return True

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self._providers.keys())

    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for name, provider in self._providers.items():
            status[name] = provider.health_status.to_dict()
        return status

    # ==================== Usage & Metrics ====================

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        if self._token_tracker:
            return self._token_tracker.get_stats()
        return {}

    def get_usage_summary(self, days: int = 1) -> Dict[str, Any]:
        """Get usage summary"""
        if self._token_tracker:
            return self._token_tracker.get_summary(days).to_dict()
        return {}

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        if self._router:
            return self._router.get_metrics()
        return {}

    # ==================== IPC Interface ====================

    def to_ipc_message(self, response: LLMResponse) -> str:
        """Convert response to IPC-friendly format"""
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
        """Parse IPC message"""
        return json.loads(message)


async def create_agent(
    config: Optional[AgentConfig] = None,
    config_path: Optional[str] = None,
) -> LLMAgent:
    """
    Factory function to create and initialize an LLM agent

    Usage:
        agent = await create_agent()
        response = await agent.generate("Hello!")
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
    Convenience function to stream a response

    Usage:
        response = await stream_response("Tell me a story", callback=print)
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
    """Handle IPC request from C++ orchestrator"""
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
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--ipc":
        # IPC mode - read from stdin, write to stdout
        import asyncio

        async def ipc_loop():
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
