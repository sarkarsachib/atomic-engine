#!/usr/bin/env python3
"""
Example Usage Scripts for Manus LLM Agent Layer
Demonstrates various usage patterns
"""

import asyncio
import json
from app.llm import (
    create_agent, LLMAgent, AgentConfig,
    load_config, LLMConfig,
    Router, RoutingStrategy,
)


async def basic_generation():
    """Basic text generation example"""
    print("=" * 60)
    print("Basic Generation Example")
    print("=" * 60)

    # Create agent with default config
    agent = await create_agent()

    # Simple generation
    response = await agent.generate("What is the capital of France?")
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Usage: {response.usage.to_dict()}")
    print()

    await agent.shutdown()


async def streaming_example():
    """Streaming response example"""
    print("=" * 60)
    print("Streaming Example")
    print("=" * 60)

    agent = await create_agent()

    print("Generating story (streaming):")
    print("-" * 40)

    full_response = ""
    async for chunk in agent.stream("Tell me a short story about a robot:"):
        print(chunk.delta, end="", flush=True)
        full_response = chunk.content

    print("\n" + "-" * 40)
    print(f"\nTotal chunks received")
    print()

    await agent.shutdown()


async def conversation_example():
    """Conversation/chat example"""
    print("=" * 60)
    print("Conversation Example")
    print("=" * 60)

    agent = await create_agent()

    # Create conversation with system prompt
    conv_id = agent.create_conversation(
        system_prompt="You are a helpful coding assistant."
    )

    # Chat messages
    messages = [
        "What is Python?",
        "How do I create a list?",
        "What's a list comprehension?",
    ]

    for msg in messages:
        print(f"You: {msg}")
        response = await agent.chat(msg, conversation_id=conv_id)
        print(f"Assistant: {response.content[:200]}...")
        print()

    # Show conversation history
    conv = agent.get_conversation(conv_id)
    print(f"Conversation has {len(conv.messages)} messages")
    print()

    await agent.shutdown()


async def provider_management_example():
    """Provider switching example"""
    print("=" * 60)
    print("Provider Management Example")
    print("=" * 60)

    agent = await create_agent()

    # List providers
    providers = agent.get_available_providers()
    print(f"Available providers: {providers}")

    # Get provider status
    status = await agent.get_provider_status()
    print("\nProvider status:")
    for name, data in status.items():
        print(f"  {name}: healthy={data.get('is_healthy', 'unknown')}")

    # Switch provider (if available)
    if len(providers) > 1:
        current = providers[0]
        next_provider = providers[1] if len(providers) > 1 else providers[0]
        await agent.switch_provider(next_provider)
        print(f"\nSwitched from {current} to {next_provider}")

    print()

    await agent.shutdown()


async def usage_tracking_example():
    """Usage tracking example"""
    print("=" * 60)
    print("Usage Tracking Example")
    print("=" * 60)

    agent = await create_agent()

    # Generate some content
    await agent.generate("Tell me about machine learning")
    await agent.generate("What is deep learning?")

    # Get stats
    stats = agent.get_usage_stats()
    print(f"Usage statistics: {json.dumps(stats, indent=2)}")

    # Get summary
    summary = agent.get_usage_summary(days=1)
    print(f"\nDaily summary: {json.dumps(summary, indent=2)}")

    # Get routing metrics
    metrics = agent.get_routing_metrics()
    print(f"\nRouting metrics: {json.dumps(metrics, indent=2)}")
    print()

    await agent.shutdown()


async def routing_example():
    """Custom routing example"""
    print("=" * 60)
    print("Custom Routing Example")
    print("=" * 60)

    # Load custom config with specific routing strategy
    config = LLMConfig(
        default_provider="openai",
        fallback_providers=["anthropic", "bedrock"],
        routing_strategy=RoutingStrategy.COST_AWARE,
    )

    agent = await create_agent()

    # Generate with cost-aware routing
    response = await agent.generate(
        "Explain quantum computing simply",
        provider="openai",  # Primary
    )
    print(f"Response (cost-aware): {response.content[:100]}...")
    print(f"Provider used: {response.provider}")
    print()

    await agent.shutdown()


async def function_calling_example():
    """Function calling example (conceptual)"""
    print("=" * 60)
    print("Function Calling Example")
    print("=" * 60)

    agent = await create_agent()

    # Define a function for the model to call
    def get_weather(city: str) -> str:
        """Get weather for a city"""
        return f"Weather in {city}: 72°F, sunny"

    # This would require proper function calling setup
    # For now, just show the concept
    print("Function calling requires tool definitions...")
    print("Example function: get_weather(city: str)")
    print()

    await agent.shutdown()


async def sse_streaming_example():
    """SSE (Server-Sent Events) streaming example"""
    print("=" * 60)
    print("SSE Streaming Example")
    print("=" * 60)

    agent = await create_agent()

    print("SSE events:")
    print("-" * 40)

    async for event in agent.stream_sse("Count to 5:"):
        # SSE events are formatted strings
        if event.startswith("data:"):
            print(event.strip()[:100])

    print("-" * 40)
    print()

    await agent.shutdown()


async def custom_config_example():
    """Custom configuration example"""
    print("=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)

    # Create custom agent config
    config = AgentConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.5,
        max_tokens=500,
        system_prompt="You are a poetic assistant.",
        streaming=True,
        enable_fallback=True,
    )

    agent = await create_agent(config=config)

    response = await agent.generate("Write a haiku about code")
    print(f"Response: {response.content}")
    print()

    await agent.shutdown()


async def ipc_example():
    """IPC communication example (for C++ integration)"""
    print("=" * 60)
    print("IPC Communication Example")
    print("=" * 60)

    # In a real scenario, the agent runs as a subprocess
    # and communicates via JSON-RPC over stdin/stdout

    # Example IPC message format:
    request = {
        "command": "generate",
        "params": {
            "prompt": "Hello from C++!",
            "temperature": 0.7,
            "max_tokens": 100,
        }
    }

    print("Example IPC Request:")
    print(json.dumps(request, indent=2))

    # Example response format:
    response = {
        "status": "success",
        "response": {
            "content": "Hello! How can I help you?",
            "model": "gpt-4o",
            "provider": "openai",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8,
                "total_tokens": 18,
            }
        }
    }

    print("\nExample IPC Response:")
    print(json.dumps(response, indent=2))
    print()


async def main():
    """Run all examples"""
    print("\nManus LLM Agent Layer - Usage Examples")
    print("=" * 60)
    print()

    examples = [
        ("Basic Generation", basic_generation),
        ("Streaming", streaming_example),
        ("Conversation", conversation_example),
        ("Provider Management", provider_management_example),
        ("Usage Tracking", usage_tracking_example),
        ("Custom Routing", routing_example),
        ("Function Calling", function_calling_example),
        ("SSE Streaming", sse_streaming_example),
        ("Custom Configuration", custom_config_example),
        ("IPC Communication", ipc_example),
    ]

    for name, coro in examples:
        try:
            await coro()
        except Exception as e:
            print(f"Error in {name}: {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
