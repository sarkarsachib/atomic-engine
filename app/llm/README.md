# Manus LLM Agent Layer for Atomic Engine

A comprehensive Python LLM abstraction layer with multi-provider support, intelligent routing, streaming, and token accounting.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, AWS Bedrock, Ollama (local), Azure OpenAI
- **Intelligent Routing**: Cost-aware, priority-based, and capability-matching routing
- **Automatic Failover**: Seamless fallback when providers fail
- **Streaming Support**: Real-time streaming responses with SSE format
- **Token Accounting**: Track usage, costs, and rate limits
- **IPC-Ready**: Designed to be called from C++ via subprocess

## Quick Start

```python
from app.llm import create_agent, LLMAgent

async def main():
    # Create and initialize agent
    agent = await create_agent()
    
    # Generate response
    response = await agent.generate("Hello, how are you?")
    print(response.content)
    
    # Stream response
    async for chunk in agent.stream("Tell me a story"):
        print(chunk.delta, end="", flush=True)
    
    await agent.shutdown()

asyncio.run(main())
```

## Installation

```bash
# Install dependencies
pip install -r app/requirements.txt

# Run examples
python app/examples/usage_examples.py
```

## Architecture

```
app/llm/
├── __init__.py           # Main module exports
├── client.py             # LLMAgent main client
├── config.py             # Configuration management
├── exceptions.py         # Custom exceptions
├── token_accounting.py   # Token tracking & rate limiting
├── streaming.py          # Streaming & SSE handling
├── router.py             # Intelligent provider routing
├── providers/
│   ├── __init__.py       # Base provider interface
│   ├── openai.py         # OpenAI provider
│   ├── anthropic.py      # Anthropic Claude provider
│   ├── bedrock.py        # AWS Bedrock provider
│   ├── ollama.py         # Ollama local provider
│   └── azure.py          # Azure OpenAI provider
└── tests/
    └── test_llm.py       # Test suite
```

## Configuration

Create a `config/llm.toml` file:

```toml
[general]
default_provider = "openai"
fallback_providers = ["anthropic", "bedrock"]
routing_strategy = "priority"
enable_streaming = true
enable_logging = true

[providers.openai]
enabled = true
priority = 1
api_key = "${OPENAI_API_KEY}"
models = [
    { name = "gpt-4o", cost_per_input = 5.0, cost_per_output = 15.0 },
    { name = "gpt-4o-mini", cost_per_input = 0.15, cost_per_output = 0.6 }
]

[providers.anthropic]
enabled = true
priority = 2
api_key = "${ANTHROPIC_API_KEY}"
models = [
    { name = "claude-sonnet-4-20250514", cost_per_input = 3.0, cost_per_output = 15.0 }
]
```

## IPC Integration

The agent can be called from C++ via JSON-RPC over stdin/stdout:

```python
# Send request
request = {
    "command": "generate",
    "params": {"prompt": "Hello!", "temperature": 0.7}
}
print(json.dumps(request))

# Receive response
response = json.loads(input())
print(response)
```

## API Reference

### LLMAgent

```python
agent = await create_agent(config)

# Generate response
response = await agent.generate(prompt, **kwargs)

# Stream response
async for chunk in agent.stream(prompt, **kwargs):
    print(chunk.delta)

# Chat with conversation
response = await agent.chat(message, conversation_id)

# Manage providers
providers = agent.get_available_providers()
await agent.switch_provider("anthropic")
```

### Routing

```python
# Configure routing strategy
config = LLMConfig(
    routing_strategy=RoutingStrategy.COST_AWARE,
    fallback_providers=["anthropic", "bedrock"]
)

# Custom criteria for provider selection
criteria = RoutingCriteria(
    requires_vision=True,
    requires_functions=False,
    prefer_cheap=False,
    task_complexity=8,
)
```

### Streaming

```python
# Get SSE events
async for event in agent.stream_sse(prompt):
    print(event)  # SSE formatted string

# Use stream handler directly
handler = StreamHandler()
context = await handler.create_stream(request, provider, model)
await handler.process_stream(provider, request, context)
```

## Testing

```bash
# Run tests
pytest app/llm/tests/ -v

# Run with coverage
pytest app/llm/tests/ --cov=app.llm
```

## License

MIT License
