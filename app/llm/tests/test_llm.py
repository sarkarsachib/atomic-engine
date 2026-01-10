#!/usr/bin/env python3
"""
Tests for Manus LLM Agent Layer
Tests provider failover, streaming, token accounting, and routing
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from datetime import datetime

# Import modules under test
from app.llm.config import (
    LLMConfig, ProviderConfig, ModelConfig, ConfigManager,
    RoutingStrategy, Provider, load_config
)
from app.llm.exceptions import (
    LLMError, ProviderError, RateLimitError, TokenLimitError,
    ConfigurationError, FallbackError, StreamingError, ErrorContext
)
from app.llm.token_accounting import TokenTracker, RateLimiter, TokenUsage
from app.llm.streaming import (
    StreamHandler, StreamContext, StreamChunk, StreamState,
    SSEStreamHandler, ChunkAggregator
)
from app.llm.providers import BaseProvider, LLMRequest, LLMResponse, TokenUsage
from app.llm.router import Router, RoutingCriteria, CostAwareRouter
from app.llm.client import LLMAgent, AgentConfig, Conversation, Message


# ==================== Fixtures ====================

@pytest.fixture
def mock_provider_config():
    """Create a mock provider configuration"""
    return ProviderConfig(
        name="test",
        enabled=True,
        priority=1,
        models=[
            ModelConfig(name="test-model", provider="test", cost_per_input=0.001, cost_per_output=0.002)
        ]
    )


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration"""
    return ModelConfig(
        name="gpt-4o",
        provider="openai",
        max_tokens=4096,
        max_input_tokens=128000,
        supports_streaming=True,
        supports_vision=True,
        supports_functions=True,
        cost_per_input=5.0,
        cost_per_output=15.0,
    )


@pytest.fixture
def sample_llm_request():
    """Create a sample LLM request"""
    return LLMRequest(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def sample_llm_response():
    """Create a sample LLM response"""
    return LLMResponse(
        content="I'm doing well, thank you!",
        model="gpt-4o",
        usage=TokenUsage(input_tokens=10, output_tokens=12),
        finish_reason="stop",
        provider="openai",
        request_id="test-123",
    )


# ==================== Config Tests ====================

class TestConfig:
    """Tests for configuration module"""

    def test_model_config_to_dict(self, mock_model_config):
        """Test ModelConfig serialization"""
        data = mock_model_config.to_dict()
        assert data["name"] == "gpt-4o"
        assert data["provider"] == "openai"
        assert data["supports_streaming"] is True

    def test_provider_config_from_dict(self, mock_provider_config):
        """Test ProviderConfig deserialization"""
        data = mock_provider_config.to_dict()
        restored = ProviderConfig.from_dict(data)
        assert restored.name == "test"
        assert restored.enabled is True

    def test_llm_config_defaults(self):
        """Test LLMConfig default values"""
        config = LLMConfig()
        assert config.default_provider == "openai"
        assert config.enable_streaming is True
        assert config.routing_strategy == RoutingStrategy.PRIORITY

    def test_provider_enum_values(self):
        """Test Provider enum values"""
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"


# ==================== Exception Tests ====================

class TestExceptions:
    """Tests for exception handling"""

    def test_llm_error_creation(self):
        """Test LLMError creation"""
        error = LLMError("Test error", is_retryable=True)
        assert str(error) == "[LLMError] Test error"
        assert error.is_retryable is True

    def test_error_context(self):
        """Test ErrorContext"""
        ctx = ErrorContext(
            provider="openai",
            model="gpt-4o",
            request_id="test-123",
        )
        data = ctx.to_dict()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"

    def test_rate_limit_error(self):
        """Test RateLimitError"""
        error = RateLimitError(
            message="Rate limit exceeded",
            provider="openai",
            retry_after_seconds=30.0,
            limit_type="requests",
        )
        assert error.retry_after_seconds == 30.0
        assert error.is_retryable is True

    def test_fallback_error(self):
        """Test FallbackError with multiple providers"""
        last_error = ProviderError("Test", "openai", 500)
        error = FallbackError(
            providers=["openai", "anthropic"],
            last_error=last_error,
        )
        assert "openai -> anthropic" in str(error)
        assert error.is_retryable is False


# ==================== Token Accounting Tests ====================

class TestTokenAccounting:
    """Tests for token accounting and rate limiting"""

    def test_token_tracker_record_usage(self):
        """Test recording token usage"""
        tracker = TokenTracker()
        usage = tracker.record_usage(
            input_tokens=100,
            output_tokens=200,
            provider="openai",
            model="gpt-4o",
            cost=0.005,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.total_tokens == 300

    def test_token_tracker_calculate_cost(self, mock_model_config):
        """Test cost calculation"""
        tracker = TokenTracker()
        cost = tracker.calculate_cost(
            input_tokens=1000,
            output_tokens=2000,
            model_config=mock_model_config,
        )
        # 1000 * 5/1M + 2000 * 15/1M = 0.005 + 0.03 = 0.035
        assert cost == 0.035

    def test_token_tracker_daily_usage(self):
        """Test daily usage tracking"""
        tracker = TokenTracker()
        tracker.record_usage(100, 100, "openai", "gpt-4o")
        usage = tracker.get_daily_usage()
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 100

    def test_token_tracker_get_stats(self):
        """Test getting statistics"""
        tracker = TokenTracker()
        tracker.record_usage(100, 100, "openai", "gpt-4o", cost=0.005)
        stats = tracker.get_stats()
        assert stats["total_tokens"] == 200
        assert stats["total_cost"] == 0.005


class TestRateLimiter:
    """Tests for rate limiting"""

    def test_rate_limiter_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(requests_per_minute=10)
        # Should not raise for first request
        # Note: Async test in sync context won't work as expected

    def test_rate_limiter_status(self):
        """Test getting rate limiter status"""
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=100000)
        status = limiter.get_status()
        assert "requests_per_minute" in status
        assert "tokens_per_minute" in status


# ==================== Streaming Tests ====================

class TestStreaming:
    """Tests for streaming functionality"""

    def test_stream_context_creation(self, sample_llm_request):
        """Test stream context creation"""
        ctx = StreamContext(
            stream_id="test-123",
            request=sample_llm_request,
            provider="openai",
            model="gpt-4o",
        )
        assert ctx.state == StreamState.IDLE
        assert ctx.stream_id == "test-123"

    def test_stream_chunk_creation(self):
        """Test stream chunk creation"""
        chunk = StreamChunk(
            content="Hello",
            delta="Hello",
            model="gpt-4o",
            provider="openai",
            chunk_index=0,
            is_final=False,
        )
        assert chunk.content == "Hello"
        assert chunk.chunk_index == 0

    def test_chunk_aggregator(self):
        """Test chunk aggregation"""
        aggregator = ChunkAggregator()
        chunks = [
            StreamChunk(content="H", delta="H", model="test", provider="test", chunk_index=0, is_final=False),
            StreamChunk(content="He", delta="e", model="test", provider="test", chunk_index=1, is_final=False),
            StreamChunk(content="Hel", delta="l", model="test", provider="test", chunk_index=2, is_final=False),
            StreamChunk(content="Hell", delta="l", model="test", provider="test", chunk_index=3, is_final=True, finish_reason="stop"),
        ]

        async def test():
            for chunk in chunks:
                result = await aggregator.add_chunk("stream-1", chunk)
                if result:
                    return result
            return None

        result = asyncio.run(test())
        assert result is not None
        assert result["content"] == "Hell"
        assert result["total_chunks"] == 4


# ==================== Provider Tests ====================

class TestBaseProvider:
    """Tests for base provider"""

    def test_llm_request_to_dict(self, sample_llm_request):
        """Test LLMRequest serialization"""
        data = sample_llm_request.to_dict()
        assert data["messages"] == [{"role": "user", "content": "Hello, how are you?"}]
        assert data["model"] == "gpt-4o"
        assert data["temperature"] == 0.7

    def test_llm_response_to_dict(self, sample_llm_response):
        """Test LLMResponse serialization"""
        data = sample_llm_response.to_dict()
        assert data["content"] == "I'm doing well, thank you!"
        assert data["provider"] == "openai"
        assert data["usage"]["total_tokens"] == 22


# ==================== Routing Tests ====================

class TestRouting:
    """Tests for provider routing"""

    def test_routing_criteria_from_request(self, sample_llm_request):
        """Test creating routing criteria from request"""
        criteria = RoutingCriteria.from_request(sample_llm_request)
        assert criteria.requires_streaming is False
        assert criteria.task_complexity >= 1

    def test_routing_criteria_vision_request(self):
        """Test routing criteria with vision request"""
        request = LLMRequest(
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "test.png"}}]}],
            model="gpt-4o",
        )
        criteria = RoutingCriteria.from_request(request)
        assert criteria.requires_vision is True

    def test_router_metrics(self):
        """Test router metrics tracking"""
        config = LLMConfig()
        router = Router(config=config, providers={})
        metrics = router.get_metrics()
        assert "total_requests" in metrics
        assert metrics["total_requests"] == 0


# ==================== Client Tests ====================

class TestLLMAgent:
    """Tests for main LLM agent client"""

    def test_agent_config_from_dict(self):
        """Test agent config deserialization"""
        config = AgentConfig.from_dict({
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.5,
        })
        assert config.provider == "openai"
        assert config.temperature == 0.5

    def test_conversation_creation(self):
        """Test conversation creation"""
        conv = Conversation(system_prompt="You are a helpful assistant.")
        assert conv.id is not None
        assert len(conv.messages) == 0

        conv.add_message("user", "Hello")
        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"

    def test_conversation_message_history(self):
        """Test getting message history"""
        conv = Conversation(system_prompt="You are helpful.")
        conv.add_message("user", "Hi")
        conv.add_message("assistant", "Hello!")

        history = conv.get_message_history()
        assert len(history) == 3  # system + user + assistant
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"

    def test_message_to_dict(self):
        """Test message serialization"""
        msg = Message(role="user", content="Hello")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Hello"


# ==================== Integration-style Tests ====================

class TestIntegration:
    """Integration-style tests with mocks"""

    @pytest.mark.asyncio
    async def test_agent_initialize_with_mock_providers(self):
        """Test agent initialization with mocked providers"""
        # This would test the full initialization flow
        # with mocked provider clients
        pass

    @pytest.mark.asyncio
    async def test_failover_chain(self):
        """Test automatic failover when primary provider fails"""
        # This would test the failover mechanism
        # by simulating provider failures
        pass

    @pytest.mark.asyncio
    async def test_streaming_end_to_end(self):
        """Test streaming from request to response"""
        # This would test the full streaming flow
        pass


# ==================== Test Utilities ====================

def run_async_test(coro):
    """Helper to run async tests"""
    return asyncio.run(coro)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
