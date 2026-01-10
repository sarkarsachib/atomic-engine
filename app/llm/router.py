#!/usr/bin/env python3
"""
Intelligent Router for LLM Provider Selection
Implements cost-aware routing, failover, and capability matching
"""

import asyncio
import random
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .config import LLMConfig, ProviderConfig, ModelConfig, RoutingStrategy
from .providers import BaseProvider, LLMRequest, LLMResponse
from .streaming import StreamHandler, StreamContext, StreamChunk
from .token_accounting import TokenTracker, RateLimiter
from .exceptions import (
    LLMError,
    ProviderError,
    RateLimitError,
    FallbackError,
    RoutingError,
    HealthCheckError,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingCriteria:
    """Criteria for provider selection"""
    requires_vision: bool = False
    requires_functions: bool = False
    requires_streaming: bool = True
    min_context_length: int = 0
    max_cost_per_token: float = 0.01
    max_latency_ms: int = 5000
    prefer_cheap: bool = False
    prefer_fast: bool = False
    prefer_reliable: bool = True
    task_complexity: int = 5  # 1-10 scale
    estimated_tokens: int = 1000

    @classmethod
    def from_request(cls, request: LLMRequest, metadata: Optional[Dict[str, Any]] = None) -> "RoutingCriteria":
        """Create criteria from request"""
        metadata = metadata or request.metadata or {}

        # Check messages for vision requirements (images)
        requires_vision = any(
            isinstance(content, dict) and content.get("type") == "image_url"
            for msg in request.messages
            for content in (msg.get("content", []) if isinstance(msg.get("content"), list) else [msg.get("content")])
        )

        # Check for function calling
        requires_functions = request.functions is not None and len(request.functions) > 0

        # Determine complexity from request
        content_length = sum(len(str(msg.get("content", ""))) for msg in request.messages)
        task_complexity = min(10, max(1, content_length // 1000))

        return cls(
            requires_vision=requires_vision,
            requires_functions=requires_functions,
            requires_streaming=request.stream,
            min_context_length=request.max_tokens or 4096,
            task_complexity=task_complexity,
            estimated_tokens=content_length // 4 + (request.max_tokens or 1000),
        )


@dataclass
class ProviderScore:
    """Score for provider selection"""
    provider: str
    model: str
    total_score: float = 0.0
    capability_score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0
    reliability_score: float = 0.0
    available: bool = True
    health_status: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_score": self.total_score,
            "capability_score": self.capability_score,
            "cost_score": self.cost_score,
            "latency_score": self.latency_score,
            "reliability_score": self.reliability_score,
            "available": self.available,
            "health_status": self.health_status,
            "error": self.error,
        }


class Router:
    """Intelligent LLM provider router"""

    def __init__(
        self,
        config: LLMConfig,
        providers: Dict[str, BaseProvider],
        token_tracker: Optional[TokenTracker] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.config = config
        self.providers = providers
        self.token_tracker = token_tracker or TokenTracker()
        self.rate_limiter = rate_limiter or RateLimiter()

        # Health check cache
        self._health_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._health_cache_ttl = 300  # 5 minutes

        # Metrics
        self._routing_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_routes": 0,
            "provider_selection_counts": {},
            "failure_reasons": {},
        }

    async def select_provider(
        self,
        request: LLMRequest,
        criteria: Optional[RoutingCriteria] = None,
    ) -> Tuple[BaseProvider, str]:
        """Select the best provider for a request"""
        criteria = criteria or RoutingCriteria.from_request(request)

        # Get available providers
        available = await self._get_available_providers(criteria)

        if not available:
            raise RoutingError(
                message="No providers available matching criteria",
                strategy=self.config.routing_strategy.value,
            )

        # Score and rank providers
        scored = await self._score_providers(available, criteria)

        if not scored:
            raise RoutingError(
                message="No providers could satisfy requirements",
                strategy=self.config.routing_strategy.value,
            )

        # Select based on strategy
        selected = self._apply_strategy(scored, criteria)

        logger.info(
            f"Selected provider {selected.provider} with model {selected.model} "
            f"(score: {selected.total_score:.2f})"
        )

        self._routing_metrics["total_requests"] += 1
        self._routing_metrics["provider_selection_counts"][selected.provider] = \
            self._routing_metrics["provider_selection_counts"].get(selected.provider, 0) + 1

        return self.providers[selected.provider], selected.model

    async def route_request(
        self,
        request: LLMRequest,
        fallback_chain: Optional[List[str]] = None,
        criteria: Optional[RoutingCriteria] = None,
    ) -> LLMResponse:
        """Route request with automatic failover"""
        criteria = criteria or RoutingCriteria.from_request(request)
        fallback_chain = fallback_chain or self.config.fallback_providers

        last_error = None
        providers_tried = []

        for provider_name in [self.config.default_provider] + fallback_chain:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            providers_tried.append(provider_name)

            try:
                # Check health
                if not await self._is_healthy(provider):
                    logger.warning(f"Provider {provider_name} unhealthy, skipping")
                    continue

                # Get model
                model = request.model or provider.default_model

                # Check rate limit
                try:
                    await self.rate_limiter.acquire(provider=provider_name)
                except RateLimitError as e:
                    logger.warning(f"Rate limited on {provider_name}: {e}")
                    continue

                # Make request
                response = await provider.generate(request)

                # Track usage
                self._track_usage(response, provider_name, model)

                self._routing_metrics["successful_routes"] += 1
                return response

            except (RateLimitError, TimeoutError) as e:
                last_error = e
                logger.warning(f"Provider {provider_name} rate limited/timed out: {e}")
                continue

            except (ProviderError, LLMError) as e:
                last_error = e
                logger.error(f"Provider {provider_name} failed: {e}")
                providers_tried.append(f"{provider_name} ({type(e).__name__})")
                continue

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error with {provider_name}: {e}")
                continue

        # All providers failed
        error_key = type(last_error).__name__ if last_error else "unknown"
        self._routing_metrics["failure_reasons"][error_key] = \
            self._routing_metrics["failure_reasons"].get(error_key, 0) + 1

        raise FallbackError(
            providers=providers_tried,
            last_error=last_error,
        )

    async def route_stream(
        self,
        request: LLMRequest,
        stream_handler: StreamHandler,
        fallback_chain: Optional[List[str]] = None,
        criteria: Optional[RoutingCriteria] = None,
    ) -> StreamContext:
        """Route streaming request with failover"""
        criteria = criteria or RoutingCriteria.from_request(request)
        fallback_chain = fallback_chain or self.config.fallback_providers

        for provider_name in [self.config.default_provider] + fallback_chain:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]

            if not await self._is_healthy(provider):
                continue

            model = request.model or provider.default_model

            try:
                # Create stream context
                context = await stream_handler.create_stream(
                    request=request,
                    provider=provider_name,
                    model=model,
                )

                # Process stream
                await stream_handler.process_stream(provider, request, context)

                if context.state.value in ("completed",):
                    return context

            except Exception as e:
                logger.warning(f"Stream failed on {provider_name}: {e}")
                continue

        raise FallbackError(
            providers=fallback_chain,
            last_error=last_error if 'last_error' in dir() else None,
        )

    async def _get_available_providers(
        self,
        criteria: RoutingCriteria,
    ) -> List[str]:
        """Get list of available providers matching criteria"""
        available = []

        for name, config in self.config.providers.items():
            if not config.enabled:
                continue

            provider = self.providers.get(name)
            if not provider:
                continue

            # Check if provider supports requirements
            if criteria.requires_vision and not provider.supports_vision:
                continue

            if criteria.requires_functions and not provider.supports_functions:
                continue

            if criteria.requires_streaming and not provider.supports_streaming:
                continue

            # Check if any model supports the requirements
            has_suitable_model = False
            for model in config.models:
                if criteria.min_context_length > model.context_window:
                    continue
                has_suitable_model = True
                break

            if not has_suitable_model:
                continue

            # Check health
            if self.config.enable_health_checks:
                if not await self._is_healthy(provider):
                    continue

            available.append(name)

        return available

    async def _score_providers(
        self,
        providers: List[str],
        criteria: RoutingCriteria,
    ) -> List[ProviderScore]:
        """Score providers based on criteria"""
        scores = []

        for name in providers:
            config = self.config.providers.get(name)
            provider = self.providers.get(name)

            if not config or not provider:
                continue

            score = ProviderScore(provider=name)

            # Capability score (0-100)
            capability = 0
            if provider.supports_vision and not criteria.requires_vision:
                capability += 10
            elif provider.supports_vision and criteria.requires_vision:
                capability += 25
            else:
                capability -= 10

            if provider.supports_functions and criteria.requires_functions:
                capability += 25
            elif not provider.supports_functions and criteria.requires_functions:
                capability -= 50

            score.capability_score = max(0, min(100, 50 + capability))

            # Cost score (0-100, higher is cheaper)
            if config.models:
                avg_cost = sum(
                    m.cost_per_input + m.cost_per_output
                    for m in config.models
                ) / len(config.models)

                if avg_cost == 0:
                    cost_score = 100  # Free provider
                else:
                    cost_score = min(100, (criteria.max_cost_per_token / avg_cost) * 50)

                if criteria.prefer_cheap:
                    cost_score *= 1.5

            else:
                cost_score = 50

            score.cost_score = cost_score

            # Latency score (0-100, lower is better)
            if provider.health_status.latency_ms:
                latency_score = max(0, 100 - (provider.health_status.latency_ms / criteria.max_latency_ms) * 100)
                if criteria.prefer_fast:
                    latency_score *= 1.5
            else:
                latency_score = 50

            score.latency_score = min(100, latency_score)

            # Reliability score (0-100)
            if provider.health_status.request_count > 0:
                error_rate = provider.health_status.error_count / provider.health_status.request_count
                reliability_score = (1 - error_rate) * 100
                if provider.health_status.consecutive_failures > 0:
                    reliability_score *= 0.5
            else:
                reliability_score = 75  # Unknown, assume reasonable

            if criteria.prefer_reliable:
                reliability_score *= 1.2

            score.reliability_score = min(100, reliability_score)

            # Health status
            score.available = provider.health_status.is_healthy
            score.health_status = "healthy" if provider.health_status.is_healthy else "unhealthy"

            # Calculate total score
            weights = {"capability": 0.3, "cost": 0.2, "latency": 0.25, "reliability": 0.25}
            score.total_score = (
                score.capability_score * weights["capability"] +
                score.cost_score * weights["cost"] +
                score.latency_score * weights["latency"] +
                score.reliability_score * weights["reliability"]
            )

            scores.append(score)

        return sorted(scores, key=lambda s: s.total_score, reverse=True)

    def _apply_strategy(
        self,
        scores: List[ProviderScore],
        criteria: RoutingCriteria,
    ) -> ProviderScore:
        """Apply routing strategy to select provider"""
        strategy = self.config.routing_strategy

        if strategy == RoutingStrategy.PRIORITY:
            # Sort by configured priority
            for score in scores:
                config = self.config.providers.get(score.provider)
                if config:
                    return score

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin would need state tracking
            # For now, return highest scored
            return scores[0]

        elif strategy == RoutingStrategy.COST_AWARE:
            # Prioritize cheaper options for simple tasks
            if criteria.task_complexity < 5:
                return min(scores, key=lambda s: s.cost_score)
            return scores[0]

        elif strategy == RoutingStrategy.CAPABILITY_MATCH:
            # Prioritize capability match
            return max(scores, key=lambda s: s.capability_score)

        elif strategy == RoutingStrategy.LOAD_BALANCED:
            # Would need load tracking
            return scores[0]

        # Default: highest score
        return scores[0]

    async def _is_healthy(self, provider: BaseProvider) -> bool:
        """Check if provider is healthy"""
        now = datetime.now()

        # Check cache
        if provider.provider_name in self._health_cache:
            cached_health, cached_time = self._health_cache[provider.provider_name]
            if (now - cached_time).total_seconds() < self._health_cache_ttl:
                return cached_health

        # Perform health check
        try:
            is_healthy = await provider.health_check()
            self._health_cache[provider.provider_name] = (is_healthy, now)
            return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {provider.provider_name}: {e}")
            self._health_cache[provider.provider_name] = (False, now)
            return False

    def _track_usage(
        self,
        response: LLMResponse,
        provider: str,
        model: str,
    ) -> None:
        """Track token usage"""
        config = self.config.providers.get(provider)
        model_config = None
        if config:
            for m in config.models:
                if m.name == model:
                    model_config = m
                    break

        cost = self.token_tracker.calculate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model_config,
        )

        self.token_tracker.record_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider=provider,
            model=model,
            cost=cost,
            request_id=response.request_id,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        return {
            "total_requests": self._routing_metrics["total_requests"],
            "successful_routes": self._routing_metrics["successful_routes"],
            "fallback_routes": self._routing_metrics["fallback_routes"],
            "provider_selection_counts": dict(self._routing_metrics["provider_selection_counts"]),
            "failure_reasons": dict(self._routing_metrics["failure_reasons"]),
            "fallback_rate": (
                self._routing_metrics["fallback_routes"] / self._routing_metrics["total_requests"]
                if self._routing_metrics["total_requests"] > 0 else 0
            ),
        }

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all providers"""
        results = {}

        for name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                results[name] = {
                    "healthy": is_healthy,
                    "latency_ms": provider.health_status.latency_ms,
                    "error": provider.health_status.last_error,
                }
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                }

        return results


class CostAwareRouter(Router):
    """Router with enhanced cost-aware routing"""

    def __init__(
        self,
        config: LLMConfig,
        providers: Dict[str, BaseProvider],
        token_tracker: Optional[TokenTracker] = None,
        budget_limit_daily: Optional[float] = None,
    ):
        super().__init__(config, providers, token_tracker)
        self.budget_limit_daily = budget_limit_daily

    async def select_provider(
        self,
        request: LLMRequest,
        criteria: Optional[RoutingCriteria] = None,
    ) -> Tuple[BaseProvider, str]:
        """Select provider with budget awareness"""
        criteria = criteria or RoutingCriteria.from_request(request)

        # Check daily budget
        if self.budget_limit_daily:
            daily_usage = self.token_tracker.get_daily_usage()
            daily_cost = daily_usage.get("cost", 0)
            estimated_cost = self._estimate_cost(request, criteria)

            if daily_cost + estimated_cost > self.budget_limit_daily:
                # Force cheaper model
                criteria.prefer_cheap = True
                criteria.max_cost_per_token = 0.001

        return await super().select_provider(request, criteria)

    def _estimate_cost(
        self,
        request: LLMRequest,
        criteria: RoutingCriteria,
    ) -> float:
        """Estimate cost for a request"""
        # Simple estimation based on token count and average model cost
        avg_cost_per_token = 0.01  # Default assumption
        return criteria.estimated_tokens * avg_cost_per_token

    async def route_request(
        self,
        request: LLMRequest,
        fallback_chain: Optional[List[str]] = None,
        criteria: Optional[RoutingCriteria] = None,
    ) -> LLMResponse:
        """Route with cost optimization"""
        criteria = criteria or RoutingCriteria.from_request(request)

        # For simple tasks, prefer cheaper models
        if criteria.task_complexity < 3 and not criteria.requires_functions:
            criteria.prefer_cheap = True

        return await super().route_request(request, fallback_chain, criteria)


class FallbackChain:
    """Manages fallback chains for requests"""

    def __init__(self, router: Router):
        self.router = router
        self._fallback_history: Dict[str, List[str]] = {}

    def create_chain(
        self,
        primary: str,
        fallbacks: List[str],
    ) -> List[str]:
        """Create a fallback chain"""
        return [primary] + fallbacks

    def get_chain_for_task(
        self,
        task_type: str,
        complexity: int,
    ) -> List[str]:
        """Get fallback chain for a task type"""
        # Define task-specific chains
        task_chains = {
            "reasoning": ["openai/o1", "anthropic/claude-opus-4", "bedrock/claude-3-opus"],
            "vision": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet", "bedrock/claude-3-5-sonnet"],
            "coding": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet", "ollama/deepseek-coder"],
            "fast": ["openai/gpt-4o-mini", "anthropic/claude-haiku-3", "ollama/llama3.2"],
            "cheap": ["ollama/llama3.2", "anthropic/claude-haiku-3", "openai/gpt-4o-mini"],
        }

        return task_chains.get(task_type, [])

    def record_fallback(
        self,
        request_id: str,
        from_provider: str,
        to_provider: str,
        reason: str,
    ) -> None:
        """Record a fallback event"""
        if request_id not in self._fallback_history:
            self._fallback_history[request_id] = []

        self._fallback_history[request_id].append({
            "from": from_provider,
            "to": to_provider,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def get_fallback_history(self, request_id: str) -> List[Dict[str, Any]]:
        """Get fallback history for a request"""
        return self._fallback_history.get(request_id, [])
