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
        """
        Derive a RoutingCriteria instance from an LLMRequest and optional metadata.
        
        Parameters:
            request (LLMRequest): The incoming LLM request used to infer routing requirements.
            metadata (Optional[Dict[str, Any]]): Optional metadata that overrides or extends request.metadata.
        
        Returns:
            RoutingCriteria: Criteria populated from the request (vision/function/streaming requirements, min context length, task complexity, and estimated token count).
        """
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
        """
        Return a dictionary representation of the ProviderScore containing all scoring components and status.
        
        Returns:
            dict: Mapping with keys:
                - "provider" (str): Provider name.
                - "model" (str): Model identifier.
                - "total_score" (float): Aggregate score used for ranking.
                - "capability_score" (float): Score for capability match.
                - "cost_score" (float): Cost-related score.
                - "latency_score" (float): Latency-related score.
                - "reliability_score" (float): Reliability-related score.
                - "available" (bool): Whether the provider is considered available.
                - "health_status" (str): Health state label.
                - "error" (Optional[str]): Optional error message if present.
        """
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
        """
        Initialize the Router with configuration, available providers, and optional utilities for token tracking and rate limiting.
        
        Parameters:
            config (LLMConfig): Router configuration and routing strategy settings.
            providers (Dict[str, BaseProvider]): Mapping of provider names to provider instances available for routing.
            token_tracker (Optional[TokenTracker]): Optional token/cost accounting utility; a default TokenTracker is created if omitted.
            rate_limiter (Optional[RateLimiter]): Optional rate-limiting utility; a default RateLimiter is created if omitted.
        
        Details:
            - Initializes an in-memory health check cache with a 5-minute TTL.
            - Initializes routing metrics used to track request counts, successes, fallbacks, per-provider selection counts, and failure reasons.
        """
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
        """
        Choose the most appropriate provider and model for the given LLM request using routing criteria and the configured routing strategy.
        
        Parameters:
            request (LLMRequest): The incoming LLM request to route.
            criteria (Optional[RoutingCriteria]): Constraints and preferences to filter and rank providers; if omitted, criteria are derived from the request.
        
        Returns:
            Tuple[BaseProvider, str]: The selected provider instance and the chosen model name.
        
        Raises:
            RoutingError: If no providers match the criteria or no providers can satisfy the requirements.
        """
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
        """
        Selects an available provider and sends the given LLMRequest, automatically failing over through the configured fallback providers until a successful LLMResponse is returned.
        
        Parameters:
            request (LLMRequest): The request to send to a provider.
            fallback_chain (Optional[List[str]]): Ordered list of provider names to try after the router's default provider. If omitted, the router's configured fallback_providers are used.
            criteria (Optional[RoutingCriteria]): Selection criteria derived from the request; if omitted, criteria are computed from the request.
        
        Returns:
            LLMResponse: The successful response from the first provider that completes the request.
        
        Raises:
            FallbackError: If every attempted provider fails or is skipped; includes the list of providers tried and the last underlying error.
        """
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
        """
        Routes a streaming LLM request across the default provider and an optional fallback chain until a provider completes the stream.
        
        Parameters:
            request (LLMRequest): The streaming request to send.
            stream_handler (StreamHandler): Handler responsible for creating and processing the stream for a provider.
            fallback_chain (Optional[List[str]]): Ordered list of provider names to try after the default provider.
            criteria (Optional[RoutingCriteria]): Constraints used to filter/select providers; derived from `request` if omitted.
        
        Returns:
            StreamContext: The stream context whose state indicates a completed stream.
        
        Raises:
            FallbackError: If no provider completes the stream after attempting the default provider and all fallbacks.
        """
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
        """
        Return the names of configured providers that meet the given routing criteria and are currently available.
        
        Filters providers by: being enabled and present in the router, supporting required capabilities (vision, function calling, streaming), having at least one model with a context window greater than or equal to `criteria.min_context_length`, and — if health checks are enabled in the router config — passing a health check.
        
        Parameters:
            criteria (RoutingCriteria): Selection constraints used to filter providers.
        
        Returns:
            List[str]: A list of provider names that satisfy the criteria and availability checks.
        """
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
        """
        Compute a score for each named provider against the given routing criteria and return providers ranked by total score (highest first).
        
        Scores combine capability, cost (higher means cheaper), latency (lower is better), and reliability into a weighted total. Capability, cost, latency, and reliability component scores range from 0 to 100; total_score is a weighted sum of those components.
        
        Parameters:
            providers (List[str]): Iterable of provider names to evaluate.
            criteria (RoutingCriteria): Selection constraints and preferences used to influence component scores (e.g., vision/functions requirements, max cost per token, max latency, and prefer_* flags).
        
        Returns:
            List[ProviderScore]: A list of ProviderScore objects for the evaluated providers, sorted by `total_score` in descending order.
        """
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
        """
        Selects a ProviderScore from scored candidates according to the router's configured routing strategy.
        
        Behavior by strategy:
        - PRIORITY: returns the first scored provider that has a provider configuration.
        - ROUND_ROBIN / LOAD_BALANCED: returns the top-scoring provider.
        - COST_AWARE: if criteria.task_complexity < 5, returns the provider with the lowest `cost_score`; otherwise returns the top-scoring provider.
        - CAPABILITY_MATCH: returns the provider with the highest `capability_score`.
        - Default: returns the top-scoring provider.
        
        Parameters:
            scores (List[ProviderScore]): Candidate provider scores, expected sorted by `total_score` descending.
            criteria (RoutingCriteria): Routing constraints used to influence strategy decisions (e.g., `task_complexity`).
        
        Returns:
            ProviderScore: The selected provider score according to the applied routing strategy.
        """
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
        """
        Determine whether the given provider is currently healthy, using an in-memory cache with a time-to-live to avoid repeated checks.
        
        Parameters:
            provider (BaseProvider): Provider instance whose health will be checked.
        
        Returns:
            bool: `true` if the provider is healthy, `false` otherwise. If a cached recent result exists it is returned; on exceptions the provider is marked unhealthy and `false` is returned.
        """
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
        """
        Record token and cost usage for a completed LLM response with the configured token tracker.
        
        Calculates the cost for the response using the provider/model configuration (if available) and records input and output token counts, the computed cost, provider, model, and the response's request_id with the token tracker.
        
        Parameters:
            response (LLMResponse): The response containing usage (input/output tokens) and request_id.
            provider (str): The provider name that produced the response.
            model (str): The model name used to generate the response.
        """
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
        """
        Return a snapshot of router operational metrics.
        
        Returns:
            metrics (Dict[str, Any]): Dictionary containing routing statistics:
                - total_requests (int): Total number of routing attempts.
                - successful_routes (int): Number of requests successfully routed.
                - fallback_routes (int): Number of times a fallback provider was used.
                - provider_selection_counts (Dict[str, int]): Per-provider selection counts.
                - failure_reasons (Dict[str, int]): Counts of failures keyed by reason.
                - fallback_rate (float): Ratio of fallback_routes to total_requests (0 if no requests).
        """
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
        """
        Run health checks for every configured provider and collect their status.
        
        Returns:
            mapping (Dict[str, Dict[str, Any]]): A dictionary keyed by provider name. Each value contains:
                - "healthy" (bool): Provider health boolean.
                - "latency_ms" (float, optional): Last observed latency in milliseconds when available.
                - "error" (str, optional): Last error message from the provider health check or an exception string if the check failed.
        """
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
        """
        Initialize a CostAwareRouter with configuration, available providers, and optional services.
        
        Parameters:
            budget_limit_daily (float | None): Optional daily budget cap in the router's currency units. If set, the router will apply cost-aware selection to avoid exceeding this daily budget; if `None`, no daily budget constraint is enforced.
        """
        super().__init__(config, providers, token_tracker)
        self.budget_limit_daily = budget_limit_daily

    async def select_provider(
        self,
        request: LLMRequest,
        criteria: Optional[RoutingCriteria] = None,
    ) -> Tuple[BaseProvider, str]:
        """
        Selects a provider and model while enforcing daily budget constraints when a daily budget is configured.
        
        If the estimated cost for the request would cause the daily spend to exceed the configured daily budget, adjusts the routing criteria to prefer cheaper models and tighten the per-token cost cap before delegating to the base router.
        
        Parameters:
            request (LLMRequest): The request to route.
            criteria (Optional[RoutingCriteria]): Optional routing criteria; created from the request if omitted.
        
        Returns:
            Tuple[BaseProvider, str]: The chosen provider instance and the selected model name.
        """
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
        """
        Estimate the monetary cost of fulfilling an LLM request using the routing criteria.
        
        Parameters:
            request (LLMRequest): The original LLM request (used for context; not all implementations use fields from it).
            criteria (RoutingCriteria): Criteria providing `estimated_tokens` used for the calculation.
        
        Returns:
            float: Estimated cost as a monetary value (float). Calculation uses `criteria.estimated_tokens` and a fixed per-token rate (default 0.01), and should be treated as a rough estimate.
        """
        # Simple estimation based on token count and average model cost
        avg_cost_per_token = 0.01  # Default assumption
        return criteria.estimated_tokens * avg_cost_per_token

    async def route_request(
        self,
        request: LLMRequest,
        fallback_chain: Optional[List[str]] = None,
        criteria: Optional[RoutingCriteria] = None,
    ) -> LLMResponse:
        """
        Route an LLMRequest while biasing provider selection toward lower-cost options for simple tasks.
        
        Builds routing criteria from the request when none is provided. If the task complexity is less than 3 and the request does not require function calling, the criteria will be adjusted to prefer cheaper providers. Performs provider selection, failover across the optional fallback_chain, health checks, and returns the successful provider response.
        
        Parameters:
            request (LLMRequest): The request to route.
            fallback_chain (Optional[List[str]]): Ordered list of provider names to try after the primary provider, used for failover.
            criteria (Optional[RoutingCriteria]): Optional routing constraints and preferences; if omitted, derived from the request.
        
        Returns:
            LLMResponse: The response produced by the selected provider.
        """
        criteria = criteria or RoutingCriteria.from_request(request)

        # For simple tasks, prefer cheaper models
        if criteria.task_complexity < 3 and not criteria.requires_functions:
            criteria.prefer_cheap = True

        return await super().route_request(request, fallback_chain, criteria)


class FallbackChain:
    """Manages fallback chains for requests"""

    def __init__(self, router: Router):
        """
        Initialize a FallbackChain associated with a Router and prepare per-request fallback history.
        
        Parameters:
            router (Router): Router instance used to derive provider order and record fallback events. The instance is retained for lookups and chain creation.
        
        Notes:
            Creates an empty in-memory mapping from request ID to a list of provider names representing the fallback sequence for that request.
        """
        self.router = router
        self._fallback_history: Dict[str, List[str]] = {}

    def create_chain(
        self,
        primary: str,
        fallbacks: List[str],
    ) -> List[str]:
        """
        Builds an ordered provider fallback chain starting with the primary provider.
        
        Parameters:
            primary (str): Name of the primary provider.
            fallbacks (List[str]): Ordered list of fallback provider names.
        
        Returns:
            List[str]: Combined list beginning with `primary` followed by the `fallbacks`.
        """
        return [primary] + fallbacks

    def get_chain_for_task(
        self,
        task_type: str,
        complexity: int,
    ) -> List[str]:
        """
        Provide a prioritized fallback provider chain for a given task type.
        
        Parameters:
            task_type (str): The task category (e.g., "reasoning", "vision", "coding", "fast", "cheap").
            complexity (int): Task complexity level; currently unused by this selection method.
        
        Returns:
            List[str]: Provider identifiers in priority order for the task; empty list if no chain is defined.
        """
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
        """
        Record a fallback transition for a request in the router's fallback history.
        
        Appends a structured event containing the source provider, destination provider,
        reason, and an ISO 8601 timestamp to the router's internal per-request fallback log.
        
        Parameters:
            request_id (str): Unique identifier of the request.
            from_provider (str): Name of the provider that failed or was replaced.
            to_provider (str): Name of the provider that was selected as the fallback.
            reason (str): Short description of why the fallback occurred.
        """
        if request_id not in self._fallback_history:
            self._fallback_history[request_id] = []

        self._fallback_history[request_id].append({
            "from": from_provider,
            "to": to_provider,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def get_fallback_history(self, request_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve the recorded fallback events for a given request.
        
        Parameters:
            request_id (str): The unique identifier of the request whose fallback history to fetch.
        
        Returns:
            List[Dict[str, Any]]: A list of fallback event records for the request, in chronological order.
            Each record is a dictionary containing at least:
              - `timestamp` (str): ISO8601 timestamp when the fallback occurred.
              - `from_provider` (str): Provider that was attempted before the fallback.
              - `to_provider` (str): Provider that was tried as a fallback.
              - `reason` (str): Short description of why the fallback occurred.
            Returns an empty list if no history exists for the given request_id.
        """
        return self._fallback_history.get(request_id, [])