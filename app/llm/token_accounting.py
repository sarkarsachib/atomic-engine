#!/usr/bin/env python3
"""
Token Accounting Module
Tracks token usage, calculates costs, and implements rate limiting
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging
import threading

from .config import ModelConfig
from .exceptions import RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage record"""
    timestamp: datetime = field(default_factory=datetime.now)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    provider: str = ""
    model: str = ""
    request_id: Optional[str] = None
    task_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "task_type": self.task_type,
            "metadata": self.metadata,
        }


@dataclass
class UsageSummary:
    """Usage summary for a period"""
    start_time: datetime
    end_time: datetime
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_task_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "by_provider": self.by_provider,
            "by_model": self.by_model,
            "by_task_type": self.by_task_type,
        }


class TokenTracker:
    """Tracks token usage and costs across providers"""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        flush_interval_seconds: int = 60,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("~/.atomic_engine/usage.json").expanduser()
        self.flush_interval = flush_interval_seconds

        # In-memory storage
        self._usage_records: List[TokenUsage] = []
        self._daily_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._provider_usage: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
        self._model_usage: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
        self._task_usage: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))

        # Thread safety
        self._lock = threading.Lock()

        # Tracking state
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._total_requests: int = 0

        # Start flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background thread to periodically flush usage data"""
        while self._running:
            time.sleep(self.flush_interval)
            try:
                self.flush()
            except Exception as e:
                logger.warning(f"Failed to flush usage data: {e}")

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
        cost: float = 0.0,
        request_id: Optional[str] = None,
        task_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenUsage:
        """Record token usage for a request"""
        usage = TokenUsage(
            timestamp=datetime.now(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            provider=provider,
            model=model,
            request_id=request_id,
            task_type=task_type,
            metadata=metadata or {},
        )

        with self._lock:
            self._usage_records.append(usage)
            self._total_cost += cost
            self._total_tokens += input_tokens + output_tokens
            self._total_requests += 1

            # Update daily usage
            today = datetime.now().strftime("%Y-%m-%d")
            self._daily_usage[today]["input_tokens"] += input_tokens
            self._daily_usage[today]["output_tokens"] += output_tokens
            self._daily_usage[today]["total_tokens"] += input_tokens + output_tokens
            self._daily_usage[today]["cost"] += cost
            self._daily_usage[today]["requests"] = self._daily_usage[today].get("requests", 0) + 1

            # Update provider usage
            self._provider_usage[provider]["requests"] = self._provider_usage[provider].get("requests", 0) + 1
            self._provider_usage[provider]["input_tokens"] = self._provider_usage[provider].get("input_tokens", 0) + input_tokens
            self._provider_usage[provider]["output_tokens"] = self._provider_usage[provider].get("output_tokens", 0) + output_tokens
            self._provider_usage[provider]["cost"] = self._provider_usage[provider].get("cost", 0) + cost

            # Update model usage
            key = f"{provider}/{model}"
            self._model_usage[key]["requests"] = self._model_usage[key].get("requests", 0) + 1
            self._model_usage[key]["input_tokens"] = self._model_usage[key].get("input_tokens", 0) + input_tokens
            self._model_usage[key]["output_tokens"] = self._model_usage[key].get("output_tokens", 0) + output_tokens
            self._model_usage[key]["cost"] = self._model_usage[key].get("cost", 0) + cost

            # Update task type usage
            if task_type:
                self._task_usage[task_type]["requests"] = self._task_usage[task_type].get("requests", 0) + 1
                self._task_usage[task_type]["input_tokens"] = self._task_usage[task_type].get("input_tokens", 0) + input_tokens
                self._task_usage[task_type]["output_tokens"] = self._task_usage[task_type].get("output_tokens", 0) + output_tokens
                self._task_usage[task_type]["cost"] = self._task_usage[task_type].get("cost", 0) + cost

        return usage

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_config: Optional[ModelConfig] = None,
    ) -> float:
        """Calculate cost for token usage"""
        if not model_config or model_config.cost_per_input == 0:
            return 0.0

        return (
            input_tokens * model_config.cost_per_input / 1_000_000 +
            output_tokens * model_config.cost_per_output / 1_000_000
        )

    def get_daily_usage(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get usage for a specific date"""
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            return dict(self._daily_usage.get(target_date, {}))

    def get_provider_usage(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get usage breakdown by provider"""
        with self._lock:
            if provider:
                return dict(self._provider_usage.get(provider, {}))
            return dict(self._provider_usage)

    def get_model_usage(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get usage breakdown by model"""
        with self._lock:
            if model:
                for key, value in self._model_usage.items():
                    if model in key:
                        return dict(value)
                return {}
            return dict(self._model_usage)

    def get_task_usage(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get usage breakdown by task type"""
        with self._lock:
            if task_type:
                return dict(self._task_usage.get(task_type, {}))
            return dict(self._task_usage)

    def get_summary(self, days: int = 1) -> UsageSummary:
        """Get usage summary for the last N days"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        summary = UsageSummary(
            start_time=start_time,
            end_time=end_time,
        )

        with self._lock:
            summary.total_requests = self._total_requests
            summary.total_input_tokens = sum(d.get("input_tokens", 0) for d in self._daily_usage.values())
            summary.total_output_tokens = sum(d.get("output_tokens", 0) for d in self._daily_usage.values())
            summary.total_cost = self._total_cost

            # Copy provider, model, and task breakdowns
            summary.by_provider = dict(self._provider_usage)
            summary.by_model = dict(self._model_usage)
            summary.by_task_type = dict(self._task_usage)

        return summary

    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        with self._lock:
            return {
                "total_cost": round(self._total_cost, 4),
                "total_tokens": self._total_tokens,
                "total_requests": self._total_requests,
                "records_count": len(self._usage_records),
            }

    def flush(self) -> None:
        """Flush usage data to storage"""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                usage_data = [u.to_dict() for u in self._usage_records]

            with open(self.storage_path, 'w') as f:
                json.dump(usage_data, f, indent=2)

            logger.debug(f"Flushed {len(usage_data)} usage records to {self.storage_path}")

        except Exception as e:
            logger.warning(f"Failed to flush usage data: {e}")

    def clear(self) -> None:
        """Clear all usage data"""
        with self._lock:
            self._usage_records.clear()
            self._daily_usage.clear()
            self._provider_usage.clear()
            self._model_usage.clear()
            self._task_usage.clear()
            self._total_cost = 0.0
            self._total_tokens = 0
            self._total_requests = 0

    def shutdown(self) -> None:
        """Shutdown the tracker and flush remaining data"""
        self._running = False
        self.flush()

    def export_csv(self, path: str) -> None:
        """Export usage data to CSV"""
        import csv

        with self._lock:
            records = [u.to_dict() for u in self._usage_records]

        with open(path, 'w', newline='') as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)


class RateLimiter:
    """Rate limiter with exponential backoff support"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
        requests_per_day: int = None,
        tokens_per_day: int = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.rpd = requests_per_day
        self.tpd = tokens_per_day
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

        # Tracking
        self._request_times: List[float] = []
        self._token_usage: List[tuple] = []  # (timestamp, tokens)
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        estimated_tokens: int = 0,
        provider: str = "default",
    ) -> None:
        """Acquire rate limit permission"""
        async with self._lock:
            now = time.time()
            minute_ago = now - 60
            day_ago = now - 86400

            # Clean old records
            self._request_times = [t for t in self._request_times if t > minute_ago]
            self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > minute_ago]

            # Check request rate limit
            request_count = len(self._request_times)
            if request_count >= self.rpm:
                retry_after = self._request_times[0] - minute_ago if self._request_times else 1.0
                raise RateLimitError(
                    message=f"Rate limit exceeded: {request_count} requests/minute",
                    provider=provider,
                    retry_after_seconds=retry_after,
                    limit_type="requests",
                    current_usage=request_count,
                    limit_value=self.rpm,
                )

            # Check token rate limit
            token_usage = sum(tokens for _, tokens in self._token_usage)
            if token_usage + estimated_tokens >= self.tpm:
                raise RateLimitError(
                    message=f"Token rate limit exceeded: {token_usage + estimated_tokens}/{self.tpm}",
                    provider=provider,
                    retry_after_seconds=60.0,
                    limit_type="tokens",
                    current_usage=token_usage + estimated_tokens,
                    limit_value=self.tpm,
                )

            # Record this request
            self._request_times.append(now)
            self._token_usage.append((now, estimated_tokens))

    async def wait_for_token(
        self,
        estimated_tokens: int = 0,
        provider: str = "default",
    ) -> float:
        """Wait for rate limit to clear, returns wait time"""
        now = time.time()
        minute_ago = now - 60

        # Check if we need to wait
        request_count = sum(1 for t in self._request_times if t > minute_ago)
        token_usage = sum(tokens for t, tokens in self._token_usage if t > minute_ago)

        if request_count >= self.rpm:
            # Wait for oldest request to expire
            if self._request_times:
                return max(0, self._request_times[0] - minute_ago)
            return 1.0

        if token_usage + estimated_tokens >= self.tpm:
            # Wait for token usage to clear
            return 60.0

        return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        now = time.time()
        minute_ago = now - 60

        request_count = sum(1 for t in self._request_times if t > minute_ago)
        token_usage = sum(tokens for t, tokens in self._token_usage if t > minute_ago)

        return {
            "requests_per_minute": {
                "used": request_count,
                "limit": self.rpm,
                "remaining": max(0, self.rpm - request_count),
            },
            "tokens_per_minute": {
                "used": token_usage,
                "limit": self.tpm,
                "remaining": max(0, self.tpm - token_usage),
            },
        }

    def reset(self) -> None:
        """Reset rate limiter state"""
        self._request_times.clear()
        self._token_usage.clear()


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on provider responses"""

    def __init__(
        self,
        base_rpm: int = 60,
        base_tpm: int = 100000,
        window_seconds: int = 60,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
        max_multiplier: float = 2.0,
    ):
        self.base_rpm = base_rpm
        self.base_tpm = base_tpm
        self.window_seconds = window_seconds
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.max_multiplier = max_multiplier

        # Current limits
        self.current_rpm = base_rpm
        self.current_tpm = base_tpm

        # Tracking
        self._lock = asyncio.Lock()
        self._consecutive_errors = 0
        self._last_error_time: Optional[float] = None

    async def acquire(
        self,
        estimated_tokens: int = 0,
        provider: str = "default",
    ) -> None:
        """Acquire rate limit with adaptive adjustment"""
        async with self._lock:
            now = time.time()

            # Check for recent errors
            if self._last_error_time:
                time_since_error = now - self._last_error_time
                if time_since_error < self.window_seconds:
                    # Still in backoff period
                    if self._consecutive_errors > 0:
                        delay = self.window_seconds * self.backoff_factor * self._consecutive_errors
                        raise RateLimitError(
                            message=f"Adaptive rate limit: {self._consecutive_errors} consecutive errors",
                            provider=provider,
                            retry_after_seconds=delay,
                        )

    async def record_success(self) -> None:
        """Record a successful request for recovery"""
        async with self._lock:
            if self._consecutive_errors > 0:
                self._consecutive_errors = 0
                self.current_rpm = min(self.base_rpm * self.max_multiplier, self.current_rpm * self.recovery_factor)
                self.current_tpm = min(self.base_tpm * self.max_multiplier, self.current_tpm * self.recovery_factor)

    async def record_error(self, is_rate_limit: bool = False) -> None:
        """Record an error for backoff"""
        async with self._lock:
            self._consecutive_errors += 1
            self._last_error_time = time.time()

            if is_rate_limit or self._consecutive_errors > 1:
                self.current_rpm = max(1, int(self.current_rpm * self.backoff_factor))
                self.current_tpm = max(1000, int(self.current_tpm * self.backoff_factor))

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            "current_rpm": self.current_rpm,
            "base_rpm": self.base_rpm,
            "current_tpm": self.current_tpm,
            "base_tpm": self.base_tpm,
            "consecutive_errors": self._consecutive_errors,
        }

    def reset(self) -> None:
        """Reset to base limits"""
        self.current_rpm = self.base_rpm
        self.current_tpm = self.base_tpm
        self._consecutive_errors = 0
        self._last_error_time = None
