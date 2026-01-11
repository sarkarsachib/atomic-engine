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
        """
        Convert the TokenUsage record into a dictionary suitable for serialization.
        
        Returns:
            dict: A mapping with keys `timestamp` (ISO 8601 string), `input_tokens`, `output_tokens`,
            `total_tokens`, `cost`, `provider`, `model`, `request_id`, `task_type`, and `metadata`.
        """
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
        """
        Convert the UsageSummary into a JSON-serializable dictionary.
        
        Returns:
            summary (dict): Dictionary containing the summary fields:
                - `start_time` (str): ISO 8601 string of the summary start time.
                - `end_time` (str): ISO 8601 string of the summary end time.
                - `total_requests` (int): Total number of requests in the period.
                - `total_input_tokens` (int): Sum of input tokens across requests.
                - `total_output_tokens` (int): Sum of output tokens across requests.
                - `total_tokens` (int): Sum of input and output tokens.
                - `total_cost` (float): Total cost aggregated for the period.
                - `by_provider` (dict): Breakdown of usage keyed by provider.
                - `by_model` (dict): Breakdown of usage keyed by provider/model.
                - `by_task_type` (dict): Breakdown of usage keyed by task type.
        """
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
        """
        Initialize the TokenTracker, configuring storage, in-memory structures, and a background flush thread.
        
        Sets the path used for persisted usage data, the periodic flush interval, initializes in-memory records and aggregations, thread-safety primitives, cumulative counters, and starts a daemon thread that runs the periodic flush loop.
        
        Parameters:
            storage_path (Optional[str]): Path to the JSON file where usage is persisted. If omitted, defaults to "~/.atomic_engine/usage.json".
            flush_interval_seconds (int): Number of seconds between automatic background flushes to disk.
        """
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
        """
        Record a token usage entry and update in-memory aggregates for daily, provider, model, and task-type usage.
        
        Parameters:
            input_tokens (int): Number of input tokens consumed by the request.
            output_tokens (int): Number of output tokens produced by the request.
            provider (str): Provider identifier (e.g., "openai", "anthropic").
            model (str): Model identifier used for the request.
            cost (float): Cost attributed to this request (same currency/units used by the tracker). Defaults to 0.0.
            request_id (Optional[str]): Optional external request identifier to correlate logs or traces.
            task_type (Optional[str]): Optional task type label to categorize usage (e.g., "chat", "embedding").
            metadata (Optional[Dict[str, Any]]): Optional arbitrary metadata stored with the usage record.
        
        Returns:
            TokenUsage: The created TokenUsage record representing the recorded request.
        """
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
        """
        Compute the total cost for the given input and output token counts using rates from the provided model configuration.
        
        Parameters:
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
            model_config (Optional[ModelConfig]): Model pricing where `cost_per_input` and `cost_per_output` are specified per 1,000,000 tokens. If `None` or if `cost_per_input` is zero, the cost is treated as 0.0.
        
        Returns:
            float: Total cost computed as (input_tokens * cost_per_input + output_tokens * cost_per_output) / 1_000_000.
        """
        if not model_config or model_config.cost_per_input == 0:
            return 0.0

        return (
            input_tokens * model_config.cost_per_input / 1_000_000 +
            output_tokens * model_config.cost_per_output / 1_000_000
        )

    def get_daily_usage(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve aggregated token usage for a specific day.
        
        Parameters:
            date (Optional[str]): Date string in "YYYY-MM-DD" format. If omitted, uses today's date.
        
        Returns:
            dict: A copy of the usage summary for the specified date containing keys such as
            "requests", "input_tokens", "output_tokens", "total_tokens", and "cost". Returns an
            empty dict if no records exist for that date.
        """
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            return dict(self._daily_usage.get(target_date, {}))

    def get_provider_usage(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve token usage statistics for providers.
        
        If `provider` is specified, return that provider's aggregated statistics; otherwise return a mapping of provider names to their aggregated statistics. The returned dictionary is a shallow copy and safe for callers to inspect or modify without affecting internal state.
        
        Parameters:
            provider (Optional[str]): Provider name to filter by. If omitted, returns all providers.
        
        Returns:
            dict: If `provider` was provided, a dict of that provider's stats (e.g., `requests`, `input_tokens`, `output_tokens`, `cost`); otherwise a mapping from provider name to such stats.
        """
        with self._lock:
            if provider:
                return dict(self._provider_usage.get(provider, {}))
            return dict(self._provider_usage)

    def get_model_usage(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve usage statistics for models, optionally filtered by a model name substring.
        
        Parameters:
            model (str | None): Optional substring to match against stored model keys (formatted as "provider/model"). If provided, the first matching model's statistics are returned.
        
        Returns:
            dict: If `model` is given, the matched model's statistics as a dictionary or an empty dict if no match is found. If `model` is None, a dictionary of all model usage keyed by stored model identifiers.
        """
        with self._lock:
            if model:
                for key, value in self._model_usage.items():
                    if model in key:
                        return dict(value)
                return {}
            return dict(self._model_usage)

    def get_task_usage(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve token usage statistics grouped by task type.
        
        Parameters:
            task_type (Optional[str]): If provided, return the usage metrics for that specific task type; otherwise return metrics for all task types.
        
        Returns:
            Dict[str, Any]: A dictionary mapping task type keys to their usage metrics. If `task_type` is provided, returns the metrics for that task type (an empty dict if the task type is not present). This operation is thread-safe.
        """
        with self._lock:
            if task_type:
                return dict(self._task_usage.get(task_type, {}))
            return dict(self._task_usage)

    def get_summary(self, days: int = 1) -> UsageSummary:
        """
        Produce a usage summary covering the previous N days.
        
        Parameters:
            days (int): Number of days to include in the summary; defaults to 1 (past 24 hours).
        
        Returns:
            UsageSummary: Summary populated with totals (requests, input tokens, output tokens, cost) and breakdowns by provider, model, and task type for the requested period. 
        """
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
        """
        Retrieve aggregate usage metrics and the number of stored records.
        
        Returns:
            stats (dict): Mapping with keys:
                - total_cost (float): Cumulative cost rounded to four decimal places.
                - total_tokens (int): Cumulative number of tokens recorded.
                - total_requests (int): Cumulative number of requests recorded.
                - records_count (int): Number of usage records currently stored.
        """
        with self._lock:
            return {
                "total_cost": round(self._total_cost, 4),
                "total_tokens": self._total_tokens,
                "total_requests": self._total_requests,
                "records_count": len(self._usage_records),
            }

    def flush(self) -> None:
        """
        Persist in-memory usage records to the configured storage file.
        
        Creates the storage directory if needed, atomically collects the current in-memory records under the internal lock, and writes them as a JSON array to `self.storage_path`. On failure, a warning is logged; no exception is raised.
        """
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
        """
        Stop the background flush loop and persist any in-memory usage records to storage.
        
        Performs a final flush of accumulated usage to the configured storage path and prevents further periodic flushes by stopping the background loop.
        """
        self._running = False
        self.flush()

    def export_csv(self, path: str) -> None:
        """
        Write all recorded token usage entries to a CSV file at the given filesystem path.
        
        Writes a CSV where columns are taken from the keys of a usage record dictionary; if there are no recorded entries an empty file is created. The operation overwrites any existing file at `path`.
        
        Parameters:
            path (str): Destination path for the CSV file.
        """
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
        """
        Initialize the rate limiter with per-minute and optional per-day limits and backoff/retry configuration.
        
        Parameters:
            requests_per_minute (int): Maximum allowed requests in any rolling 60-second window.
            tokens_per_minute (int): Maximum allowed token consumption in any rolling 60-second window.
            requests_per_day (int | None): Optional maximum requests per calendar day; None disables a daily request limit.
            tokens_per_day (int | None): Optional maximum tokens per calendar day; None disables a daily token limit.
            max_retries (int): Maximum number of retry attempts the limiter will allow when requests are throttled.
            base_delay (float): Base delay in seconds used to compute exponential backoff.
            max_delay (float): Upper bound in seconds for any backoff delay.
            jitter (bool): If True, apply random jitter to backoff delays to avoid synchronized retries.
        """
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
        """
        Ensure the request fits within the current per-minute request and token limits or raise a rate-limit error.
        
        Parameters:
            estimated_tokens (int): Estimated number of tokens this request will consume; used against the tokens-per-minute limit.
            provider (str): Identifier for the provider used in error context.
        
        Raises:
            RateLimitError: If accepting this request would exceed the requests-per-minute limit or the tokens-per-minute limit. The exception includes details such as `retry_after_seconds`, `limit_type`, `current_usage`, and `limit_value`.
        """
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
        """
        Determine how many seconds to wait before issuing a request with the given estimated token usage under the current per-minute limits.
        
        Parameters:
            estimated_tokens (int): Estimated number of tokens the upcoming request will consume.
            provider (str): Optional provider identifier used for scoping limits.
        
        Returns:
            float: Seconds to wait before the request can proceed; `0.0` if no wait is required.
        """
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
        """
        Report current per-minute request and token usage and remaining quotas.
        
        Returns:
            status (dict): A dictionary with two keys:
                - "requests_per_minute": dict with keys
                    - "used": number of requests in the last 60 seconds,
                    - "limit": configured requests-per-minute limit (`rpm`),
                    - "remaining": remaining requests available in the current minute (0 if none).
                - "tokens_per_minute": dict with keys
                    - "used": number of tokens consumed in the last 60 seconds,
                    - "limit": configured tokens-per-minute limit (`tpm`),
                    - "remaining": remaining tokens available in the current minute (0 if none).
        """
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
        """
        Reset internal request and token usage counters to their initial empty state.
        
        Clears recorded request timestamps and per-minute token usage.
        """
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
        """
        Initialize the adaptive rate limiter with base limits and backoff/recovery behavior.
        
        Parameters:
            base_rpm (int): Base requests-per-minute limit used as the starting and reference value.
            base_tpm (int): Base tokens-per-minute limit used as the starting and reference value.
            window_seconds (int): Time window (in seconds) used to evaluate recent errors for backoff decisions.
            backoff_factor (float): Fraction by which current limits are multiplied on each recorded error (values <1 reduce limits).
            recovery_factor (float): Multiplier applied to current limits on recovery to gradually increase capacity.
            max_multiplier (float): Maximum multiple of the base limits that recovery may grow current limits to.
        
        Notes:
            - Sets current limits to the provided base limits.
            - Initializes internal tracking for consecutive errors and the timestamp of the last error.
        """
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
        """
        Enforces adaptive backoff and blocks acquiring permissions when recent consecutive errors indicate a backoff period.
        
        If there have been one or more consecutive errors within the configured window, this method raises RateLimitError indicating how long the caller should wait before retrying.
        
        Parameters:
            estimated_tokens (int): Estimated number of tokens the request will consume; used to evaluate throttling (defaults to 0).
            provider (str): Identifier of the provider for which the acquire is requested (defaults to "default").
        
        Raises:
            RateLimitError: When the limiter is in an adaptive backoff state due to recent consecutive errors; the exception's `retry_after_seconds` indicates the suggested delay.
        """
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
        """
        Adjusts the limiter's state after a successful request to recover capacity.
        
        If there are recorded consecutive errors, resets the consecutive error count to 0 and increases
        current_rpm and current_tpm by the recovery factor, capped at base_rpm * max_multiplier and
        base_tpm * max_multiplier respectively.
        """
        async with self._lock:
            if self._consecutive_errors > 0:
                self._consecutive_errors = 0
                self.current_rpm = min(self.base_rpm * self.max_multiplier, self.current_rpm * self.recovery_factor)
                self.current_tpm = min(self.base_tpm * self.max_multiplier, self.current_tpm * self.recovery_factor)

    async def record_error(self, is_rate_limit: bool = False) -> None:
        """
        Record an error occurrence and apply backoff to current rate limits when appropriate.
        
        Parameters:
            is_rate_limit (bool): If True, treat the error as a rate-limit response and immediately apply backoff; otherwise backoff is applied only after the second consecutive error.
        
        """
        async with self._lock:
            self._consecutive_errors += 1
            self._last_error_time = time.time()

            if is_rate_limit or self._consecutive_errors > 1:
                self.current_rpm = max(1, int(self.current_rpm * self.backoff_factor))
                self.current_tpm = max(1000, int(self.current_tpm * self.backoff_factor))

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current rate limiter limits and recent error count.
        
        Returns:
            status (dict): Dictionary with the following keys:
                - current_rpm: Current requests-per-minute limit.
                - base_rpm: Configured base requests-per-minute limit.
                - current_tpm: Current tokens-per-minute limit.
                - base_tpm: Configured base tokens-per-minute limit.
                - consecutive_errors: Number of consecutive errors recorded.
        """
        return {
            "current_rpm": self.current_rpm,
            "base_rpm": self.base_rpm,
            "current_tpm": self.current_tpm,
            "base_tpm": self.base_tpm,
            "consecutive_errors": self._consecutive_errors,
        }

    def reset(self) -> None:
        """
        Reset adaptive limits and error state to defaults.
        
        Sets current RPM and TPM back to their base values and clears the consecutive error count and last error timestamp.
        """
        self.current_rpm = self.base_rpm
        self.current_tpm = self.base_tpm
        self._consecutive_errors = 0
        self._last_error_time = None