#!/usr/bin/env python3
"""
Streaming Module for LLM Responses
Implements SSE (Server-Sent Events) and real-time response handling
"""

import json
import asyncio
from typing import AsyncIterator, Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import uuid

from .providers import BaseProvider, LLMRequest, StreamChunk, TokenUsage
from .config import ModelConfig

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream state"""
    IDLE = "idle"
    STARTING = "starting"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamContext:
    """Streaming context for maintaining state across provider boundaries"""
    stream_id: str
    request: LLMRequest
    state: StreamState = StreamState.IDLE
    provider: str = ""
    model: str = ""
    accumulated_content: str = ""
    total_chunks: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    token_usage: Optional[TokenUsage] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, List[Callable]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "request": self.request.to_dict(),
            "state": self.state.value,
            "provider": self.provider,
            "model": self.model,
            "accumulated_content_length": len(self.accumulated_content),
            "total_chunks": self.total_chunks,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "error": self.error,
            "metadata": self.metadata,
        }


class StreamHandler:
    """Handles streaming responses from LLM providers"""

    def __init__(
        self,
        buffer_size: int = 10,
        flush_interval: float = 0.1,
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        # Active streams
        self._active_streams: Dict[str, StreamContext] = {}
        self._lock = asyncio.Lock()

    async def create_stream(
        self,
        request: LLMRequest,
        provider: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamContext:
        """Create a new streaming context"""
        context = StreamContext(
            stream_id=str(uuid.uuid4()),
            request=request,
            provider=provider,
            model=model,
            metadata=metadata or {},
        )

        async with self._lock:
            self._active_streams[context.stream_id] = context

        return context

    async def process_stream(
        self,
        provider: BaseProvider,
        request: LLMRequest,
        context: StreamContext,
        chunk_callback: Optional[Callable[[StreamChunk], None]] = None,
    ) -> StreamContext:
        """Process streaming response and update context"""
        context.state = StreamState.STARTING

        try:
            context.state = StreamState.STREAMING

            async for chunk in provider.stream(request):
                context.total_chunks += 1
                context.accumulated_content = chunk.content

                # Update token usage from final chunk
                if chunk.is_final and chunk.usage:
                    context.token_usage = chunk.usage

                # Call callback if provided
                if chunk_callback:
                    chunk_callback(chunk)

                # Fire registered callbacks
                await self._fire_callbacks("chunk", chunk, context)

            context.state = StreamState.COMPLETED
            context.end_time = datetime.now()

        except Exception as e:
            context.state = StreamState.ERROR
            context.error = str(e)
            context.end_time = datetime.now()
            logger.error(f"Stream error for {context.stream_id}: {e}")

        finally:
            # Clean up from active streams
            async with self._lock:
                self._active_streams.pop(context.stream_id, None)

        return context

    def register_callback(
        self,
        stream_id: str,
        event: str,
        callback: Callable,
    ) -> bool:
        """Register a callback for stream events"""
        if stream_id not in self._active_streams:
            return False

        self._active_streams[stream_id].callbacks.setdefault(event, []).append(callback)
        return True

    def unregister_callback(
        self,
        stream_id: str,
        event: str,
        callback: Callable,
    ) -> bool:
        """Unregister a callback"""
        if stream_id not in self._active_streams:
            return False

        callbacks = self._active_streams[stream_id].callbacks.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)
            return True
        return False

    async def _fire_callbacks(
        self,
        event: str,
        *args,
        **kwargs,
    ) -> None:
        """Fire callbacks for an event"""
        # This is simplified - in practice, you'd want to track context per event
        pass

    def get_stream(self, stream_id: str) -> Optional[StreamContext]:
        """Get a stream by ID"""
        return self._active_streams.get(stream_id)

    def get_active_streams(self) -> List[StreamContext]:
        """Get all active streams"""
        return list(self._active_streams.values())

    async def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream"""
        if stream_id not in self._active_streams:
            return False

        context = self._active_streams[stream_id]
        context.state = StreamState.CANCELLED
        context.end_time = datetime.now()

        async with self._lock:
            self._active_streams.pop(stream_id, None)

        return True

    async def pause_stream(self, stream_id: str) -> bool:
        """Pause a streaming context"""
        context = self._active_streams.get(stream_id)
        if not context or context.state != StreamState.STREAMING:
            return False

        context.state = StreamState.PAUSED
        return True

    async def resume_stream(self, stream_id: str) -> bool:
        """Resume a paused stream"""
        # This would require provider support for pause/resume
        context = self._active_streams.get(stream_id)
        if not context or context.state != StreamState.PAUSED:
            return False

        context.state = StreamState.STREAMING
        return True

    async def cleanup(self) -> None:
        """Clean up all active streams"""
        async with self._lock:
            for context in self._active_streams.values():
                context.state = StreamState.ERROR
                context.error = "Cleanup"
            self._active_streams.clear()


class SSEStreamHandler(StreamHandler):
    """SSE (Server-Sent Events) compatible stream handler"""

    def __init__(
        self,
        buffer_size: int = 10,
        flush_interval: float = 0.05,
        sse_content_type: str = "text/event-stream",
    ):
        super().__init__(buffer_size, flush_interval)
        self.sse_content_type = sse_content_type

    def format_sse_event(
        self,
        event: str,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> str:
        """Format data as SSE event"""
        lines = []

        if event_id:
            lines.append(f"id: {event_id}")

        if event:
            lines.append(f"event: {event}")

        # Serialize data and handle newlines
        json_data = json.dumps(data)
        for line in json_data.split('\n'):
            lines.append(f"data: {line}")

        lines.append("")  # Empty line to end event
        return '\n'.join(lines) + '\n'

    def format_chunk_event(
        self,
        chunk: StreamChunk,
        include_content: bool = True,
    ) -> str:
        """Format a stream chunk as SSE event"""
        data = {
            "type": "chunk",
            "chunk_index": chunk.chunk_index,
            "delta": chunk.delta if include_content else None,
            "content": chunk.content if include_content else None,
            "is_final": chunk.is_final,
            "finish_reason": chunk.finish_reason,
            "model": chunk.model,
            "provider": chunk.provider,
            "timestamp": chunk.timestamp.isoformat(),
        }

        if chunk.usage:
            data["usage"] = chunk.usage.to_dict()

        return self.format_sse_event("chunk", data, str(chunk.chunk_index))

    def format_start_event(
        self,
        context: StreamContext,
    ) -> str:
        """Format stream start as SSE event"""
        data = {
            "type": "start",
            "stream_id": context.stream_id,
            "model": context.model,
            "provider": context.provider,
            "timestamp": context.start_time.isoformat(),
        }

        return self.format_sse_event("start", data, context.stream_id)

    def format_end_event(
        self,
        context: StreamContext,
    ) -> str:
        """Format stream end as SSE event"""
        data = {
            "type": "end",
            "stream_id": context.stream_id,
            "state": context.state.value,
            "total_chunks": context.total_chunks,
            "content_length": len(context.accumulated_content),
            "timestamp": context.end_time.isoformat() if context.end_time else None,
        }

        if context.token_usage:
            data["usage"] = context.token_usage.to_dict()

        if context.error:
            data["error"] = context.error

        return self.format_sse_event("end", data, context.stream_id)

    def format_error_event(
        self,
        stream_id: str,
        error: str,
    ) -> str:
        """Format error as SSE event"""
        data = {
            "type": "error",
            "stream_id": stream_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        return self.format_sse_event("error", data, stream_id)

    async def stream_to_sse(
        self,
        provider: BaseProvider,
        request: LLMRequest,
        context: StreamContext,
        include_content: bool = True,
    ) -> AsyncIterator[str]:
        """Yield SSE formatted events from a stream"""
        # Send start event
        yield self.format_start_event(context)

        # Process stream
        async for chunk in provider.stream(request):
            yield self.format_chunk_event(chunk, include_content)

            if chunk.is_final:
                context.token_usage = chunk.usage
                break

        # Send end event
        yield self.format_end_event(context)

    def create_sse_response(
        self,
        events: AsyncIterator[str],
    ) -> Dict[str, Any]:
        """Create a response object for SSE"""
        return {
            "status_code": 200,
            "headers": {
                "Content-Type": self.sse_content_type,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
            "body": events,
        }


class StreamBuffer:
    """Buffers streaming chunks for efficient processing"""

    def __init__(
        self,
        max_size: int = 100,
        flush_interval: float = 0.1,
    ):
        self.max_size = max_size
        self.flush_interval = flush_interval

        self._buffer: List[StreamChunk] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        _last_flush_time: float = 0

    async def add(self, chunk: StreamChunk) -> None:
        """Add a chunk to the buffer"""
        async with self._lock:
            self._buffer.append(chunk)

            if len(self._buffer) >= self.max_size:
                await self.flush()

    async def flush(self) -> List[StreamChunk]:
        """Flush and return all buffered chunks"""
        async with self._lock:
            chunks = self._buffer.copy()
            self._buffer.clear()
            return chunks

    async def get_all(self) -> List[StreamChunk]:
        """Get all chunks without flushing"""
        async with self._lock:
            return self._buffer.copy()

    async def clear(self) -> None:
        """Clear the buffer"""
        async with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """Get current buffer size"""
        return len(self._buffer)

    async def close(self) -> None:
        """Close and cleanup"""
        await self.flush()


class ChunkAggregator:
    """Aggregates streaming chunks into complete responses"""

    def __init__(self):
        self._responses: Dict[str, List[StreamChunk]] = {}

    async def add_chunk(self, stream_id: str, chunk: StreamChunk) -> Optional[Dict[str, Any]]:
        """Add a chunk and return complete response if done"""
        if stream_id not in self._responses:
            self._responses[stream_id] = []

        self._responses[stream_id].append(chunk)

        if chunk.is_final:
            chunks = self._responses.pop(stream_id)
            return self._aggregate(stream_id, chunks)

        return None

    def _aggregate(
        self,
        stream_id: str,
        chunks: List[StreamChunk],
    ) -> Dict[str, Any]:
        """Aggregate chunks into a complete response"""
        content = "".join(chunk.content for chunk in chunks)
        last_chunk = chunks[-1]

        return {
            "stream_id": stream_id,
            "content": content,
            "model": last_chunk.model,
            "provider": last_chunk.provider,
            "finish_reason": last_chunk.finish_reason,
            "total_chunks": len(chunks),
            "usage": last_chunk.usage.to_dict() if last_chunk.usage else None,
            "start_time": chunks[0].timestamp.isoformat(),
            "end_time": last_chunk.timestamp.isoformat(),
        }

    def get_incomplete(self, stream_id: str) -> Optional[List[StreamChunk]]:
        """Get incomplete chunks for a stream"""
        return self._responses.get(stream_id)

    def remove(self, stream_id: str) -> None:
        """Remove a stream from aggregation"""
        self._responses.pop(stream_id, None)
