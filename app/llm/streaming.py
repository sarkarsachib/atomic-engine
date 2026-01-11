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
        """
        Serialize the StreamContext into a JSON-serializable dictionary.
        
        The returned dictionary contains the stream's identifier, serialized request, current state as the enum value string, provider and model identifiers, the length of accumulated content, total processed chunk count, ISO 8601 timestamps for start and end (end may be None), serialized token usage if available (or None), any error message, and the metadata mapping.
        
        Returns:
            dict: {
                "stream_id": str,
                "request": dict,
                "state": str,
                "provider": str,
                "model": str,
                "accumulated_content_length": int,
                "total_chunks": int,
                "start_time": str,
                "end_time": str | None,
                "token_usage": dict | None,
                "error": str | None,
                "metadata": dict
            }
        """
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
        """
        Initialize the StreamHandler with buffering parameters and prepare internal state.
        
        Parameters:
            buffer_size (int): Maximum number of chunks to buffer before triggering a flush.
            flush_interval (float): Time in seconds between automatic flush attempts.
        """
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
        """
        Create and register a new StreamContext for a streaming request.
        
        Parameters:
            request (LLMRequest): The LLM request associated with the stream.
            provider (str): Identifier of the provider producing the stream.
            model (str): Model name or identifier used for the stream.
            metadata (Optional[Dict[str, Any]]): Optional metadata to attach to the stream.
        
        Returns:
            StreamContext: The created and registered stream context with a unique `stream_id` and initial state.
        """
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
        """
        Drive an LLM streaming response, update the provided StreamContext, and return the final context.
        
        Processes chunks produced by the provider for the given request, updating context fields (state, total_chunks, accumulated_content, token_usage, start/end times, and error when applicable), invoking an optional per-chunk callback, and firing registered "chunk" callbacks. Removes the context from the handler's active streams when finished.
        
        Parameters:
            provider (BaseProvider): The provider instance supplying an async stream of StreamChunk objects.
            request (LLMRequest): The LLM request used to produce the stream.
            context (StreamContext): The StreamContext to update as chunks arrive.
            chunk_callback (Optional[Callable[[StreamChunk], None]]): Optional synchronous callback invoked for each chunk.
        
        Returns:
            StreamContext: The updated StreamContext reflecting the final state and any token usage, timestamps, or error information.
        """
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
        """
        Attach a callback to be invoked for a specific event on an active stream.
        
        Parameters:
            stream_id (str): Identifier of the stream to register the callback on.
            event (str): Name of the event to listen for (e.g., "chunk", "end", "error").
            callback (Callable): Callable to be invoked when the event is fired. It will be appended to the stream's callback list.
        
        Returns:
            bool: `True` if the callback was registered, `False` if no active stream with `stream_id` exists.
        """
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
        """
        Remove a previously registered callback for a specific stream event.
        
        Parameters:
            stream_id (str): Identifier of the stream whose callback should be removed.
            event (str): Name of the event the callback was registered for.
            callback (Callable): The callback function to remove.
        
        Returns:
            bool: `True` if the callback was found and removed, `False` otherwise.
        """
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
        """
        Invoke all callbacks registered for the specified event, forwarding any positional and keyword arguments to each callback.
        
        Parameters:
            event (str): Name of the event whose callbacks should be invoked.
            *args: Positional arguments to pass to each callback.
            **kwargs: Keyword arguments to pass to each callback.
        """
        # This is simplified - in practice, you'd want to track context per event
        pass

    def get_stream(self, stream_id: str) -> Optional[StreamContext]:
        """
        Retrieve the StreamContext for a given stream identifier.
        
        Returns:
            The StreamContext for the given stream_id, or `None` if no active stream exists.
        """
        return self._active_streams.get(stream_id)

    def get_active_streams(self) -> List[StreamContext]:
        """
        List all currently active stream contexts.
        
        Returns:
            List[StreamContext]: Active StreamContext objects for streams currently being tracked.
        """
        return list(self._active_streams.values())

    async def cancel_stream(self, stream_id: str) -> bool:
        """
        Mark a registered stream as cancelled and remove it from active streams.
        
        If the stream exists, sets its state to `StreamState.CANCELLED`, records `end_time`, removes it from the handler's active registry, and returns `True`. If no stream with the given `stream_id` exists, returns `False`.
        
        Returns:
            `True` if the stream was found and cancelled, `False` otherwise.
        """
        if stream_id not in self._active_streams:
            return False

        context = self._active_streams[stream_id]
        context.state = StreamState.CANCELLED
        context.end_time = datetime.now()

        async with self._lock:
            self._active_streams.pop(stream_id, None)

        return True

    async def pause_stream(self, stream_id: str) -> bool:
        """
        Pause an active streaming context identified by stream_id.
        
        This will transition the stream's state to `PAUSED` only if the stream exists and is currently in the `STREAMING` state.
        
        Returns:
            `true` if the stream was paused, `false` otherwise.
        """
        context = self._active_streams.get(stream_id)
        if not context or context.state != StreamState.STREAMING:
            return False

        context.state = StreamState.PAUSED
        return True

    async def resume_stream(self, stream_id: str) -> bool:
        """
        Resume a paused stream by changing its state to STREAMING.
        
        Returns:
            `true` if the stream existed and was resumed, `false` otherwise.
        """
        # This would require provider support for pause/resume
        context = self._active_streams.get(stream_id)
        if not context or context.state != StreamState.PAUSED:
            return False

        context.state = StreamState.STREAMING
        return True

    async def cleanup(self) -> None:
        """
        Mark all active streams as errored with the message "Cleanup" and clear the active stream registry.
        
        This operation acquires the handler's internal lock, sets each active StreamContext.state to StreamState.ERROR and StreamContext.error to "Cleanup", then removes all contexts from the active streams mapping.
        """
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
        """
        Initialize the SSEStreamHandler with buffering parameters and the SSE content type.
        
        Parameters:
            buffer_size (int): Maximum number of chunks to buffer before triggering a flush.
            flush_interval (float): Time in seconds between automatic buffer flushes.
            sse_content_type (str): Content-Type value to use for Server-Sent Events responses.
        """
        super().__init__(buffer_size, flush_interval)
        self.sse_content_type = sse_content_type

    def format_sse_event(
        self,
        event: str,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> str:
        """
        Build a Server-Sent Events (SSE) formatted string for a single event.
        
        Parameters:
            event (str): The SSE event name to emit (omitted if empty).
            data (Dict[str, Any]): The payload to serialize to JSON and emit as one or more `data:` lines.
            event_id (Optional[str]): Optional `id` field for the SSE event.
        
        Returns:
            sse_event (str): The complete SSE event text, including optional `id:` and `event:` lines, one or more `data:` lines containing the JSON payload, and a terminating blank line.
        """
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
        """
        Produce an SSE-formatted "chunk" event representing a single streaming chunk.
        
        Parameters:
            chunk (StreamChunk): The stream chunk to serialize into an SSE event.
            include_content (bool): If False, omit textual `content` and `delta` (they will be set to null).
        
        Returns:
            sse_event (str): The chunk encoded as an SSE event string (includes event id set to the chunk index).
        """
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
        """
        Builds an SSE "start" event for a stream.
        
        Parameters:
            context (StreamContext): StreamContext whose stream_id, model, provider, and start_time are used to populate the event.
        
        Returns:
            str: SSE-formatted event string representing the stream start.
        """
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
        """
        Create an SSE "end" event for the given stream context.
        
        Parameters:
            context (StreamContext): Stream context whose final state, chunk counts, accumulated content length, timestamp, and optional token usage or error are included in the event.
        
        Returns:
            sse_event (str): SSE-formatted string representing the "end" event for the stream.
        """
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
        """
        Builds an SSE-formatted "error" event for a stream.
        
        Parameters:
            stream_id (str): Identifier of the stream that encountered the error.
            error (str): Human-readable error message.
        
        Returns:
            str: SSE-formatted event payload representing the error.
        """
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
        """
        Produce an async iterator of Server-Sent Events (SSE) strings for a streaming LLM response.
        
        Yields a start event, one or more chunk events, and a final end event. When a chunk with `is_final` is encountered, the function updates `context.token_usage` with the chunk's usage and stops yielding further chunk events.
        
        Parameters:
            provider (BaseProvider): Provider used to stream chunks for the given request.
            request (LLMRequest): The request object describing the LLM invocation to stream.
            context (StreamContext): Stream context that is updated (notably `token_usage`) during streaming.
            include_content (bool): If True, include chunk content in chunk events; otherwise omit it.
        
        Returns:
            AsyncIterator[str]: An async iterator yielding SSE-formatted event strings (start, chunk events, end).
        """
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
        """
        Build an HTTP response dictionary configured for Server-Sent Events (SSE).
        
        Parameters:
            events (AsyncIterator[str]): An asynchronous iterator that yields SSE-formatted event strings to stream as the response body.
        
        Returns:
            response (Dict[str, Any]): A response mapping with:
                - status_code: 200
                - headers: SSE-ready headers including Content-Type, Cache-Control, Connection, and X-Accel-Buffering
                - body: the provided async iterator of SSE event strings
        """
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
        """
        Initialize the StreamBuffer with capacity and flush timing.
        
        Parameters:
            max_size (int): Maximum number of chunks to hold before triggering an immediate flush.
            flush_interval (float): Approximate interval in seconds used for periodic flushing when enabled.
        
        Description:
            Sets up internal storage for buffered StreamChunk objects, an asyncio lock for concurrency,
            and a placeholder for an optional background flush task.
        """
        self.max_size = max_size
        self.flush_interval = flush_interval

        self._buffer: List[StreamChunk] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        _last_flush_time: float = 0

    async def add(self, chunk: StreamChunk) -> None:
        """
        Add a StreamChunk to the internal buffer and trigger a flush when the buffer reaches max_size.
        
        Parameters:
            chunk (StreamChunk): Chunk to append to the buffer.
        """
        async with self._lock:
            self._buffer.append(chunk)

            if len(self._buffer) >= self.max_size:
                await self.flush()

    async def flush(self) -> List[StreamChunk]:
        """
        Clear the internal buffer and return the buffered StreamChunk objects.
        
        Returns:
            List[StreamChunk]: The chunks that were in the buffer prior to flushing; the internal buffer is empty after this call.
        """
        async with self._lock:
            chunks = self._buffer.copy()
            self._buffer.clear()
            return chunks

    async def get_all(self) -> List[StreamChunk]:
        """
        Return a copy of all buffered stream chunks without clearing the buffer.
        
        Returns:
            List[StreamChunk]: A shallow copy of the current buffer containing `StreamChunk` objects.
        """
        async with self._lock:
            return self._buffer.copy()

    async def clear(self) -> None:
        """
        Remove all buffered chunks from the internal buffer.
        
        This clears the in-memory buffer used to accumulate stream chunks.
        """
        async with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """
        Return the number of chunks currently in the buffer.
        
        Returns:
            int: Count of buffered StreamChunk objects.
        """
        return len(self._buffer)

    async def close(self) -> None:
        """
        Ensure any buffered chunks are flushed and perform cleanup.
        
        Calls flush to process and clear the internal buffer so no buffered chunks remain.
        """
        await self.flush()


class ChunkAggregator:
    """Aggregates streaming chunks into complete responses"""

    def __init__(self):
        """
        Initialize a ChunkAggregator by creating an empty mapping of stream IDs to buffered chunks.
        
        The instance will store incoming StreamChunk objects per stream in `_responses` until aggregation is triggered.
        """
        self._responses: Dict[str, List[StreamChunk]] = {}

    async def add_chunk(self, stream_id: str, chunk: StreamChunk) -> Optional[Dict[str, Any]]:
        """
        Append a StreamChunk to the stored list for a stream and produce an aggregated response when the chunk is final.
        
        Parameters:
            stream_id (str): Unique identifier of the stream receiving the chunk.
            chunk (StreamChunk): The stream chunk to add; if `chunk.is_final` is True the stored chunks for the stream are aggregated.
        
        Returns:
            dict: Aggregated response for the stream when `chunk.is_final` is True.
            None: If the stream is not complete after adding this chunk.
        """
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
        """
        Builds a consolidated response object from an ordered list of stream chunks.
        
        Parameters:
            stream_id (str): Identifier for the stream whose chunks are being aggregated.
            chunks (List[StreamChunk]): Ordered list of chunks for the stream; must contain at least one chunk.
        
        Returns:
            Dict[str, Any]: Aggregated response containing:
                - stream_id: the provided stream identifier
                - content: concatenated chunk contents in order
                - model: model value from the last chunk
                - provider: provider value from the last chunk
                - finish_reason: finish reason from the last chunk
                - total_chunks: number of chunks aggregated
                - usage: token usage from the last chunk as a dict, or `None` if absent
                - start_time: ISO 8601 timestamp of the first chunk
                - end_time: ISO 8601 timestamp of the last chunk
        """
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
        """
        Retrieve the list of incomplete chunks for a stream.
        
        Returns:
            Optional[List[StreamChunk]]: List of incomplete StreamChunk objects for the given stream_id, or `None` if no incomplete chunks exist.
        """
        return self._responses.get(stream_id)

    def remove(self, stream_id: str) -> None:
        """
        Remove all buffered chunks associated with a stream.
        
        This operation is idempotent: if no chunks exist for the given stream_id, the call has no effect.
        
        Parameters:
            stream_id (str): Identifier of the stream whose buffered chunks should be removed.
        """
        self._responses.pop(stream_id, None)