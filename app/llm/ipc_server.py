#!/usr/bin/env python3
"""
IPC Server for C++ Orchestrator
Provides Unix socket interface for LLM requests from C++ orchestrator
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .client import LLMClient
from .config import LLMConfig, load_config
from .providers import LLMRequest

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class MessageType:
    """Message types for IPC protocol"""
    REQUEST = 0
    RESPONSE = 1
    STREAM_CHUNK = 2
    ERROR = 3
    HEALTH_CHECK = 4
    HEALTH_RESPONSE = 5


class RequestType:
    """Request types from C++ orchestrator"""
    PARSE_IDEA = 0
    GENERATE_CODE = 1
    GENERATE_SPEC = 2
    GENERATE_DOCS = 3
    RESEARCH = 4
    CREATE_BRAND = 5
    BUSINESS_PLAN = 6
    LAUNCH_CONFIG = 7


class LLMAgentIPCServer:
    """Unix socket server for C++ orchestrator communication"""
    
    def __init__(self, socket_path: str = "/tmp/atomic_llm_agent.sock"):
        """
        Initialize the IPC server and default internal state.
        
        Parameters:
            socket_path (str): Filesystem path for the Unix domain socket the server will bind to. Defaults to "/tmp/atomic_llm_agent.sock".
        """
        self.socket_path = socket_path
        self.config: Optional[LLMConfig] = None
        self.client: Optional[LLMClient] = None
        self.active_connections = 0
        
    async def initialize(self):
        """
        Load the LLM configuration and instantiate the LLM client for this server.
        
        Sets self.config to the loaded configuration and self.client to a newly created LLMClient. If initialization fails, the original exception is logged and re-raised.
        """
        try:
            logger.info("Loading LLM configuration...")
            self.config = load_config()
            
            logger.info(f"Initializing LLM client with provider: {self.config.default_provider}")
            self.client = LLMClient(self.config)
            
            logger.info("✓ LLM client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Serve a single client connection: read line-delimited JSON messages, dispatch them to process_message, and send back non-streaming responses.
        
        Reads messages from `reader` in a loop until EOF, parsing each line as JSON and calling `process_message(message, writer)`. If a non-streaming response is returned, it is sent back as a line-delimited JSON. On JSON parse errors an error response is sent. Tracks and logs active connection count and ensures the connection is closed and cleaned up on exit.
        
        Parameters:
            reader (asyncio.StreamReader): Asynchronous stream to read incoming data.
            writer (asyncio.StreamWriter): Asynchronous stream to write responses and stream chunks.
        """
        self.active_connections += 1
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection (total: {self.active_connections})")
        
        try:
            while True:
                # Read line-delimited JSON
                data = await reader.readline()
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode())
                    response = await self.process_message(message, writer)
                    
                    # Send response (non-streaming)
                    if response:
                        response_json = json.dumps(response) + '\n'
                        writer.write(response_json.encode())
                        await writer.drain()
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = self.create_error_response(
                        "parse_error",
                        f"Invalid JSON: {e}"
                    )
                    writer.write((json.dumps(error_response) + '\n').encode())
                    await writer.drain()
                    
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            
        finally:
            self.active_connections -= 1
            writer.close()
            await writer.wait_closed()
            logger.info(f"Connection closed (remaining: {self.active_connections})")
    
    async def process_message(
        self,
        message: Dict[str, Any],
        writer: asyncio.StreamWriter
    ) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming IPC message by routing it to the appropriate handler and producing an IPC response or streaming results.
        
        Parses the incoming `message` dict (expected keys: `"type"`, `"id"`, and optional `"payload"`). Supports `HEALTH_CHECK` (returns a health response), `REQUEST` (dispatches either a streaming request that writes chunks to `writer` or a non-streaming request that returns a response dict), and returns an error response for unknown types or on exceptions.
        
        Parameters:
            message (Dict[str, Any]): Incoming message with keys:
                - "type": message type constant
                - "id": message identifier
                - "payload" (optional): request payload containing fields such as "request_id", "request_type", "prompt", "stream", and "metadata".
            writer (asyncio.StreamWriter): Connection writer used to send streaming chunks back to the client for streaming requests.
        
        Returns:
            Dict[str, Any]: An IPC response message for health checks, non-streaming request results, or error responses.
            None: When the request is handled as a streaming request and responses are written directly to `writer`.
        """
        msg_type = message.get("type")
        msg_id = message.get("id")
        payload = message.get("payload", {})
        
        try:
            # Health check
            if msg_type == MessageType.HEALTH_CHECK:
                return self.create_health_response(msg_id)
            
            # LLM request
            elif msg_type == MessageType.REQUEST:
                request_id = payload["request_id"]
                request_type = payload.get("request_type", RequestType.GENERATE_CODE)
                prompt = payload["prompt"]
                stream = payload.get("stream", False)
                metadata = payload.get("metadata", {})
                
                logger.info(f"Processing request {request_id} (stream={stream})")
                
                if stream:
                    # Streaming response
                    await self.handle_streaming_request(
                        msg_id,
                        request_id,
                        prompt,
                        metadata,
                        writer
                    )
                    return None  # Already sent via writer
                else:
                    # Non-streaming response
                    return await self.handle_non_streaming_request(
                        msg_id,
                        request_id,
                        prompt,
                        metadata
                    )
            
            else:
                return self.create_error_response(
                    msg_id,
                    f"Unknown message type: {msg_type}"
                )
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self.create_error_response(msg_id, str(e))
    
    async def handle_streaming_request(
        self,
        msg_id: str,
        request_id: str,
        prompt: str,
        metadata: Dict[str, Any],
        writer: asyncio.StreamWriter
    ):
        """
        Stream LLM-generated chunks for a request and send them as line-delimited STREAM_CHUNK messages to the connected client.
        
        Sends one STREAM_CHUNK message per received chunk containing `request_id`, the chunk `delta`, `accumulated_content`, `chunk_index`, `is_final`, and `finish_reason`. If an exception occurs, sends an ERROR message with the `request_id` and error string.
        
        Parameters:
            msg_id (str): The IPC message identifier to correlate responses with the incoming message.
            request_id (str): The LLM request identifier used inside payloads to correlate chunks with the original request.
            prompt (str): The user prompt to send to the LLM.
            metadata (Dict[str, Any]): Additional metadata to include with the LLM request.
            writer (asyncio.StreamWriter): Stream writer used to send line-delimited JSON messages to the client.
        """
        try:
            # Create LLM request
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                metadata=metadata
            )
            
            # Stream responses
            accumulated_content = ""
            chunk_index = 0
            
            async for chunk in self.client.stream_generate(llm_request):
                accumulated_content = chunk.content
                
                # Send chunk to C++ client
                chunk_response = {
                    "id": msg_id,
                    "type": MessageType.STREAM_CHUNK,
                    "timestamp": int(time.time() * 1000),
                    "payload": {
                        "request_id": request_id,
                        "delta": chunk.delta,
                        "accumulated_content": accumulated_content,
                        "chunk_index": chunk_index,
                        "is_final": chunk.is_final,
                        "finish_reason": chunk.finish_reason or ""
                    }
                }
                
                writer.write((json.dumps(chunk_response) + '\n').encode())
                await writer.drain()
                
                chunk_index += 1
                
                if chunk.is_final:
                    break
            
            logger.info(f"✓ Streaming request {request_id} completed ({chunk_index} chunks)")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_response = {
                "id": msg_id,
                "type": MessageType.ERROR,
                "timestamp": int(time.time() * 1000),
                "payload": {
                    "request_id": request_id,
                    "error": str(e)
                }
            }
            writer.write((json.dumps(error_response) + '\n').encode())
            await writer.drain()
    
    async def handle_non_streaming_request(
        self,
        msg_id: str,
        request_id: str,
        prompt: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send the prompt to the configured LLM in non-streaming mode and return a complete IPC response message or an error message.
        
        On success, the returned dictionary is a MESSAGE_TYPE.RESPONSE envelope containing a payload with the final generated content, model/provider identifiers, token usage counts, and finalization flags. On failure, the returned dictionary is a MESSAGE_TYPE.ERROR envelope whose payload contains the request_id and an error string.
        
        Parameters:
            metadata (Dict[str, Any]): Optional provider-specific metadata or generation parameters to attach to the LLM request.
        
        Returns:
            Dict[str, Any]: A line-delimited IPC message dict with one of the following shapes:
                - Success (type == MessageType.RESPONSE):
                    {
                        "id": msg_id,
                        "type": MessageType.RESPONSE,
                        "timestamp": <ms since epoch>,
                        "payload": {
                            "request_id": request_id,
                            "content": <generated text>,
                            "model": <model identifier>,
                            "provider": <provider identifier>,
                            "input_tokens": <int>,
                            "output_tokens": <int>,
                            "is_final": True,
                            "error": ""
                        }
                    }
                - Error (type == MessageType.ERROR):
                    {
                        "id": msg_id,
                        "type": MessageType.ERROR,
                        "timestamp": <ms since epoch>,
                        "payload": {
                            "request_id": request_id,
                            "error": <error message string>
                        }
                    }
        """
        try:
            # Create LLM request
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                metadata=metadata
            )
            
            # Get response
            response = await self.client.generate(llm_request)
            
            logger.info(f"✓ Request {request_id} completed ({response.usage.total_tokens} tokens)")
            
            return {
                "id": msg_id,
                "type": MessageType.RESPONSE,
                "timestamp": int(time.time() * 1000),
                "payload": {
                    "request_id": request_id,
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "is_final": True,
                    "error": ""
                }
            }
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {
                "id": msg_id,
                "type": MessageType.ERROR,
                "timestamp": int(time.time() * 1000),
                "payload": {
                    "request_id": request_id,
                    "error": str(e)
                }
            }
    
    def create_health_response(self, msg_id: str) -> Dict[str, Any]:
        """
        Builds a HEALTH_RESPONSE IPC message summarizing server health.
        
        Returns:
            Dict[str, Any]: A HEALTH_RESPONSE message with keys:
                - `id`: the provided message id,
                - `type`: MessageType.HEALTH_RESPONSE,
                - `timestamp`: epoch milliseconds,
                - `payload`: dict containing `status` ("healthy"), `active_connections` (current count), and `provider` (configured provider or "unknown").
        """
        return {
            "id": msg_id,
            "type": MessageType.HEALTH_RESPONSE,
            "timestamp": int(time.time() * 1000),
            "payload": {
                "status": "healthy",
                "active_connections": self.active_connections,
                "provider": self.config.default_provider if self.config else "unknown"
            }
        }
    
    def create_error_response(self, msg_id: str, error: str) -> Dict[str, Any]:
        """
        Builds a structured ERROR IPC message for the given incoming message ID and error text.
        
        Parameters:
            msg_id (str): The original message identifier to include in the error response.
            error (str): Human-readable error message describing the failure.
        
        Returns:
            Dict[str, Any]: IPC message dict with keys:
                - "id": original message id,
                - "type": MessageType.ERROR,
                - "timestamp": epoch milliseconds,
                - "payload": {"error": error}
        """
        return {
            "id": msg_id,
            "type": MessageType.ERROR,
            "timestamp": int(time.time() * 1000),
            "payload": {
                "error": error
            }
        }
    
    async def start(self):
        """
        Start and run the Unix-domain socket IPC server that accepts LLM requests.
        
        Removes any existing socket file at the configured path, initializes the LLM client, binds an asyncio Unix-domain server to the socket path, logs startup details (socket path, provider, model count) and runs the server until shutdown.
        """
        # Remove old socket if exists
        socket_path = Path(self.socket_path)
        socket_path.unlink(missing_ok=True)
        
        # Initialize LLM client
        await self.initialize()
        
        # Start server
        server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path
        )
        
        logger.info("═" * 60)
        logger.info("  LLM Agent IPC Server Started")
        logger.info("═" * 60)
        logger.info(f"  Socket: {self.socket_path}")
        logger.info(f"  Provider: {self.config.default_provider}")
        logger.info(f"  Models: {len(self.config.providers)}")
        logger.info("═" * 60)
        logger.info("")
        logger.info("Waiting for connections from C++ orchestrator...")
        logger.info("Press Ctrl+C to shutdown")
        logger.info("")
        
        async with server:
            await server.serve_forever()


async def main():
    """
    Start the LLMAgent IPC server using an optional Unix socket path from command-line arguments.
    
    Parses sys.argv[1] as the socket path (defaults to /tmp/atomic_llm_agent.sock), creates and runs an LLMAgentIPCServer, removes the socket file on graceful shutdown or on error, and exits with status 1 for unhandled exceptions.
    """
    import sys
    
    socket_path = "/tmp/atomic_llm_agent.sock"
    
    if len(sys.argv) > 1:
        socket_path = sys.argv[1]
    
    server = LLMAgentIPCServer(socket_path)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\n\nShutdown requested, cleaning up...")
        Path(socket_path).unlink(missing_ok=True)
        logger.info("✓ Server stopped gracefully")
    except Exception as e:
        logger.error(f"Server error: {e}")
        Path(socket_path).unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())