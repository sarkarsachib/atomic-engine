# Atomic Engine - C++ Orchestrator

High-performance C++ orchestration core for the Atomic Engine hybrid LLM system.

## Architecture

The C++ orchestrator is the central nervous system that:
- Routes HTTP/WebSocket requests from clients
- Manages multi-stage pipelines (Parse → Generate → Package → Export)
- Communicates with Python LLM agents via IPC
- Controls Docker sandboxes for code execution
- Handles request queueing and load balancing
- Collects metrics and manages resources

## Components

### Orchestrator
- **orchestrator.{h,cpp}** - Main coordinator class
- **pipeline.{h,cpp}** - Multi-stage pipeline state machine
- **metrics.h** - Performance tracking and monitoring

### Server
- **http_server.{h,cpp}** - HTTP + WebSocket server (Boost.Beast)
- **request_handler.h** - Request routing and response handling

### IPC
- **python_agent_client.{h,cpp}** - Unix socket client for Python LLM agents
- **message_protocol.h** - JSON message format definitions

### Sandbox
- **docker_controller.{h,cpp}** - Docker container lifecycle management
- **resource_limits.h** - CPU/memory/disk constraints

### Queue
- **request_queue.h** - Thread-safe priority queue
- **priority_queue.h** - Priority levels for requests

### Utils
- **logger.h** - Thread-safe logging
- **config.{h,cpp}** - Configuration management
- **helpers.h** - Utility functions

## Building

### Prerequisites

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev libssl-dev

# Or use Conan (recommended)
pip install conan
```

### Build Steps

```bash
cd src/cpp
mkdir -p build && cd build

# Option 1: Without Conan
cmake ..
make -j$(nproc)

# Option 2: With Conan (recommended)
conan install .. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
make -j$(nproc)
```

### Build Output

```
build/
  atomic_orchestrator          # Main executable
  libatomic_orchestrator_lib.a # Static library
```

## Running

```bash
# Set environment variables
export ATOMIC_HOST=0.0.0.0
export ATOMIC_HTTP_PORT=8080
export ATOMIC_IPC_SOCKET=/tmp/atomic_llm_agent.sock
export ATOMIC_LOG_LEVEL=INFO

# Run the orchestrator
./build/atomic_orchestrator
```

## Configuration

Configuration can be provided via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ATOMIC_HOST` | 0.0.0.0 | Server bind address |
| `ATOMIC_HTTP_PORT` | 8080 | HTTP server port |
| `ATOMIC_WS_PORT` | 8081 | WebSocket server port |
| `ATOMIC_IPC_SOCKET` | /tmp/atomic_llm_agent.sock | Python agent socket path |
| `ATOMIC_LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `ATOMIC_MAX_QUEUE_SIZE` | 1000 | Maximum queue size |
| `ATOMIC_MAX_CONCURRENT` | 10 | Max concurrent requests |
| `ATOMIC_SANDBOX_IMAGE` | atomic-sandbox:latest | Docker sandbox image |
| `ATOMIC_MEMORY_LIMIT_MB` | 512 | Container memory limit |

## API Endpoints

### HTTP REST API

**POST /api/generate**
```json
{
  "prompt": "Build a REST API for user management",
  "metadata": {
    "language": "typescript",
    "framework": "express"
  }
}
```

Response:
```json
{
  "success": true,
  "request_id": "uuid",
  "pipeline_id": "uuid",
  "content": "...",
  "artifacts": {...},
  "processing_time_ms": 1234
}
```

**GET /health**
```json
{
  "status": "ok",
  "llm_agent": "healthy",
  "docker": "available",
  "uptime_ms": 123456
}
```

**GET /api/metrics**
```json
{
  "total_requests": 100,
  "successful_requests": 95,
  "average_latency_ms": 245.3,
  "active_pipelines": 3,
  "queue_size": 5
}
```

### WebSocket Streaming

Connect to `ws://localhost:8080/ws/stream`

Send:
```json
{
  "prompt": "Create a Python web scraper",
  "metadata": {"stream": true}
}
```

Receive:
```json
{"type": "chunk", "delta": "import ", "content": "import ", "is_final": false}
{"type": "chunk", "delta": "requests", "content": "import requests", "is_final": false}
...
{"type": "chunk", "delta": "", "content": "...", "is_final": true}
```

## Integration with Python LLM Layer

The C++ orchestrator communicates with Python LLM agents via Unix domain sockets using a JSON message protocol:

```
C++ Orchestrator  <--Unix Socket-->  Python LLM Agent
    (this code)                      (app/llm/*.py)
```

### Message Protocol

**Request Format:**
```json
{
  "id": "message-uuid",
  "type": 0,
  "timestamp": 1234567890,
  "payload": {
    "request_id": "request-uuid",
    "request_type": 1,
    "prompt": "...",
    "stream": true,
    "max_tokens": 4096,
    "temperature": 0.7
  }
}
```

**Response Format:**
```json
{
  "id": "message-uuid",
  "type": 1,
  "timestamp": 1234567890,
  "payload": {
    "request_id": "request-uuid",
    "content": "...",
    "model": "gpt-4",
    "provider": "openai",
    "is_final": true
  }
}
```

## Performance Characteristics

- **Request Latency**: <100ms (excluding LLM inference)
- **Concurrent Requests**: 10+ simultaneous
- **Memory Footprint**: <500MB for orchestrator
- **Container Startup**: <2s per sandbox
- **Throughput**: 100+ requests/minute

## Docker Integration

The orchestrator can execute generated code in isolated Docker containers:

```cpp
ExecutionResult result = docker_controller->execute_in_container(
    "atomic-sandbox:latest",
    {"python", "generated_script.py"},
    resource_limits
);
```

Resource limits:
- Memory: 512MB
- CPU: 1 core
- Disk: 1GB
- Network: Isolated
- Execution timeout: 5 minutes

## Development

### Code Style
- C++17 standard
- Google C++ Style Guide
- RAII patterns
- Exception-based error handling

### Testing

```bash
cd build
ctest --verbose
```

### Debugging

```bash
# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
make

# Run with AddressSanitizer
./atomic_orchestrator
```

## License

MIT License - See root LICENSE file
