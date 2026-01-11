# C++ Orchestrator Documentation

## Overview

The C++ Orchestrator is the high-performance central nervous system of the Atomic Engine. It provides:

- **HTTP & WebSocket Server**: REST API and real-time streaming
- **Pipeline Orchestration**: Multi-stage execution flow
- **IPC Bridge**: Communication with Python LLM agents
- **Docker Sandbox**: Isolated code execution
- **Request Queue**: Priority-based load management
- **Metrics Collection**: Performance monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     C++ Orchestrator Core                       │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │   HTTP/WS    │   │   Pipeline   │   │    Queue     │      │
│  │    Server    │──▶│  Coordinator │──▶│   Manager    │      │
│  └──────────────┘   └──────┬───────┘   └──────────────┘      │
│                            │                                    │
│                            ▼                                    │
│         ┌─────────────────────────────────────┐               │
│         │     IPC to Python LLM Agents        │               │
│         │  (Unix Socket + JSON Protocol)      │               │
│         └─────────────────────────────────────┘               │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   Docker Sandbox       │
                 │  (Code Execution)      │
                 └────────────────────────┘
```

## Components

### 1. HTTP/WebSocket Server (`server/`)

**Files**: `http_server.{h,cpp}`, `request_handler.h`

Provides:
- REST API endpoints (POST /api/generate, GET /health, etc.)
- WebSocket streaming (/ws/stream)
- CORS support
- JSON request/response handling
- Concurrent connection management

**Key Features**:
- Built on Boost.Beast
- Multi-threaded request handling
- Automatic CORS headers
- Graceful shutdown

### 2. Pipeline Orchestrator (`orchestrator/`)

**Files**: `pipeline.{h,cpp}`, `orchestrator.{h,cpp}`, `metrics.h`

Implements:
- Multi-stage pipeline execution
- State machine (IDLE → PARSING → GENERATING → PACKAGING → EXPORTING)
- Progress tracking
- Error recovery
- Metrics collection

**Pipeline Stages**:
1. **PARSING**: Parse user intent and extract requirements
2. **GENERATING**: Generate code/content via LLM
3. **PACKAGING**: Bundle artifacts and metadata
4. **EXPORTING**: Export to file system or Git

### 3. IPC Bridge (`ipc/`)

**Files**: `python_agent_client.{h,cpp}`, `message_protocol.h`

Provides:
- Unix socket client for Python LLM agents
- Connection pooling (configurable size)
- Request/response marshaling (JSON)
- Streaming support
- Health checking
- Automatic reconnection

**Message Protocol**:
```cpp
enum class MessageType {
    REQUEST,        // LLM request from C++
    RESPONSE,       // Non-streaming response
    STREAM_CHUNK,   // Streaming chunk
    ERROR,          // Error message
    HEALTH_CHECK,   // Health check ping
    HEALTH_RESPONSE // Health check response
};
```

### 4. Docker Sandbox (`sandbox/`)

**Files**: `docker_controller.{h,cpp}`, `resource_limits.h`

Features:
- Container lifecycle management
- Resource limits (CPU, memory, disk)
- Network isolation
- Execution timeout
- Output capture (stdout/stderr)
- Automatic cleanup

**Resource Limits**:
```cpp
struct ResourceLimits {
    int64_t memory_bytes = 512MB;
    int64_t cpu_millicores = 1000;  // 1 CPU
    int64_t disk_bytes = 1GB;
    int timeout_seconds = 300;
    int max_processes = 100;
};
```

### 5. Request Queue (`queue/`)

**Files**: `request_queue.h`, `priority_queue.h`

Implements:
- Thread-safe priority queue
- Priority levels (LOW, NORMAL, HIGH)
- Backpressure handling
- Worker thread pool
- Concurrent request limiting

**Usage**:
```cpp
queue::RequestQueue<GenerateRequest> queue(
    max_size = 1000,
    max_concurrent = 10
);

queue.enqueue(request, Priority::HIGH);
```

### 6. Utilities (`utils/`)

**Files**: `logger.h`, `config.{h,cpp}`, `helpers.h`

Provides:
- Thread-safe logging
- Environment-based configuration
- String utilities
- UUID generation
- Timestamp helpers

## API Reference

### REST Endpoints

#### POST /api/generate
Generate code/content from prompt.

**Request**:
```json
{
  "prompt": "Create a REST API for user management",
  "metadata": {
    "language": "typescript",
    "framework": "express"
  }
}
```

**Response**:
```json
{
  "success": true,
  "request_id": "abc123",
  "pipeline_id": "def456",
  "content": "...",
  "artifacts": {...},
  "processing_time_ms": 1234
}
```

#### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "llm_agent": "healthy",
  "docker": "available",
  "uptime_ms": 123456
}
```

#### GET /api/metrics
Performance metrics.

**Response**:
```json
{
  "total_requests": 100,
  "successful_requests": 95,
  "average_latency_ms": 234.5,
  "active_pipelines": 3,
  "queue_size": 5
}
```

#### GET /api/status
System status.

**Response**:
```json
{
  "running": true,
  "active_pipelines": 3,
  "queue_size": 5
}
```

### WebSocket Protocol

**Endpoint**: `ws://localhost:8080/ws/stream`

**Client → Server**:
```json
{
  "prompt": "Build a REST API",
  "metadata": {"stream": true}
}
```

**Server → Client** (streaming chunks):
```json
{"type": "chunk", "delta": "import ", "content": "import ", "is_final": false}
{"type": "chunk", "delta": "express", "content": "import express", "is_final": false}
...
{"type": "chunk", "delta": "", "content": "...", "is_final": true}
```

**Error**:
```json
{"type": "error", "error": "Error message"}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATOMIC_HOST` | 0.0.0.0 | Server bind address |
| `ATOMIC_HTTP_PORT` | 8080 | HTTP port |
| `ATOMIC_WS_PORT` | 8081 | WebSocket port |
| `ATOMIC_THREADS` | 4 | Worker threads |
| `ATOMIC_IPC_SOCKET` | /tmp/atomic_llm_agent.sock | IPC socket path |
| `ATOMIC_IPC_POOL_SIZE` | 4 | IPC connection pool |
| `ATOMIC_SANDBOX_IMAGE` | atomic-sandbox:latest | Docker image |
| `ATOMIC_MEMORY_LIMIT_MB` | 512 | Container memory |
| `ATOMIC_MAX_QUEUE_SIZE` | 1000 | Queue capacity |
| `ATOMIC_MAX_CONCURRENT` | 10 | Max concurrent requests |
| `ATOMIC_LOG_LEVEL` | INFO | Log level |

### Configuration File

```cpp
Config config;
config.server.host = "0.0.0.0";
config.server.http_port = 8080;
config.ipc.socket_path = "/tmp/atomic_llm_agent.sock";
config.sandbox.memory_limit_mb = 512;
```

## Performance

### Benchmarks

- **Request Latency**: <100ms (excluding LLM inference)
- **Throughput**: 100+ requests/minute
- **Concurrent Requests**: 10+ simultaneous
- **Memory Footprint**: <500MB
- **Container Startup**: <2s

### Optimization Tips

1. **Increase IPC Pool**: More connections for LLM agents
   ```bash
   export ATOMIC_IPC_POOL_SIZE=8
   ```

2. **More Workers**: Higher concurrency
   ```bash
   export ATOMIC_MAX_CONCURRENT=20
   export ATOMIC_THREADS=8
   ```

3. **Larger Queue**: Handle traffic bursts
   ```bash
   export ATOMIC_MAX_QUEUE_SIZE=2000
   ```

## Development

### Building

```bash
cd src/cpp
./build.sh
```

### Testing

```bash
# Run integration tests
./test_integration.sh

# Manual testing
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello World in Python"}'
```

### Debugging

```bash
# Build with debug symbols
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
make

# Run with verbose logging
export ATOMIC_LOG_LEVEL=DEBUG
./atomic_orchestrator
```

### Adding New Pipeline Stages

```cpp
// Define stage handler
PipelineResult my_stage(PipelineContext& ctx) {
    PipelineResult result;
    result.stage = "my_stage";
    
    // Do work...
    result.success = true;
    result.content = "Stage output";
    
    return result;
}

// Register stage
pipeline_->add_stage(PipelineStage::MY_STAGE, [this](auto& ctx) {
    return my_stage(ctx);
});
```

### Adding New Endpoints

```cpp
router_->add_route("GET", "/api/my_endpoint", 
    [this](const server::HttpRequest& req) {
        server::HttpResponse response;
        
        boost::json::object obj;
        obj["data"] = "value";
        response.set_json(obj);
        
        return response;
    }
);
```

## Deployment

### Systemd Service

```ini
[Unit]
Description=Atomic C++ Orchestrator
After=network.target docker.service

[Service]
Type=simple
User=atomic
WorkingDirectory=/opt/atomic-engine/src/cpp
ExecStart=/opt/atomic-engine/src/cpp/build/atomic_orchestrator
Restart=always
Environment="ATOMIC_LOG_LEVEL=INFO"

[Install]
WantedBy=multi-user.target
```

### Docker Compose

```yaml
services:
  orchestrator:
    build:
      context: src/cpp
    ports:
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - ATOMIC_LOG_LEVEL=INFO
```

## Troubleshooting

### Connection Refused

```bash
# Check if IPC socket exists
ls -la /tmp/atomic_llm_agent.sock

# Start Python agent first
python3 -m app.llm.ipc_server
```

### Docker Not Available

```bash
# Verify Docker
docker ps

# Check permissions
sudo usermod -aG docker $USER
```

### High Memory Usage

```bash
# Reduce limits
export ATOMIC_MEMORY_LIMIT_MB=256
export ATOMIC_MAX_CONCURRENT=5
```

## Security

### Best Practices

1. **Network Isolation**: Sandboxes have no network access
2. **Resource Limits**: CPU/memory/disk constraints enforced
3. **Input Validation**: All JSON inputs validated
4. **Read-Only Containers**: Docker containers run read-only
5. **Non-Root User**: Containers run as non-root user

### Production Hardening

```bash
# Enable firewall
sudo ufw allow 8080/tcp

# Use TLS proxy (nginx/caddy)
# Implement rate limiting
# Add authentication middleware
```

## Contributing

### Code Style

- C++17 standard
- Google C++ Style Guide
- RAII for resource management
- Exception-based error handling

### Pull Request Checklist

- [ ] Code compiles without warnings
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] Memory leaks checked (valgrind)

## License

MIT License - See LICENSE file

## Support

- GitHub Issues: https://github.com/atomic-engine/issues
- Documentation: /docs/
- Examples: /examples/
