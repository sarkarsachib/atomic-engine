# C++ Orchestrator - Quick Reference

## Build & Run

```bash
# Build
cd src/cpp && ./build.sh

# Start Python LLM Agent (Terminal 1)
python3 -m app.llm.ipc_server

# Start C++ Orchestrator (Terminal 2)
./build/atomic_orchestrator

# Test (Terminal 3)
curl http://localhost:8080/health
```

## Environment Variables

```bash
export ATOMIC_HTTP_PORT=8080
export ATOMIC_IPC_SOCKET=/tmp/atomic_llm_agent.sock
export ATOMIC_LOG_LEVEL=INFO
export ATOMIC_MAX_CONCURRENT=10
```

## API Endpoints

### Generate Code
```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Create a Hello World program"}'
```

### Health Check
```bash
curl http://localhost:8080/health
```

### Metrics
```bash
curl http://localhost:8080/api/metrics
```

### Status
```bash
curl http://localhost:8080/api/status
```

## WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/stream');
ws.onopen = () => ws.send(JSON.stringify({prompt: "Hello"}));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

## File Structure

```
src/cpp/
├── main.cpp                    # Entry point
├── orchestrator/               # Core orchestration
│   ├── orchestrator.{h,cpp}    # Main coordinator
│   ├── pipeline.{h,cpp}        # Pipeline engine
│   └── metrics.h               # Metrics collection
├── server/                     # HTTP/WebSocket
│   ├── http_server.{h,cpp}     # Server implementation
│   └── request_handler.h       # Request routing
├── ipc/                        # Python communication
│   ├── python_agent_client.*   # IPC client
│   └── message_protocol.h      # Message format
├── sandbox/                    # Docker integration
│   ├── docker_controller.*     # Container control
│   └── resource_limits.h       # Resource constraints
├── queue/                      # Request management
│   └── request_queue.h         # Priority queue
└── utils/                      # Utilities
    ├── logger.h                # Logging
    ├── config.{h,cpp}          # Configuration
    └── helpers.h               # Helpers
```

## Common Tasks

### Add New Endpoint
```cpp
router_->add_route("GET", "/api/my_endpoint", 
    [](const HttpRequest& req) {
        HttpResponse res;
        res.set_json({{"data", "value"}});
        return res;
    }
);
```

### Add Pipeline Stage
```cpp
pipeline_->add_stage(PipelineStage::MY_STAGE, 
    [](PipelineContext& ctx) {
        PipelineResult result;
        result.stage = "my_stage";
        result.success = true;
        return result;
    }
);
```

### Docker Execution
```cpp
ExecutionResult result = docker_controller_->execute_in_container(
    "atomic-sandbox:latest",
    {"python", "script.py"},
    resource_limits
);
```

## Troubleshooting

### Connection Refused
```bash
# Check IPC socket
ls -la /tmp/atomic_llm_agent.sock

# Start Python agent first
python3 -m app.llm.ipc_server
```

### Build Errors
```bash
# Clean build
rm -rf build && ./build.sh

# Check dependencies
cmake --version
g++ --version
```

### High Memory
```bash
export ATOMIC_MEMORY_LIMIT_MB=256
export ATOMIC_MAX_CONCURRENT=5
```

## Performance Tuning

### High Throughput
```bash
export ATOMIC_MAX_CONCURRENT=20
export ATOMIC_THREADS=8
export ATOMIC_IPC_POOL_SIZE=8
```

### Low Latency
```bash
export ATOMIC_IPC_POOL_SIZE=16
export ATOMIC_MAX_CONCURRENT=30
```

## Logging Levels

```bash
export ATOMIC_LOG_LEVEL=DEBUG   # Verbose
export ATOMIC_LOG_LEVEL=INFO    # Normal (default)
export ATOMIC_LOG_LEVEL=WARNING # Warnings only
export ATOMIC_LOG_LEVEL=ERROR   # Errors only
```

## IPC Message Types

| Type | Value | Description |
|------|-------|-------------|
| REQUEST | 0 | LLM request |
| RESPONSE | 1 | Non-streaming response |
| STREAM_CHUNK | 2 | Streaming chunk |
| ERROR | 3 | Error message |
| HEALTH_CHECK | 4 | Health ping |
| HEALTH_RESPONSE | 5 | Health response |

## Resource Limits

```cpp
ResourceLimits limits;
limits.memory_bytes = 512 * 1024 * 1024;  // 512MB
limits.cpu_millicores = 1000;              // 1 CPU
limits.disk_bytes = 1024 * 1024 * 1024;   // 1GB
limits.timeout_seconds = 300;              // 5 min
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `Orchestrator` | Main coordinator |
| `Pipeline` | Stage execution engine |
| `HttpServer` | HTTP/WebSocket server |
| `PythonAgentClient` | IPC to Python |
| `DockerController` | Container management |
| `RequestQueue` | Priority queue |
| `MetricsCollector` | Metrics tracking |

## Documentation

- **README.md**: Getting started
- **INTEGRATION.md**: Integration guide
- **SUMMARY.md**: Implementation summary
- **docs/CPP_ORCHESTRATOR.md**: Full documentation

## Support

- Issues: GitHub Issues
- Logs: `/tmp/orchestrator.log`
- IPC: Check Python agent logs
