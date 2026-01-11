# C++ Orchestrator Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│  (Frontend, CLI, API Clients)                                   │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP/WebSocket
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              C++ Orchestrator (this module)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  HTTP    │  │ Pipeline │  │  Queue   │  │ Metrics  │       │
│  │  Server  │  │  Engine  │  │ Manager  │  │ Collector│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└────────────┬────────────────────────────────┬───────────────────┘
             │ Unix Socket (IPC)              │ Docker API
             ▼                                ▼
┌─────────────────────────┐      ┌───────────────────────────────┐
│  Python LLM Agent       │      │   Docker Sandbox              │
│  (app/llm/*.py)         │      │   (Code Execution)            │
│  ┌──────────────────┐   │      │   ┌────────────────────────┐ │
│  │  Router          │   │      │   │ Isolated Container     │ │
│  │  Streaming       │   │      │   │ - CPU/Memory Limits    │ │
│  │  Multi-Provider  │   │      │   │ - Network Isolation    │ │
│  │  Token Tracking  │   │      │   │ - Execution Timeout    │ │
│  └──────────────────┘   │      │   └────────────────────────┘ │
└─────────────────────────┘      └───────────────────────────────┘
```

## Integration Points

### 1. Python LLM Agent (IPC)

The C++ orchestrator communicates with Python LLM agents via Unix domain sockets.

#### Python Server Example

```python
#!/usr/bin/env python3
import asyncio
import json
import logging
from pathlib import Path

# Assumes you have the app/llm module
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.llm.router import Router

logger = logging.getLogger(__name__)

class LLMAgentServer:
    def __init__(self, socket_path="/tmp/atomic_llm_agent.sock"):
        self.socket_path = socket_path
        self.config = LLMConfig.from_env()
        self.client = LLMClient(self.config)
        
    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        logger.info(f"Connection from {addr}")
        
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                    
                message = json.loads(data.decode())
                response = await self.process_message(message)
                
                writer.write(json.dumps(response).encode() + b'\n')
                await writer.drain()
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def process_message(self, message):
        msg_type = message.get("type")
        payload = message.get("payload", {})
        
        if msg_type == 4:  # HEALTH_CHECK
            return {
                "id": message["id"],
                "type": 5,  # HEALTH_RESPONSE
                "timestamp": int(time.time() * 1000),
                "payload": {"status": "healthy"}
            }
        
        elif msg_type == 0:  # REQUEST
            request_id = payload["request_id"]
            prompt = payload["prompt"]
            stream = payload.get("stream", False)
            
            if stream:
                # Streaming response
                async for chunk in self.client.stream_generate(prompt):
                    yield {
                        "id": message["id"],
                        "type": 2,  # STREAM_CHUNK
                        "timestamp": int(time.time() * 1000),
                        "payload": {
                            "request_id": request_id,
                            "delta": chunk.delta,
                            "accumulated_content": chunk.content,
                            "chunk_index": chunk.chunk_index,
                            "is_final": chunk.is_final
                        }
                    }
            else:
                # Non-streaming response
                response = await self.client.generate(prompt)
                return {
                    "id": message["id"],
                    "type": 1,  # RESPONSE
                    "timestamp": int(time.time() * 1000),
                    "payload": {
                        "request_id": request_id,
                        "content": response.content,
                        "model": response.model,
                        "provider": response.provider,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "is_final": True
                    }
                }
    
    async def start(self):
        # Remove old socket if exists
        Path(self.socket_path).unlink(missing_ok=True)
        
        server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path
        )
        
        logger.info(f"LLM Agent listening on {self.socket_path}")
        
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = LLMAgentServer()
    asyncio.run(server.start())
```

#### Starting the Python Server

```bash
# Terminal 1: Start Python LLM Agent
cd /path/to/atomic-engine
python3 python_llm_server.py

# Terminal 2: Start C++ Orchestrator
cd src/cpp
./build/atomic_orchestrator
```

### 2. Docker Sandbox Integration

The orchestrator can execute generated code in isolated Docker containers.

#### Creating the Sandbox Image

```dockerfile
# Dockerfile.sandbox
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN chmod 755 /workspace

# Install common dependencies
RUN pip install --no-cache-dir requests numpy pandas

# Create non-root user
RUN useradd -m -u 1000 sandbox
USER sandbox

CMD ["/bin/bash"]
```

Build the image:
```bash
docker build -t atomic-sandbox:latest -f Dockerfile.sandbox .
```

#### Using the Sandbox from C++

```cpp
// Example usage in orchestrator
sandbox::ResourceLimits limits;
limits.memory_bytes = 512 * 1024 * 1024;  // 512MB
limits.cpu_millicores = 1000;              // 1 CPU
limits.timeout_seconds = 60;

auto result = docker_controller_->execute_in_container(
    "atomic-sandbox:latest",
    {"python", "-c", "print('Hello from sandbox!')"},
    limits
);

if (result.success) {
    LOG_INFO("Sandbox output: ", result.stdout_output);
} else {
    LOG_ERROR("Sandbox error: ", result.error);
}
```

### 3. Frontend Integration

#### REST API Example

```javascript
// JavaScript client
async function generateCode(prompt) {
    const response = await fetch('http://localhost:8080/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            prompt: prompt,
            metadata: {
                language: 'python',
                framework: 'flask'
            }
        })
    });
    
    const result = await response.json();
    return result;
}

// Usage
const result = await generateCode('Create a REST API for todo management');
console.log(result.content);
```

#### WebSocket Streaming Example

```javascript
// WebSocket client
const ws = new WebSocket('ws://localhost:8080/ws/stream');

ws.onopen = () => {
    ws.send(JSON.stringify({
        prompt: 'Build a real-time chat application',
        metadata: { stream: true }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'chunk') {
        console.log('Received chunk:', data.delta);
        updateUI(data.content);
    } else if (data.type === 'error') {
        console.error('Error:', data.error);
    }
};
```

### 4. Downstream Module Integration

The orchestrator feeds into downstream generation modules:

#### Code Generator (Forge)
```cpp
// In generate_stage()
ipc::LLMRequest request;
request.request_type = ipc::RequestType::GENERATE_CODE;
request.prompt = ctx.original_prompt;
auto response = llm_client_->send_request(request);
```

#### Spec Generator (BlackBox)
```cpp
// In parse_stage()
ipc::LLMRequest request;
request.request_type = ipc::RequestType::GENERATE_SPEC;
request.prompt = "Generate specification for: " + ctx.original_prompt;
auto response = llm_client_->send_request(request);
```

#### Documentation (Script)
```cpp
ipc::LLMRequest request;
request.request_type = ipc::RequestType::GENERATE_DOCS;
request.metadata["code"] = generated_code;
auto response = llm_client_->send_request(request);
```

## Performance Tuning

### Connection Pooling

```bash
# Increase IPC connection pool
export ATOMIC_IPC_POOL_SIZE=8
```

### Concurrent Requests

```bash
# Allow more concurrent processing
export ATOMIC_MAX_CONCURRENT=20
export ATOMIC_THREADS=8
```

### Queue Management

```bash
# Larger queue for burst traffic
export ATOMIC_MAX_QUEUE_SIZE=2000
```

### Resource Limits

```bash
# Adjust sandbox limits
export ATOMIC_MEMORY_LIMIT_MB=1024
export ATOMIC_CPU_LIMIT_MILLICORES=2000
```

## Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8080/api/metrics
```

Response:
```json
{
  "total_requests": 1523,
  "successful_requests": 1495,
  "failed_requests": 28,
  "average_latency_ms": 234.5,
  "success_rate": 0.98,
  "active_pipelines": 3,
  "queue_size": 5,
  "uptime_ms": 3600000
}
```

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "llm_agent": "healthy",
  "docker": "available",
  "uptime_ms": 3600000
}
```

## Troubleshooting

### Connection Issues

```bash
# Check if Python agent is listening
ls -la /tmp/atomic_llm_agent.sock

# Test connection
echo '{"id":"test","type":4,"timestamp":0,"payload":{}}' | nc -U /tmp/atomic_llm_agent.sock
```

### Docker Issues

```bash
# Verify Docker is running
docker ps

# Check image exists
docker images | grep atomic-sandbox

# Test sandbox manually
docker run --rm atomic-sandbox:latest python -c "print('test')"
```

### Build Issues

```bash
# Clean build
cd src/cpp
rm -rf build
./build.sh

# Check dependencies
cmake --version
g++ --version
pkg-config --modversion boost
```

## Production Deployment

### Systemd Service

```ini
# /etc/systemd/system/atomic-orchestrator.service
[Unit]
Description=Atomic Engine C++ Orchestrator
After=network.target docker.service

[Service]
Type=simple
User=atomic
WorkingDirectory=/opt/atomic-engine/src/cpp
ExecStart=/opt/atomic-engine/src/cpp/build/atomic_orchestrator
Restart=always
RestartSec=10
Environment="ATOMIC_LOG_LEVEL=INFO"
Environment="ATOMIC_HTTP_PORT=8080"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable atomic-orchestrator
sudo systemctl start atomic-orchestrator
sudo systemctl status atomic-orchestrator
```

### Docker Compose

```yaml
version: '3.8'

services:
  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    ports:
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./artifacts:/tmp/atomic_artifacts
    environment:
      - ATOMIC_LOG_LEVEL=INFO
      - ATOMIC_IPC_SOCKET=/tmp/atomic_llm_agent.sock
    depends_on:
      - llm-agent
  
  llm-agent:
    build:
      context: .
      dockerfile: Dockerfile.llm
    volumes:
      - ./app:/app
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

## Next Steps

1. **Implement Python LLM Server**: Create the Unix socket server for LLM communication
2. **Build Sandbox Image**: Create Docker image for code execution
3. **Test Integration**: Run end-to-end tests with all components
4. **Add Authentication**: Implement API key/JWT authentication
5. **Setup Monitoring**: Deploy Prometheus/Grafana for metrics
6. **Load Testing**: Benchmark with realistic workloads
