#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════"
echo "  Atomic Engine - Integration Test"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
HTTP_PORT=${ATOMIC_HTTP_PORT:-8080}
IPC_SOCKET=${ATOMIC_IPC_SOCKET:-/tmp/atomic_llm_agent.sock}

echo -e "${BLUE}Test Configuration:${NC}"
echo "  HTTP Port: $HTTP_PORT"
echo "  IPC Socket: $IPC_SOCKET"
echo ""

# test_endpoint tests an HTTP endpoint using the given method and optional JSON payload, prints colored PASS/FAIL and the response body, and returns 0 for HTTP 200 or 1 otherwise.
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    
    echo -ne "${YELLOW}Testing $name...${NC} "
    
    if [ -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" "$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$endpoint")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        echo "  Response: $body" | jq . 2>/dev/null || echo "  Response: $body"
        return 0
    else
        echo -e "${RED}✗ FAIL (HTTP $http_code)${NC}"
        echo "  Response: $body"
        return 1
    fi
}

# wait_for_service polls the given check command once per second for up to 30 seconds and exits with 0 if the check succeeds or 1 on timeout.
wait_for_service() {
    local service=$1
    local check_cmd=$2
    local max_wait=30
    local waited=0
    
    echo -ne "${YELLOW}Waiting for $service...${NC} "
    
    while [ $waited -lt $max_wait ]; do
        if eval "$check_cmd" &>/dev/null; then
            echo -e "${GREEN}✓ Ready${NC}"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        echo -ne "."
    done
    
    echo -e "${RED}✗ Timeout${NC}"
    return 1
}

echo "════════════════════════════════════════════════════════════"
echo "  Step 1: Check Prerequisites"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if orchestrator is built
if [ ! -f "build/atomic_orchestrator" ]; then
    echo -e "${RED}✗ Orchestrator not built${NC}"
    echo "  Run: ./build.sh"
    exit 1
fi
echo -e "${GREEN}✓ Orchestrator binary exists${NC}"

# Check if Python IPC server exists
if [ ! -f "../../app/llm/ipc_server.py" ]; then
    echo -e "${RED}✗ Python IPC server not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python IPC server exists${NC}"

# Check for required tools
for tool in curl jq; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}✗ $tool not found${NC}"
        echo "  Install: sudo apt-get install $tool"
        exit 1
    fi
done
echo -e "${GREEN}✓ Required tools available${NC}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 2: Start Services"
echo "════════════════════════════════════════════════════════════"
echo ""

# Start Python IPC server in background
echo -e "${BLUE}Starting Python LLM Agent...${NC}"
cd ../../
python3 -m app.llm.ipc_server &
PYTHON_PID=$!
cd src/cpp
echo "  PID: $PYTHON_PID"

# Wait for IPC socket
wait_for_service "IPC Socket" "[ -S $IPC_SOCKET ]"

# Start C++ orchestrator in background
echo -e "${BLUE}Starting C++ Orchestrator...${NC}"
./build/atomic_orchestrator > /tmp/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "  PID: $ORCHESTRATOR_PID"

# Wait for HTTP server
wait_for_service "HTTP Server" "curl -s http://localhost:$HTTP_PORT/health"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 3: Run API Tests"
echo "════════════════════════════════════════════════════════════"
echo ""

# Test health endpoint
test_endpoint "Health Check" "GET" "http://localhost:$HTTP_PORT/health"
echo ""

# Test metrics endpoint
test_endpoint "Metrics" "GET" "http://localhost:$HTTP_PORT/api/metrics"
echo ""

# Test status endpoint
test_endpoint "Status" "GET" "http://localhost:$HTTP_PORT/api/status"
echo ""

# Test generate endpoint (mocked response expected)
test_endpoint "Generate Request" "POST" "http://localhost:$HTTP_PORT/api/generate" \
    '{"prompt":"Create a simple Hello World program","metadata":{"language":"python"}}'
echo ""

echo "════════════════════════════════════════════════════════════"
echo "  Step 4: Cleanup"
echo "════════════════════════════════════════════════════════════"
echo ""

echo -e "${YELLOW}Stopping services...${NC}"

if [ ! -z "$ORCHESTRATOR_PID" ]; then
    kill $ORCHESTRATOR_PID 2>/dev/null || true
    echo -e "${GREEN}✓ C++ Orchestrator stopped${NC}"
fi

if [ ! -z "$PYTHON_PID" ]; then
    kill $PYTHON_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Python LLM Agent stopped${NC}"
fi

# Clean up socket
rm -f $IPC_SOCKET

echo ""
echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Integration Test Complete${NC}"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Logs available at: /tmp/orchestrator.log"
echo ""