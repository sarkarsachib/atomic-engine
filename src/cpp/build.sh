#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building Atomic Engine C++ Orchestrator"
echo "════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# check_dependency checks whether a command exists in PATH, prints a colored success or error message, suggests an installation hint when missing, and returns 0 on success or 1 on failure.
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 not found${NC}"
        echo "  Install: $2"
        return 1
    else
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    fi
}

echo ""
echo "Checking dependencies..."
check_dependency "cmake" "sudo apt-get install cmake"
check_dependency "g++" "sudo apt-get install build-essential"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo ""
    echo -e "${YELLOW}Build directory exists. Clean build? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf "$BUILD_DIR"
        echo "✓ Cleaned build directory"
    fi
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Running CMake"
echo "════════════════════════════════════════════════════════════"

# Check if we should use Conan
USE_CONAN=false
if command -v conan &> /dev/null; then
    echo -e "${GREEN}✓ Conan found${NC}"
    echo "Use Conan for dependency management? (Y/n)"
    read -r response
    if [[ ! "$response" =~ ^([nN][oO]|[nN])$ ]]; then
        USE_CONAN=true
    fi
fi

if [ "$USE_CONAN" = true ]; then
    echo ""
    echo "Installing dependencies with Conan..."
    conan install .. --build=missing
    cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
else
    echo ""
    echo "Using system dependencies..."
    cmake ..
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Compiling"
echo "════════════════════════════════════════════════════════════"

# Get number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc)
fi

echo "Building with $CORES parallel jobs..."
make -j$CORES

echo ""
echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Build Complete${NC}"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Executable: $BUILD_DIR/atomic_orchestrator"
echo "Library:    $BUILD_DIR/libatomic_orchestrator_lib.a"
echo ""
echo "Run with: ./build/atomic_orchestrator"
echo ""