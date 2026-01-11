cmake_minimum_required(VERSION 3.20)
project(AtomicOrchestrator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Find dependencies
find_package(Boost 1.75.0 REQUIRED COMPONENTS system thread filesystem json)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# Source files
set(ORCHESTRATOR_SOURCES
    orchestrator/orchestrator.cpp
    orchestrator/pipeline.cpp
    server/http_server.cpp
    ipc/python_agent_client.cpp
    sandbox/docker_controller.cpp
    queue/request_queue.cpp
    utils/logger.cpp
    utils/config.cpp
)

set(ORCHESTRATOR_HEADERS
    orchestrator/orchestrator.h
    orchestrator/pipeline.h
    orchestrator/metrics.h
    server/http_server.h
    server/request_handler.h
    ipc/python_agent_client.h
    ipc/message_protocol.h
    sandbox/docker_controller.h
    sandbox/resource_limits.h
    queue/request_queue.h
    queue/priority_queue.h
    utils/logger.h
    utils/config.h
    utils/helpers.h
)

# Library target
add_library(atomic_orchestrator_lib STATIC
    ${ORCHESTRATOR_SOURCES}
)

target_link_libraries(atomic_orchestrator_lib
    PUBLIC
        Boost::system
        Boost::thread
        Boost::filesystem
        Boost::json
        Threads::Threads
        OpenSSL::SSL
        OpenSSL::Crypto
)

target_include_directories(atomic_orchestrator_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Executable target
add_executable(atomic_orchestrator
    main.cpp
)

target_link_libraries(atomic_orchestrator
    PRIVATE
        atomic_orchestrator_lib
)

# Installation
install(TARGETS atomic_orchestrator
    RUNTIME DESTINATION bin
)

install(TARGETS atomic_orchestrator_lib
    ARCHIVE DESTINATION lib
)

install(FILES ${ORCHESTRATOR_HEADERS}
    DESTINATION include/atomic_orchestrator
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
