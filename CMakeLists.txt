cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit 11.4)

set(CUDNN_DIR "" CACHE STRING "")

option(BUILD_BENCHMARKS "Build the resnet50 benchmark" OFF)
option(BUILD_SHMEM "Build the resnet50 benchmark" ON)

option(SPDLOG_LEVEL_INFO "Set the spdlog level to info" OFF)
option(SPDLOG_LEVEL_DEBUG "Set the spdlog level to debug" OFF)
option(SPDLOG_LEVEL_TRACE "Set the spdlog level to trace" OFF)
option(SPDLOG_LEVEL_OFF "Set the spdlog level to off" OFF)

#if(SPDLOG_LEVEL_INFO)
#    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
#    message("Set SPDLOG_LEVEL_INFO")
#elseif(SPDLOG_LEVEL_DEBUG)
#    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
#    message("Set SPDLOG_LEVEL_DEBUG")
#elseif(SPDLOG_LEVEL_OFF)
#    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF)
#    message("Set SPDLOG_LEVEL_OFF")
#elseif(SPDLOG_LEVEL_TRACE)
#    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
#    message("Set SPDLOG_LEVEL_TRACE")
#else()
#    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF)
#    message("Set SPDLOG_LEVEL_OFF")
#endif()
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF)

if(BUILD_SHMEM)
  find_package(iceoryx_posh CONFIG REQUIRED)
  find_package(iceoryx_hoofs CONFIG REQUIRED)
endif()

add_compile_definitions(GPULESS_ELF_ANALYZER)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# special options for compiling on the CSCS cluster
cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
if (${_hostname} MATCHES ".*\.cscs\.ch")
    add_compile_options(-DCSCS)
    include_directories(${CUDNN_DIR}/include)
    link_directories(${CUDNN_DIR}/lib64)
    link_directories(${CUDNN_DIR}/lib)
else()
    include_directories(${CUDNN_DIR}/include)
    link_directories(${CUDNN_DIR}/lib)
    link_directories(${CUDNN_DIR}/lib64)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -Wall -Wextra -Wno-unused-parameter -ggdb -fno-operator-names")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# build flatbuffer schemas with external Makefile
add_custom_target(
    flatbuffer
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/schemas
)

add_executable(manager_trace
    src/trace/manager/manager.cpp
    src/trace/manager/manager_device.cpp
    src/trace/manager/memory_store.cpp
    src/trace/cuda_trace.cpp
    src/trace/cuda_trace_converter.cpp
    src/trace/cuda_api_calls.cpp
    src/trace/cudnn_api_calls.cpp
    src/trace/cublas_api_calls.cpp
    src/trace/cuda_virtual_device.cpp
    src/trace/cuda_virtual_device.cpp
    src/trace/tcp_gpu_session.cpp
    src/trace/trace_executor_local.cpp
    src/trace/trace_executor_tcp_client.cpp
    src/trace/trace_executor_shmem_client.cpp
    src/trace/dlsym_util.cpp
    src/TcpClient.cpp
    src/utils.cpp)
target_include_directories(
    manager_trace SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)
add_dependencies(manager_trace spdlog)
add_dependencies(manager_trace flatbuffer)
target_link_libraries(manager_trace
    iberty
    stdc++fs
    cudnn
    spdlog::spdlog
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    pthread
    flatbuffers
    iceoryx_posh::iceoryx_posh
)

if(BUILD_SHMEM)
  target_link_libraries(manager_trace iceoryx_posh::iceoryx_posh iceoryx_hoofs::iceoryx_hoofs)
endif()

add_executable(manager_device
    src/trace/manager/manager_device_cli.cpp
    src/trace/manager/manager_device.cpp
    src/trace/manager/memory_store.cpp
    src/trace/cuda_trace.cpp
    src/trace/cuda_trace_converter.cpp
    src/trace/cuda_api_calls.cpp
    src/trace/cudnn_api_calls.cpp
    src/trace/cublas_api_calls.cpp
    src/trace/cuda_virtual_device.cpp
    src/trace/cuda_virtual_device.cpp
    src/trace/tcp_gpu_session.cpp
    src/trace/trace_executor_local.cpp
    src/trace/trace_executor_tcp_client.cpp
    src/trace/trace_executor_shmem_client.cpp
    src/trace/dlsym_util.cpp
    src/TcpClient.cpp
    src/utils.cpp)
target_include_directories(
    manager_device SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)
add_dependencies(manager_device spdlog)
add_dependencies(manager_device flatbuffer)
target_link_options(manager_device PRIVATE "LINKER:-no-as-needed")
target_link_libraries(manager_device
    iberty
    stdc++fs
    cudnn
    spdlog::spdlog
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    pthread
    flatbuffers
    iceoryx_posh::iceoryx_posh
)

if(BUILD_SHMEM)
  target_link_libraries(manager_device iceoryx_posh::iceoryx_posh iceoryx_hoofs::iceoryx_hoofs)
endif()

add_library(analysis SHARED
    src/analysis/libcudaanalysis.cpp
    src/trace/cubin_analysis.cpp
    src/utils.cpp src/trace/trace_executor.hpp)
add_dependencies(analysis spdlog)
target_link_libraries(analysis
    CUDA::cudart
    CUDA::cuda_driver
    spdlog::spdlog
)

add_library(argument_analysis SHARED
        src/argument_analysis/libarganalysis.cpp
        src/trace/cubin_analysis.cpp
        src/utils.cpp src/trace/trace_executor.hpp)
add_dependencies(argument_analysis spdlog)
target_link_libraries(argument_analysis
        CUDA::cudart
        CUDA::cuda_driver
    	spdlog::spdlog
)

add_library(gpuless SHARED
    src/trace/libgpuless.cpp
    src/trace/hijack_cudnn.cpp
    src/trace/hijack_cublas.cpp
    src/trace/cuda_trace.cpp
    src/trace/cuda_trace_converter.cpp
    src/trace/cuda_api_calls.cpp
    src/trace/cudnn_api_calls.cpp
    src/trace/cublas_api_calls.cpp
    src/trace/cuda_virtual_device.cpp
    src/trace/cuda_virtual_device.cpp
    src/trace/tcp_gpu_session.cpp
    src/trace/trace_executor_local.cpp
    src/trace/trace_executor_tcp_client.cpp
    src/trace/trace_executor_shmem_client.cpp
    src/trace/dlsym_util.cpp
    src/trace/manager/memory_store.cpp
    src/TcpClient.cpp
    src/utils.cpp
    src/trace/cubin_analysis.cpp)
target_include_directories(gpuless SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(gpuless flatbuffer)
add_dependencies(gpuless spdlog)
target_link_libraries(gpuless
    iberty
    stdc++fs
    cudnn
    spdlog::spdlog
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    flatbuffers
    iceoryx_posh::iceoryx_posh
)

if(BUILD_SHMEM)
  target_link_libraries(gpuless iceoryx_posh::iceoryx_posh iceoryx_hoofs::iceoryx_hoofs)
endif()

add_executable(cubin_analyzer
    src/cubin_analyzer.cpp
    src/trace/cubin_analysis.cpp
    src/utils.cpp
)
target_include_directories(
    cubin_analyzer SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)
add_dependencies(cubin_analyzer spdlog)
target_link_libraries(cubin_analyzer
    iberty
    spdlog::spdlog
    CUDA::cuda_driver
)

add_custom_command(
    TARGET gpuless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR})
