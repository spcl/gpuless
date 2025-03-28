#ifndef __CUDA_API_CALLS_H__
#define __CUDA_API_CALLS_H__

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <typeinfo>
#include <vector>

#include "../schemas/trace_execution_protocol_generated.h"
#include "abstract_cuda_api_call.hpp"
#include "cubin_analysis.hpp"
#include "cuda_virtual_device.hpp"
#include "flatbuffers/flatbuffers.h"
#include "manager/memory_store.hpp"

namespace gpuless {

class CudaRuntimeApiCall : public AbstractCudaApiCall {
  public:
    std::string nativeErrorToString(uint64_t err) override;
};

class CudaMalloc : public CudaRuntimeApiCall {
  public:
    void *devPtr;
    size_t size;

    explicit CudaMalloc(size_t size);
    explicit CudaMalloc(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaMemcpyH2D : public CudaRuntimeApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    // Data was transmitted directly in the payload - legacy option for FB
    std::vector<uint8_t> buffer;
    // Data is located on our side as a pointer
    uint8_t* buffer_ptr;
    // The name of the shared memory page passed from another process
    std::string shared_name;

    CudaMemcpyH2D(void *dst, const void *src, size_t size);
    CudaMemcpyH2D(void *dst, const void *src, size_t size, std::string shared_name);
    explicit CudaMemcpyH2D(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    virtual bool is_memop() override { return true; }

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaMemcpyD2H : public CudaRuntimeApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    // Data was transmitted directly in the payload - legacy option for FB
    std::vector<uint8_t> buffer;
    // Data is located on our side as a pointer
    uint8_t* buffer_ptr;
    // The name of the shared memory page passed from another process
    std::string shared_name;

    CudaMemcpyD2H(void *dst, const void *src, size_t size);
    CudaMemcpyD2H(void *dst, const void *src, size_t size, std::string shared_name);
    explicit CudaMemcpyD2H(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    virtual bool is_memop() override { return true; }

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaMemcpyD2D : public CudaRuntimeApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;

    CudaMemcpyD2D(void *dst, const void *src, size_t size);
    explicit CudaMemcpyD2D(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    virtual bool is_memop() override { return true; }

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaMemcpyAsyncH2D : public CudaRuntimeApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;
    //std::vector<uint8_t> buffer;

    // Data was transmitted directly in the payload - legacy option for FB
    std::vector<uint8_t> buffer;
    // Data is located on our side as a pointer
    uint8_t* buffer_ptr;
    // The name of the shared memory page passed from another process
    std::string shared_name;

    CudaMemcpyAsyncH2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    CudaMemcpyAsyncH2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream, std::string shared_name);
    explicit CudaMemcpyAsyncH2D(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    virtual bool is_memop() override { return true; }

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaMemcpyAsyncD2H : public CudaRuntimeApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;
    //std::vector<uint8_t> buffer;

    std::vector<uint8_t> buffer;
    // Data is located on our side as a pointer
    uint8_t* buffer_ptr;
    // The name of the shared memory page passed from another process
    std::string shared_name;

    CudaMemcpyAsyncD2H(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    CudaMemcpyAsyncD2H(void *dst, const void *src, size_t size,
                       cudaStream_t stream, std::string shared_name);
    explicit CudaMemcpyAsyncD2H(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    virtual bool is_memop() override { return true; }

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaMemcpyAsyncD2D : public CudaRuntimeApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;

    CudaMemcpyAsyncD2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    explicit CudaMemcpyAsyncD2D(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaFree : public CudaRuntimeApiCall {
  public:
    void *devPtr;

    explicit CudaFree(void *devPtr);
    explicit CudaFree(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaLaunchKernel : public CudaRuntimeApiCall {
  public:
    std::string symbol;
    std::vector<uint64_t> required_cuda_modules_;
    std::vector<std::string> required_function_symbols_;
    const void *fnPtr;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;

    std::vector<std::vector<uint8_t>> paramBuffers;
#if defined(GPULESS_ELF_ANALYZER)
    std::vector<int> paramInfos;
#else
    std::vector<PTXKParamInfo> paramInfos;
#endif

#if defined(GPULESS_ELF_ANALYZER)
    CudaLaunchKernel(std::string symbol,
                     std::vector<uint64_t> required_cuda_modules,
                     std::vector<std::string> required_function_symbols,
                     const void *fnPtr, const dim3 &gridDim,
                     const dim3 &blockDim, size_t sharedMem,
                     cudaStream_t stream,
                     std::vector<std::vector<uint8_t>> &paramBuffers,
                     std::vector<int> &paramInfos);
#else
    CudaLaunchKernel(std::string symbol,
                     std::vector<uint64_t> required_cuda_modules,
                     std::vector<std::string> required_function_symbols,
                     const void *fnPtr, const dim3 &gridDim,
                     const dim3 &blockDim, size_t sharedMem,
                     cudaStream_t stream,
                     std::vector<std::vector<uint8_t>> &paramBuffers,
                     std::vector<PTXKParamInfo> &paramInfos);
#endif
    explicit CudaLaunchKernel(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    virtual bool is_kernel() override { return true; }

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;

    std::vector<uint64_t> requiredCudaModuleIds() override;
    std::vector<std::string> requiredFunctionSymbols() override;
};

class CudaStreamSynchronize : public CudaRuntimeApiCall {
  public:
    cudaStream_t stream;

    explicit CudaStreamSynchronize(cudaStream_t stream);
    explicit CudaStreamSynchronize(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaGetDeviceProperties : public CudaRuntimeApiCall {
  public:
    cudaDeviceProp properties{};

    CudaGetDeviceProperties();
    explicit CudaGetDeviceProperties(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaDeviceSynchronize : public CudaRuntimeApiCall {
  public:
    CudaDeviceSynchronize();
    explicit CudaDeviceSynchronize(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudaFuncGetAttributes : public CudaRuntimeApiCall {
  public:
    cudaFuncAttributes cfa{};

    std::string symbol;
    std::vector<uint64_t> required_cuda_modules_;
    std::vector<std::string> required_function_symbols_;

    CudaFuncGetAttributes(
        std::string symbol,
        std::vector<uint64_t> requiredCudaModules,
        std::vector<std::string> requiredFunctionSymbols);

    explicit CudaFuncGetAttributes(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;

    std::vector<uint64_t> requiredCudaModuleIds() override;
    std::vector<std::string> requiredFunctionSymbols() override;
};

} // namespace gpuless

#endif // __CUDA_API_CALLS_H__
