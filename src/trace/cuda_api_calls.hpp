#ifndef __CUDA_API_CALLS_H__
#define __CUDA_API_CALLS_H__

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "../schemas/trace_execution_protocol_generated.h"
#include "cubin_analysis.hpp"
#include "cuda_virtual_device.hpp"
#include "flatbuffers/flatbuffers.h"

namespace gpuless {

class CudaApiCall {
  public:
    virtual ~CudaApiCall() = default;

    virtual cudaError_t executeNative(CudaVirtualDevice &vdev) = 0;

    // serialize to flatbuffer object and append to list of serialized calls
    // TODO: make pure virtual
    virtual void appendToFBCudaApiCallList(
        flatbuffers::FlatBufferBuilder &builder,
        std::vector<flatbuffers::Offset<FBCudaApiCall>> &api_calls) {
        std::cerr << "appendToFBCudaApiCallList(): not implemented" << std::endl;
        std::exit(EXIT_FAILURE);
    };

    virtual std::vector<uint64_t> requiredCudaModuleIds() { return {}; };
    virtual std::vector<std::string> requiredFunctionSymbols() { return {}; };
    virtual std::string typeName() { return typeid(*this).name(); }
};

/*
 * Public CUDA API functions
 */

class CudaMalloc : public CudaApiCall {
  public:
    void *devPtr;
    size_t size;

    explicit CudaMalloc(size_t size);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;

    void appendToFBCudaApiCallList(
        flatbuffers::FlatBufferBuilder &builder,
        std::vector<flatbuffers::Offset<FBCudaApiCall>> &api_calls) override;
};

class CudaMemcpyH2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    std::vector<uint8_t> buffer;

    CudaMemcpyH2D(void *dst, const void *src, size_t size);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;

    void appendToFBCudaApiCallList(
        flatbuffers::FlatBufferBuilder &builder,
        std::vector<flatbuffers::Offset<FBCudaApiCall>> &api_calls) override;
};

class CudaMemcpyD2H : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    std::vector<uint8_t> buffer;

    CudaMemcpyD2H(void *dst, const void *src, size_t size);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;

    void appendToFBCudaApiCallList(
        flatbuffers::FlatBufferBuilder &builder,
        std::vector<flatbuffers::Offset<FBCudaApiCall>> &api_calls) override;
};

class CudaMemcpyD2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;

    CudaMemcpyD2D(void *dst, const void *src, size_t size);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudaMemcpyAsyncH2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;
    std::vector<uint8_t> buffer;

    CudaMemcpyAsyncH2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudaMemcpyAsyncD2H : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;
    std::vector<uint8_t> buffer;

    CudaMemcpyAsyncD2H(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudaMemcpyAsyncD2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;

    CudaMemcpyAsyncD2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudaFree : public CudaApiCall {
  public:
    void *devPtr;

    CudaFree(void *devPtr);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;

    void appendToFBCudaApiCallList(
        flatbuffers::FlatBufferBuilder &builder,
        std::vector<flatbuffers::Offset<FBCudaApiCall>> &api_calls) override;
};

class CudaLaunchKernel : public CudaApiCall {
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
    std::vector<KParamInfo> paramInfos;

    CudaLaunchKernel(std::string symbol,
                     std::vector<uint64_t> required_cuda_modules,
                     std::vector<std::string> required_function_symbols,
                     const void *fnPtr, const dim3 &gridDim,
                     const dim3 &blockDim, size_t sharedMem,
                     cudaStream_t stream,
                     std::vector<std::vector<uint8_t>> &paramBuffers,
                     std::vector<KParamInfo> &paramInfos);

    cudaError_t executeNative(CudaVirtualDevice &vdev) override;

    void appendToFBCudaApiCallList(
        flatbuffers::FlatBufferBuilder &builder,
        std::vector<flatbuffers::Offset<FBCudaApiCall>> &api_calls) override;

    std::vector<uint64_t> requiredCudaModuleIds() override;
    std::vector<std::string> requiredFunctionSymbols() override;
};

class CudaStreamSynchronize : public CudaApiCall {
  public:
    cudaStream_t stream;

    CudaStreamSynchronize(cudaStream_t stream);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudaStreamIsCapturing : public CudaApiCall {
  public:
    cudaStream_t stream;
    enum cudaStreamCaptureStatus cudaStreamCaptureStatus;

    CudaStreamIsCapturing(cudaStream_t stream);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

/*
 * Private CUDA API functions
 */

class PrivCudaRegisterFatBinary : public CudaApiCall {
  public:
    uint64_t client_handle_id;

    PrivCudaRegisterFatBinary(uint64_t client_handle_id);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class PrivCudaRegisterFatBinaryEnd : public CudaApiCall {
  public:
    uint64_t client_handle_id;

    PrivCudaRegisterFatBinaryEnd(uint64_t client_handle_id);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class PrivCudaUnregisterFatBinary : public CudaApiCall {
  public:
};

class PrivCudaRegisterFunction : public CudaApiCall {
  public:
    uint64_t client_handle_id;
    void *host_fn_ptr;
    void *device_fn_ptr;
    std::string fn_name;
    int thread_limit;
    uint3 *tid;
    uint3 *bid;
    dim3 *block_dim;
    dim3 *grid_dim;
    int *wSize;

    PrivCudaRegisterFunction(uint64_t client_handle_id, void *hostFnPtr,
                             void *deviceFnPtr, const std::string &fnName,
                             int threadLimit, uint3 *tid, uint3 *bid,
                             dim3 *blockDim, dim3 *gridDim, int *wSize);
    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
};

class PrivCudaRegisterVar : public CudaApiCall {
  private:
    std::vector<uint64_t> required_cuda_modules_;

  public:
    uint64_t client_handle_id;
    void *host_var;
    void *device_address;
    std::string device_name;
    int ext;
    size_t size;
    int constant;
    int global;

    PrivCudaRegisterVar(std::vector<uint64_t> required_cuda_modules,
                        uint64_t clientHandleId, void *hostVar,
                        void *deviceAddress, std::string deviceName, int ext,
                        size_t size, int constant, int global);

    cudaError_t executeNative(CudaVirtualDevice &vdev) override;
    std::vector<uint64_t> requiredCudaModuleIds() override;
};

} // namespace gpuless

#endif // __CUDA_API_CALLS_H__
