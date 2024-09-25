#ifndef GPULESS_ABSTRACT_CUDA_API_CALL_HPP
#define GPULESS_ABSTRACT_CUDA_API_CALL_HPP

#include "cuda_virtual_device.hpp"

namespace gpuless {

class AbstractCudaApiCall {
  public:
    virtual uint64_t executeNative(CudaVirtualDevice &vdev) = 0;
    virtual std::string nativeErrorToString(uint64_t err) = 0;

    virtual flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) = 0;

    virtual std::vector<uint64_t> requiredCudaModuleIds() { return {}; };
    virtual std::vector<std::string> requiredFunctionSymbols() { return {}; };
    virtual std::string typeName() { return typeid(*this).name(); }

    virtual bool is_memop() { return false; }
    virtual bool is_kernel() { return false; }
};

} // namespace gpuless

#endif // GPULESS_ABSTRACT_CUDA_API_CALL_HPP
