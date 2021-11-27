#ifndef GPULESS_CUDA_VDEV_H
#define GPULESS_CUDA_VDEV_H

#include "../utils.hpp"
#include <cstdint>
#include <cublas.h>
#include <cuda.h>
#include <cudnn.h>
#include <fatbinary_section.h>
#include <iostream>
#include <map>
#include <string>

class CudaVirtualDevice {
  private:
    bool initialized = false;

  public:
    // virtualization of cudnn handles to avoid costly synchronizations
    std::vector<cudnnHandle_t> cudnn_handles_virtual_to_real;
    std::vector<cudnnTensorDescriptor_t>
        cudnn_tensor_descriptor_virtual_to_real;
    std::vector<cudnnFilterDescriptor_t>
        cudnn_filter_descriptor_virtual_to_real;
    std::vector<cudnnConvolutionDescriptor_t>
        cudnn_convolution_descriptor_virtual_to_real;

    // virtualization of cublas handles to avoid costly synchronizations
    std::vector<cublasHandle_t> cublas_handle_virtual_to_real;

    std::map<uint64_t, CUmodule> module_registry_;
    std::map<std::string, CUfunction> function_registry_;
    CUdevice device;
    CUcontext context;

    void initRealDevice();
};

#endif // GPULESS_CUDA_VDEV_H
