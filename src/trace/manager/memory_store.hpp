#ifndef __MEMORY_STORE_HPP__
#define __MEMORY_STORE_HPP__

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <cuda.h>
#include "../cuda_api_calls.hpp"
#include "../cudnn_api_calls.hpp"
#include "../cublas_api_calls.hpp"
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <nvml.h>

struct DataElement
{
  CUmemAllocationProp mem_alloc_prop;
  CUmemGenericAllocationHandle alloc_handle;
  CUdeviceptr devicePtr;
  //void* devicePtr;
  void* hostPtr;
  size_t size;
};

#define CHECK_CUDA(call) {\
  CUresult result = call;\
  if (result != CUDA_SUCCESS) {\
      const char* errorString;\
      cuGetErrorString(result, &errorString);\
      printf("[%d] CUDA Error: %s\n", __LINE__, errorString);\
  }\
}

#define CHECK_CUDA_E(call) {\
  cudaError_t result = call;\
  if (result != cudaSuccess) {\
      printf("[%d] CUDA Error: %s\n", __LINE__, cudaGetErrorString(result));\
  }\
}

enum class MemoryCheckResult { OK, OOM, UNCHANGED };

struct MemoryStore
{

  static MemoryStore& get_instance()
  {
    static MemoryStore instance;
    return instance;
  }

  bool uses_vmm()
  {
    return _virtual_memory_management;
  }

  size_t align_size(size_t size)
  {
    return ((size + _alloc_granularity - 1) / _alloc_granularity) * _alloc_granularity;
  }

  void* create_allocation(size_t size);
  bool release_allocation(const void* ptrToRemove);
  void add_allocation(void* devicePtr, size_t size);
  void remove_allocation(const void* ptrToRemove);
  void print_stats();
  void swap_in();
  void swap_out();
  void enable_vmm();

  // New memory check API
  MemoryCheckResult check_memory(const char* api_call_name);
  void signal_likely_check();
  void shutdown_background_thread();
  void check_memory_final();
  void set_max_memory(unsigned long long max_bytes);
  bool is_oom() const { return _oom_detected.load(std::memory_order_acquire); }

  // Legacy API - kept for backward compatibility
  unsigned long long nvml_used_memory();

  void print_memory_report();
  unsigned long long nvml_gpu_memory() const { return _nvml_gpu_memory; }
  size_t current_bytes() const { return _current_bytes; }
  size_t peak_bytes() const { return _peak_bytes; }
  size_t current_count() const { return _current_count; }
  double nvml_max_time() const { return _nvml_max_time; }

  static constexpr std::array<std::string_view, 1> tracked_kernels = {
    "_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_39_GLOBAL__N__d5663c65_7_Copy_cu_ddc55cb923direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE6_clEvEUlfE_NS_6detail5ArrayIPcLi2EEE23TrivialOffsetCalculatorILi1EjESD_NS0_6memory12LoadWithCastILi1EEENSE_13StoreWithCastEEEviT_T0_T1_T2_T3_T4_"
  };

  // Classification helpers
  static bool is_must_call(gpuless::AbstractCudaApiCall* call)
  {
    return dynamic_cast<gpuless::CudnnCreate*>(call) != nullptr
        || dynamic_cast<gpuless::CublasCreateV2*>(call) != nullptr
        || is_tracked_kernel(call);
  }

  static bool is_likely_call(gpuless::AbstractCudaApiCall* call)
  {
    return dynamic_cast<gpuless::CudnnSetConvolutionMathType*>(call) != nullptr;
  }

  static bool is_tracked_kernel(gpuless::AbstractCudaApiCall* call)
  {
    auto kernel_ptr = dynamic_cast<gpuless::CudaLaunchKernel*>(call);
    if (!kernel_ptr) return false;
    for (const auto& name : tracked_kernels) {
      if (kernel_ptr->symbol == name) return true;
    }
    return false;
  }

private:

  bool _check_nvml_initialized();
  std::tuple<unsigned long long, double> _nvml_used_memory() const;

  void on_malloc(size_t size);
  void on_free(size_t size);

  void _bg_thread_func();

  size_t _alloc_granularity = false;
  bool _virtual_memory_management = false;
  CUmemAllocationProp _mem_alloc_prop;
  CUmemAccessDesc _access_desc;
  std::unordered_map<const void*, DataElement> ptrSizeStore;

  // Allocation tracking state
  size_t _current_bytes = 0;
  size_t _peak_bytes = 0;
  size_t _current_count = 0;
  size_t _peak_count = 0;

  // NVML state
  bool _nvml_initialized = false;
  nvmlDevice_t _nvml_device{};
  unsigned long long _nvml_gpu_memory = 0;
  unsigned long long _nvml_gpu_memory_wo_mallocs = 0;
  double _nvml_time_us = 0;
  int _nvml_time_count = 0;
  double _nvml_max_time = 0;

  // OOM threshold
  unsigned long long _max_gpu_memory = 0;  // 0 = no limit
  std::atomic<bool> _oom_detected{false};

  // Background thread for "likely" checks
  std::thread _bg_thread;
  std::mutex _bg_mutex;
  std::condition_variable _bg_cv;
  bool _bg_pending = false;
  bool _bg_shutdown = false;

  // Mutex for shared stats accessed by bg thread
  std::mutex _stats_mutex;

  // Number of streams to create
  static const int numStreams = 32;
  cudaStream_t streams[numStreams];

  pid_t _my_pid;

  MemoryStore();

  ~MemoryStore();

};

#endif
