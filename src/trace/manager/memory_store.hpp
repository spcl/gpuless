#ifndef __MEMORY_STORE_HPP__
#define __MEMORY_STORE_HPP__


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

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

private:

  size_t _alloc_granularity = false;
  bool _virtual_memory_management = false;
  CUmemAllocationProp _mem_alloc_prop;
  CUmemAccessDesc _access_desc;
  std::unordered_map<const void*, DataElement> ptrSizeStore;

  // Number of streams to create
  static const int numStreams = 32;
  cudaStream_t streams[numStreams];

  MemoryStore();

  ~MemoryStore();

};

#endif
