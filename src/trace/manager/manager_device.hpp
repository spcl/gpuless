#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iceoryx_posh/popo/untyped_server.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_watcher.hpp>
#include <spdlog/spdlog.h>

void manage_device(const std::string& device, uint16_t port);
void swap_in();
void swap_out();
void manage_device_shmem(const std::string& device, const std::string& app_name, const std::string& poll_type, const char* user_name, bool use_vmm);

struct SigHandler
{
  static iox::popo::WaitSet<>* waitset_ptr;
  static bool quit;

  static void sigHandler(int sig [[maybe_unused]])
  {
    quit = true;
    if(SigHandler::waitset_ptr) {
      SigHandler::waitset_ptr->markForDestruction();
    }
  }
};

enum class GPUlessMessage {

  LOCK_DEVICE = 0,
  BASIC_EXEC = 1,
  MEMCPY_ONLY = 2,
  FULL_EXEC = 3,
  SWAP_OFF = 4,
  SWAP_IN = 5,

  REGISTER = 10,
  SWAP_OFF_CONFIRM = 11
};


enum class Status
{
  NO_EXEC,
  BASIC_OPS,
  MEMCPY,
  EXEC
};

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

  void* create_allocation(size_t size)
  {
    size = align_size(size);

    DataElement de;
    CHECK_CUDA(cuMemAddressReserve(&de.devicePtr, size, 0, 0, 0));
    de.hostPtr = nullptr;
    de.size = size;

    de.mem_alloc_prop = _mem_alloc_prop;
    CHECK_CUDA(cuMemCreate(&de.alloc_handle, size, &de.mem_alloc_prop, 0));

    CHECK_CUDA(cuMemMap(de.devicePtr, size, 0, de.alloc_handle, 0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(de.devicePtr, size, &accessDesc, 1));

    ptrSizeStore[reinterpret_cast<void*>(de.devicePtr)] = de;

    return reinterpret_cast<void*>(de.devicePtr);
  }

  bool release_allocation(const void* ptrToRemove)
  {
    auto it = ptrSizeStore.find(ptrToRemove);
    if(it == ptrSizeStore.end()) {
      spdlog::error("Couldn't find data for pointer {}", fmt::ptr(ptrToRemove));
      return false;
    }
    DataElement& de = (*it).second;

    CHECK_CUDA(cuMemUnmap(de.devicePtr, de.size));
    CHECK_CUDA(cuMemAddressFree(de.devicePtr, de.size));
    CHECK_CUDA(cuMemRelease(de.alloc_handle));

    ptrSizeStore.erase(it);

    return true;
  }

  void add_allocation(void* devicePtr, size_t size)
  {
    DataElement de;
    de.devicePtr = reinterpret_cast<std::uintptr_t>(devicePtr);
    de.hostPtr = nullptr;
    de.size = size;
    ptrSizeStore[devicePtr] = de;
  }

  void remove_allocation(const void* ptrToRemove)
  {
    // TODO: check existence
    ptrSizeStore.erase(ptrToRemove);
  }

  void print_stats()
  {
    size_t size = 0;
    size_t size2 = 0;
    for (auto& [dev_ptr, buffer] : ptrSizeStore) {
      size += buffer.size;
      if(buffer.size < 2*1024*1024)
        size2 += buffer.size;
    }

    spdlog::error("Total size {} {}", size, size2);
  }

  void swap_in()
  {
    //size_t aligned_sz;
    //CUmemAllocationProp prop = {};
    //prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    //prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    //cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    //CUdeviceptr reservedAddress = 0;

    //// Reserve memory on device
    //int streamIdx = 0;
    //for (auto& [dev_ptr, buffer] : ptrSizeStore) {

    //  size_t new_size = ((buffer.size + aligned_sz - 1) / aligned_sz) * aligned_sz;

    //  // Step 1: Reserve the address
    //  CHECK_CUDA(cuMemAddressReserve(&reservedAddress, bufferSize, 0, 0, 0));
    //  printf("Reserved address: %p\n", (void*)reservedAddress);

    //  // Step 2: Allocate physical memory
    //  CUmemGenericAllocationHandle allocHandle;
    //  CHECK_CUDA(cuMemCreate(&allocHandle, bufferSize, &prop, 0));

    //  // Step 3: Map the physical memory to the reserved address
    //  CHECK_CUDA(cuMemMap(reservedAddress, bufferSize, 0, allocHandle, 0));

    //  // Step 4: Set memory access flags
    //  CUmemAccessDesc accessDesc = {};
    //  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    //  accessDesc.location.id = 0;
    //  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    //  CHECK_CUDA(cuMemSetAccess(reservedAddress, bufferSize, &accessDesc, 1));


    //  cudaMalloc(&buffer.devicePtr, buffer.size);

    //  //cuMemAddressReserve(&reservedAddress, buffer.size, 0, 0, 0);

    //  cudaMemcpyAsync(
    //    buffer.devicePtr, buffer.hostPtr,
    //    buffer.size,
    //    cudaMemcpyHostToDevice,
    //    streams[streamIdx % numStreams]
    //  );
    //  streamIdx += 1;
    //}

    //cudaDeviceSynchronize();

    //ptrSizeStore.clear();
  }

  void swap_out()
  {
    //int streamIdx = 0;
    //for (auto& [devPtr, buffer] : ptrSizeStore) {
    //  void* tempVector;
    //  cudaHostAlloc(&tempVector, buffer.size, cudaHostAllocDefault);
    //  buffer.hostPtr = tempVector;
    //  cudaMemcpyAsync(
    //    buffer.hostPtr, buffer.devicePtr, buffer.size,
    //    cudaMemcpyDeviceToHost,
    //    streams[streamIdx % numStreams]
    //  );
    //  streamIdx += 1;
    //}

    //// Wait for transfers to complete
    //cudaDeviceSynchronize();

    //for (auto& [devPtr, buffer] : ptrSizeStore) {
    //  cudaFree(buffer.devicePtr);
    //}
  }

  void enable_vmm()
  {
    _virtual_memory_management = true;

    _mem_alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    _mem_alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    cuMemGetAllocationGranularity(&_alloc_granularity, &_mem_alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    _access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    _access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

private:

  size_t _alloc_granularity = false;
  bool _virtual_memory_management = false;
  CUmemAllocationProp _mem_alloc_prop;
  CUmemAccessDesc _access_desc;
  std::unordered_map<const void*, DataElement> ptrSizeStore;

  // Number of streams to create
  static const int numStreams = 32;
  cudaStream_t streams[numStreams];

  MemoryStore()
  {

    for (int i = 0; i < numStreams; ++i) {
      cudaStreamCreate(&streams[i]);
    }
  }

  ~MemoryStore()
  {
    for (int i = 0; i < numStreams; ++i) {
      cudaStreamDestroy(streams[i]);
    }
  }

};

struct ExecutionStatus
{

  bool can_exec()
  {
    return _status != Status::NO_EXEC;
  }

  bool can_memcpy()
  {
    return _status == Status::MEMCPY || _status == Status::EXEC;
  }

  bool can_exec_kernels()
  {
    return _status == Status::EXEC;
  }

  void lock()
  {
    spdlog::info("[ExecutionStatus] Locking device!");
    _status = Status::NO_EXEC;
  }

  void basic_exec()
  {
    spdlog::info("[ExecutionStatus] CPU execution!");
    _status = Status::BASIC_OPS;
  }

  void memcpy()
  {
    spdlog::info("[ExecutionStatus] Memory operations!");
    _status = Status::MEMCPY;
  }

  void exec()
  {
    spdlog::info("[ExecutionStatus] Full execution!");
    _status = Status::EXEC;
  }

  void save(int pos)
  {
    _pos = pos;
  }

  void save_payload(const void* ptr)
  {
    _payload_ptr = ptr;
  }

  const void* load_payload()
  {
    return _payload_ptr;
  }

  bool has_unfinished_trace()
  {
    return _pos != -1;
  }

  int load()
  {
    return _pos;
  }

  static ExecutionStatus& instance()
  {
    static ExecutionStatus status;
    return status;
  }

private:
  Status _status = Status::NO_EXEC;

  int _pos = -1;
  const void* _payload_ptr = nullptr;
};

struct ShmemServer {

  std::unique_ptr<iox::popo::UntypedServer> server;

  void setup(const std::string app_name);
  void loop(const char*);
  void loop_wait(const char*);
  void finish();

  void* take();
  void release(void*);

  bool _process_client(const void* payload);
  bool _process_remainder();
  double _sum = 0;

  std::optional<iox::posix::SignalGuard> sigint;
  std::optional<iox::posix::SignalGuard> sigterm;
};

#endif // __MANAGER_DEVICE_HPP__
