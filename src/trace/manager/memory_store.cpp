#include "memory_store.hpp"
#include <iostream>
#include <ostream>
#include <chrono>
#include <nvml.h>
#include <unistd.h>

MemoryStore::MemoryStore()
{
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
  _my_pid = getpid();

  // Start background thread for "likely" checks
  _bg_thread = std::thread(&MemoryStore::_bg_thread_func, this);
}

MemoryStore::~MemoryStore()
{
  shutdown_background_thread();

  // FIXME: release pending resources
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}

void MemoryStore::_bg_thread_func()
{
  while (true) {
    std::unique_lock<std::mutex> lock(_bg_mutex);
    _bg_cv.wait(lock, [this] { return _bg_pending || _bg_shutdown; });

    if (_bg_shutdown) {
      return;
    }

    _bg_pending = false;
    lock.unlock();

    check_memory("likely_batch_check");
    // OOM flag is already set atomically if detected
  }
}

void MemoryStore::shutdown_background_thread()
{
  {
    std::lock_guard<std::mutex> lock(_bg_mutex);
    if (_bg_shutdown) return;
    _bg_shutdown = true;
  }
  _bg_cv.notify_one();
  if (_bg_thread.joinable()) {
    _bg_thread.join();
  }
}

void MemoryStore::set_max_memory(unsigned long long max_bytes)
{
  _max_gpu_memory = max_bytes;
  spdlog::info("MemoryStore: max GPU memory set to {} bytes ({:.2f} MB)",
               _max_gpu_memory, _max_gpu_memory / (1024.0 * 1024.0));
}

MemoryCheckResult MemoryStore::check_memory(const char* api_call_name)
{

  auto [new_memory, time] = _nvml_used_memory();

  std::lock_guard<std::mutex> lock(_stats_mutex);
  _nvml_time_us += time;
  _nvml_time_count += 1;
  if (time > _nvml_max_time) {
    _nvml_max_time = time;
  }

  MemoryCheckResult result = MemoryCheckResult::OK;

  // With full profiling, we call the check always -> it is either a change
  // or we report unchanged.
  //
  // With normal profiling, we just print the actual checks performed for the selected calls.
  if (new_memory == _nvml_gpu_memory) {
    if(std::string_view{api_call_name} != "likely_batch_check") {
      spdlog::warn("Memory unchanged after {}: {} MB",
                   api_call_name, new_memory / (1024.0 * 1024.0));
    }
    result = MemoryCheckResult::UNCHANGED;
  } else {
#if defined(MIGNIFICIENT_WITH_PROFILING)
      spdlog::info("Memory check after {}: nvml={} mallocs={} nvml_wo_mallocs={}",
                  api_call_name, new_memory, _current_bytes, _nvml_gpu_memory_wo_mallocs);
      print_memory_report();
#endif
    _nvml_gpu_memory = new_memory;
    _nvml_gpu_memory_wo_mallocs = new_memory - _current_bytes;
  }


  std::cerr << _max_gpu_memory << " " <<  new_memory << std::endl;
  if (_max_gpu_memory > 0 && new_memory > _max_gpu_memory) {
    spdlog::error("OOM detected after {}: nvml={} > max={}",
                  api_call_name, new_memory, _max_gpu_memory);
    _oom_detected.store(true, std::memory_order_release);
    return MemoryCheckResult::OOM;
  }

  return result;
}

void MemoryStore::signal_likely_check()
{
  std::lock_guard<std::mutex> lock(_bg_mutex);
  _bg_pending = true;
  _bg_cv.notify_one();
}

void MemoryStore::check_memory_final()
{
  std::lock_guard<std::mutex> lock(_stats_mutex);

  auto [new_memory, time] = _nvml_used_memory();
  unsigned long long expected = _current_bytes + _nvml_gpu_memory_wo_mallocs;

  if (new_memory != expected) {
    spdlog::warn("Memory consistency check failed! NVML reports {} but expected {} (mallocs={} + nvml_wo_mallocs={})",
      new_memory, expected, _current_bytes, _nvml_gpu_memory_wo_mallocs);
  } else {
    spdlog::info("Memory consistency check passed: {} bytes", new_memory);
  }
}

void* MemoryStore::create_allocation(size_t size)
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

  on_malloc(size);

  return reinterpret_cast<void*>(de.devicePtr);
}

bool MemoryStore::release_allocation(const void* ptrToRemove)
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

  on_free(de.size);

  return true;
}

void MemoryStore::add_allocation(void* devicePtr, size_t size)
{
  DataElement de;
  de.devicePtr = reinterpret_cast<std::uintptr_t>(devicePtr);
  de.hostPtr = nullptr;
  de.size = size;
  ptrSizeStore[devicePtr] = de;

  on_malloc(size);
}

void MemoryStore::remove_allocation(const void* ptrToRemove)
{
  ptrSizeStore.erase(ptrToRemove);
}

void MemoryStore::print_stats()
{
  size_t size = 0;
  size_t size2 = 0;
  for (auto& [dev_ptr, buffer] : ptrSizeStore) {
    size += buffer.size;
    if(buffer.size < 2*1024*1024)
      size2 += buffer.size;
  }
  spdlog::error("Total size {} {}", size/1024.0/1024.0, size2);
}

void MemoryStore::swap_in()
{
  if(!_virtual_memory_management) {
    spdlog::error("Swapping out not supported on traditional CUDA allocations!");
    return;
  }
  size_t free_mem, total_mem;
  // FIXME: check errror
  cudaMemGetInfo(&free_mem, &total_mem);
  SPDLOG_DEBUG("Before Swap In: memory free {}, occupied {}", free_mem, total_mem - free_mem);

  int streamIdx = 0;
  for (auto& [dev_ptr, de] : ptrSizeStore) {

    CHECK_CUDA(cuMemCreate(&de.alloc_handle, de.size, &de.mem_alloc_prop, 0));
    CHECK_CUDA(cuMemMap(de.devicePtr, de.size, 0, de.alloc_handle, 0));
    CUmemAccessDesc accessDesc;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = 0;
    CHECK_CUDA(cuMemSetAccess(de.devicePtr, de.size, &accessDesc, 1));

    auto ret = cudaMemcpyAsync(
      reinterpret_cast<void*>(de.devicePtr), de.hostPtr,
      de.size,
      cudaMemcpyHostToDevice,
      streams[streamIdx % numStreams]
    );
    CHECK_CUDA_E(ret);
    SPDLOG_DEBUG("MemcpyAsyncTo from {} to {} size {} ret {} stream {}", fmt::ptr(reinterpret_cast<void*>(de.hostPtr)), fmt::ptr(reinterpret_cast<void*>(de.devicePtr)), de.size, ret, streamIdx % numStreams);
    streamIdx += 1;
  }

  CHECK_CUDA_E(cudaDeviceSynchronize());

  for (auto& [devPtr, de] : ptrSizeStore) {
    cudaFreeHost(de.hostPtr);
  }

  cudaMemGetInfo(&free_mem, &total_mem);
  SPDLOG_DEBUG("After Swap In: memory free {}, occupied {}", free_mem, total_mem - free_mem);
}

void MemoryStore::swap_out()
{
  if(!_virtual_memory_management) {
    spdlog::error("Swapping out not supported on traditional CUDA allocations!");
    return;
  }
  size_t free_mem, total_mem;
  // FIXME: check errror
  cudaMemGetInfo(&free_mem, &total_mem);
  SPDLOG_DEBUG("Before: memory free {}, occupied {}", free_mem, total_mem - free_mem);

  int streamIdx = 0;
  for (auto& [devPtr, buffer] : ptrSizeStore) {
    void* tempVector;
    cudaHostAlloc(&tempVector, buffer.size, cudaHostAllocDefault);
    buffer.hostPtr = tempVector;
    auto ret = cudaMemcpyAsync(
      buffer.hostPtr, reinterpret_cast<void*>(buffer.devicePtr), buffer.size,
      cudaMemcpyDeviceToHost,
      streams[streamIdx % numStreams]
    );
    CHECK_CUDA_E(ret);
    SPDLOG_DEBUG("MemcpyAsyncTo from {} to {} size {} ret {} stream {}", fmt::ptr(reinterpret_cast<void*>(buffer.devicePtr)), fmt::ptr(reinterpret_cast<void*>(buffer.hostPtr)), buffer.size, ret, streamIdx % numStreams);
    streamIdx += 1;
  }

  CHECK_CUDA_E(cudaDeviceSynchronize());

  for (auto& [devPtr, de] : ptrSizeStore) {
    CHECK_CUDA(cuMemUnmap(de.devicePtr, de.size));
    CHECK_CUDA(cuMemRelease(de.alloc_handle));
  }

  // FIXME: check errror
  cudaMemGetInfo(&free_mem, &total_mem);
  SPDLOG_DEBUG("After: memory free {}, occupied {}", free_mem, total_mem - free_mem);
}

void MemoryStore::enable_vmm()
{
  _virtual_memory_management = true;

  _mem_alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  _mem_alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  cuMemGetAllocationGranularity(&_alloc_granularity, &_mem_alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  _access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

void MemoryStore::on_malloc(size_t size)
{
  _current_bytes += size;
  ++_current_count;
  if (_current_bytes > _peak_bytes) {
    _peak_bytes = _current_bytes;
  }
  if (_current_count > _peak_count) {
    _peak_count = _current_count;
  }
}

void MemoryStore::on_free(size_t size)
{
  _current_bytes -= size;
  --_current_count;
}

bool MemoryStore::_check_nvml_initialized()
{
  if (!_nvml_initialized) {
    nvmlReturn_t ret = nvmlInit_v2();
    if (ret != NVML_SUCCESS) {
      spdlog::error("NVML init failed: {}", nvmlErrorString(ret));
      return false;
    }
    ret = nvmlDeviceGetHandleByIndex_v2(0, &_nvml_device);
    if (ret != NVML_SUCCESS) {
      spdlog::error("NVML get device handle failed: {}", nvmlErrorString(ret));
      return false;
    }
    _nvml_initialized = true;
  }
  return true;
}

std::tuple<unsigned long long, double> MemoryStore::_nvml_used_memory() const
{
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned int process_count = 64;
  nvmlProcessInfo_t processes[64];
  nvmlReturn_t ret = nvmlDeviceGetComputeRunningProcesses_v3(_nvml_device, &process_count, processes);
  if (ret != NVML_SUCCESS) {
    spdlog::error("NVML get running processes failed: {}", nvmlErrorString(ret));
    return std::make_tuple(0, 0);
  }

  unsigned long long nvml_gpu_memory = 0;
  for (unsigned int i = 0; i < process_count; ++i) {
    if (processes[i].pid == static_cast<unsigned int>(_my_pid)) {
      nvml_gpu_memory = processes[i].usedGpuMemory;
      break;
    }
  }
  auto t_end = std::chrono::high_resolution_clock::now();

  return std::make_tuple(
    nvml_gpu_memory,
    std::chrono::duration<double, std::micro>(t_end - t_start).count()
  );
}

unsigned long long MemoryStore::nvml_used_memory()
{
  if(!_check_nvml_initialized()) {
    return 0;
  }

  auto [mem, time] = _nvml_used_memory();
  _nvml_gpu_memory = mem;
  return _nvml_gpu_memory;
}

void MemoryStore::print_memory_report()
{
  spdlog::info("Memory Report: alocated_mallocs: {} peak_mallocs: {} nvml_used: {}", _current_bytes, _peak_bytes, _nvml_gpu_memory);
  spdlog::info("Memory Report: nvml_calls: {} nvml_total_time: {} nvml_max_time: {}", _nvml_time_count, _nvml_time_us, _nvml_max_time);
}
