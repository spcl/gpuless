#include "memory_store.hpp"
#include <iostream>
#include <ostream>

MemoryStore::MemoryStore()
{
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

MemoryStore::~MemoryStore()
{
  // FIXME: release pending resources
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamDestroy(streams[i]);
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
    CHECK_CUDA(cuMemSetAccess(de.devicePtr, de.size, &accessDesc, 1));

    auto ret = cudaMemcpyAsync(
      reinterpret_cast<void*>(de.devicePtr), de.hostPtr,
      de.size,
      cudaMemcpyHostToDevice,
      streams[streamIdx % numStreams]
    );
    SPDLOG_DEBUG("MemcpyAsyncTo {} {} {}", fmt::ptr(reinterpret_cast<void*>(de.devicePtr)), de.size, ret);
    streamIdx += 1;
  }

  auto ret = cudaDeviceSynchronize();
  spdlog::error("Synchronize {}", ret);

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
    SPDLOG_DEBUG("MemcpyAsyncFrom {} {} {}", fmt::ptr(reinterpret_cast<void*>(buffer.devicePtr)), buffer.size, ret);
    streamIdx += 1;
  }

  auto ret = cudaDeviceSynchronize();

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

void MemoryStore::print_memory_report() const
{
  spdlog::info("Memory Report: {} allocated, {} peak", _current_bytes, _peak_bytes);
}
