#ifndef __CUDA_TRACE_HPP__
#define __CUDA_TRACE_HPP__

#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

#include "cubin_analysis.hpp"
#include "cuda_api_calls.hpp"

namespace gpuless {

class CudaTrace {
  private:
    std::vector<std::shared_ptr<AbstractCudaApiCall>> synchronized_history_;
    std::vector<std::shared_ptr<AbstractCudaApiCall>> call_stack_;

    // map of all registered symbols
    // symbol -> (module_id, is_loaded)
    std::map<std::string, std::pair<uint64_t, bool>> symbol_to_module_id_;

    // map module ids to storage location and size of a fatbin module
    // module_id -> (resource_ptr, size, is_loaded)
    std::map<uint64_t, std::tuple<void *, uint64_t, bool>>
        module_id_to_fatbin_resource_;

    size_t already_sent = 0;

  public:

    typedef std::vector<std::shared_ptr<AbstractCudaApiCall>>::iterator it_t;

    CudaTrace();

    const std::shared_ptr<AbstractCudaApiCall> &historyTop();
    void setHistoryTop(std::shared_ptr<AbstractCudaApiCall> top);
    //std::vector<std::shared_ptr<AbstractCudaApiCall>>& callStack();
    std::tuple<it_t, it_t> callStack();
    std::tuple<it_t, it_t> fullCallStack();

    std::map<std::string, std::pair<uint64_t, bool>> &getSymbolToModuleId();
    std::map<uint64_t, std::tuple<void *, uint64_t, bool>> &
    getModuleIdToFatbinResource();

    void
    setCallStack(const std::vector<std::shared_ptr<AbstractCudaApiCall>> &callStack);

    void recordFatbinData(void *data, uint64_t size, uint64_t module_id);
    void recordSymbolMapEntry(std::string &symbol, uint64_t module_id);
    void recordGlobalVarMapEntry(std::string &symbol, uint64_t module_id);

    void record(const std::shared_ptr<AbstractCudaApiCall> &cudaApiCall);
    void markSynchronized(int64_t positions = -1);

    void markSent();
};

} // namespace gpuless

#endif //  __CUDA_TRACE_HPP__
