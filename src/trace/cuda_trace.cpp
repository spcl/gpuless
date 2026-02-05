#include "cuda_trace.hpp"
#include "libgpuless.hpp"

#include <utility>

namespace gpuless {

CudaTrace::CudaTrace() = default;

void CudaTrace::record(
    const std::shared_ptr<AbstractCudaApiCall> &cudaApiCall) {
    this->call_stack_.push_back(cudaApiCall);

    if(executor && this->sizeCallStackNotSent() % 50 == 0) {
      executor->send_only(*this);
    }
}

void CudaTrace::markSent()
{
  already_sent = this->call_stack_.size();
}

void CudaTrace::markSynchronized(int64_t positions)
{
  // move current trace to history
  if(positions >= 0) {
    std::move(std::begin(this->call_stack_), std::begin(this->call_stack_) + positions,
              std::back_inserter(this->synchronized_history_));
  } else {
    std::move(std::begin(this->call_stack_), std::end(this->call_stack_),
              std::back_inserter(this->synchronized_history_));
  }


  // clear the current trace
  if(positions >= 0) {
    SPDLOG_INFO("Cuda trace history size: {}, already sent {}, removing {} positions.", this->synchronized_history_.size(), already_sent, positions);
    this->call_stack_.erase(this->call_stack_.begin(), this->call_stack_.begin() + positions);
    already_sent -= positions;
  } else {
    SPDLOG_INFO("Cuda trace history size: {}, clearing everything.", this->synchronized_history_.size());
    this->call_stack_.clear();
    already_sent = 0;
  }

}

const std::shared_ptr<AbstractCudaApiCall> &CudaTrace::historyTop() {
    return this->synchronized_history_.back();
}

//std::vector<std::shared_ptr<AbstractCudaApiCall>>& CudaTrace::callStack() {
//    return this->call_stack_;
//}
std::tuple<CudaTrace::it_t, CudaTrace::it_t> CudaTrace::callStack()
{
  SPDLOG_INFO("CallStack Query: already_sent {}, callstack size {}", already_sent, this->call_stack_.size());
  if(already_sent > 0) {
    return std::make_tuple(this->call_stack_.begin() + already_sent, this->call_stack_.end());
  } else {
    return std::make_tuple(this->call_stack_.begin(), this->call_stack_.end());
  }
}

size_t CudaTrace::sizeCallStack()
{
  return this->call_stack_.size();
}

size_t CudaTrace::sizeCallStackNotSent()
{
  return this->call_stack_.size() - already_sent;
}

std::tuple<CudaTrace::it_t, CudaTrace::it_t> CudaTrace::fullCallStack()
{
  return std::make_tuple(this->call_stack_.begin(), this->call_stack_.end());
}

void CudaTrace::recordFatbinData(void *data, uint64_t size,
                                 uint64_t module_id) {
    this->module_id_to_fatbin_resource_.emplace(
        module_id, std::make_tuple(data, size, false));
}

void CudaTrace::recordSymbolMapEntry(std::string &symbol, uint64_t module_id) {
    this->symbol_to_module_id_.emplace(symbol,
                                       std::make_pair(module_id, false));
}

void CudaTrace::recordGlobalVarMapEntry(std::string &symbol,
                                        uint64_t module_id) {}

std::map<std::string, std::pair<uint64_t, bool>> &
CudaTrace::getSymbolToModuleId() {
    return symbol_to_module_id_;
}

std::map<uint64_t, std::tuple<void *, uint64_t, bool>> &
CudaTrace::getModuleIdToFatbinResource() {
    return module_id_to_fatbin_resource_;
}

void CudaTrace::setHistoryTop(std::shared_ptr<AbstractCudaApiCall> top) {
    this->synchronized_history_.back() = std::move(top);
}

void CudaTrace::setCallStack(
    const std::vector<std::shared_ptr<AbstractCudaApiCall>> &callStack) {
    call_stack_ = callStack;
}

} // namespace gpuless
