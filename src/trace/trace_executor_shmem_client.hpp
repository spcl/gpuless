#ifndef GPULESS_TRACE_EXECUTOR_SHMEM_H
#define GPULESS_TRACE_EXECUTOR_SHMEM_H

#include <cstdint>

#include "trace_executor.hpp"
#include "shmem/mempool.hpp"

#include "readerwriterqueue.h"

#include <iceoryx_posh/popo/wait_set.hpp>
#include <iceoryx_posh/internal/runtime/posh_runtime_impl.hpp>
#include <iceoryx_posh/popo/untyped_client.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <iceoryx_hoofs/cxx/optional.hpp>

namespace gpuless {

class TraceExecutorShmem : public TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};
    int32_t session_id_ = -1;
    uint64_t synchronize_counter_ = 0;
    double synchronize_total_time_ = 0;

    // Not great - internal feature - but we don't have a better solution.
    std::unique_ptr<iox::runtime::PoshRuntimeImpl> _impl;
    std::unique_ptr<iox::popo::UntypedPublisher> request_publisher;
    std::unique_ptr<iox::popo::UntypedSubscriber> request_subscriber;
    std::optional<iox::popo::WaitSet<>> waitset;

    bool wait_poll;

    int64_t last_sent = 0;
    int64_t last_synchronized = 0;

  private:
    bool negotiateSession(manager::instance_profile profile);
    bool getDeviceAttributes();

  public:

    moodycamel::BlockingReaderWriterQueue<std::pair<std::shared_ptr<AbstractCudaApiCall>, int>> results;

    double serialize_total_time = 0;

    MemPool _pool;
    TraceExecutorShmem();
    ~TraceExecutorShmem();

    //static void init_runtime();
    //static void reset_runtime();

    bool init(const char *ip, short port,
              manager::instance_profile profile) override;
    bool synchronize(gpuless::CudaTrace &cuda_trace) override;
    bool send_only(gpuless::CudaTrace &cuda_trace) override;
    bool deallocate() override;

    double getSynchronizeTotalTime() const override;

    static iox::runtime::PoshRuntime* runtime_factory_impl(iox::cxx::optional<const iox::RuntimeName_t*> var, TraceExecutorShmem* ptr = nullptr);
};

} // namespace gpuless

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
