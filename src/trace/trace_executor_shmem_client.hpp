#ifndef GPULESS_TRACE_EXECUTOR_SHMEM_H
#define GPULESS_TRACE_EXECUTOR_SHMEM_H

#include <cstdint>

#include "trace_executor.hpp"
#include "shmem/mempool.hpp"
#include "readerwriterqueue.h"

#include <mignificient/ipc/config.hpp>

#include <iceoryx_posh/popo/wait_set.hpp>
#include <iceoryx_posh/internal/runtime/posh_runtime_impl.hpp>
#include <iceoryx_posh/popo/untyped_client.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <iceoryx_hoofs/cxx/optional.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#endif

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

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    // iceoryx2 node and communication objects
    //
    std::optional<iox2::Node<iox2::ServiceType::Ipc>> iox2_node;

    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> iox2_event_notifier;
    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> iox2_event_listener;
    std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> iox2_request_notifier;
    std::optional<iox2::Listener<iox2::ServiceType::Ipc>> iox2_response_listener;

    std::optional<iox2::Publisher<iox2::ServiceType::Ipc, iox2::bb::Slice<uint8_t>, int>> iox2_request_publisher;
    std::optional<iox2::Subscriber<iox2::ServiceType::Ipc, iox2::bb::Slice<uint8_t>, int>> iox2_response_subscriber;

    std::optional<iox2::WaitSet<iox2::ServiceType::Ipc>> iox2_waitset;
    std::optional<iox2::WaitSetGuard<iox2::ServiceType::Ipc>> iox2_waitset_guard;
#endif

    // IPC backend configuration
    mignificient::ipc::IPCBackend _ipc_backend;
    mignificient::ipc::PollingMode _polling_mode;
    uint32_t _poll_interval_us;
    mignificient::ipc::BufferConfig _buffer_config;  // gpuless-server channel sizes

    int64_t last_sent = 0;
    int64_t last_synchronized = 0;

    // iceoryx2 only: store last received API call for use by send_only() and synchronize()
    std::shared_ptr<AbstractCudaApiCall> _last_api_call;

  private:
    bool negotiateSession(manager::instance_profile profile);
    bool getDeviceAttributes();

    // iceoryx2 only: helper method to receive pending responses directly in main thread
    // If blocking=true, waits until at least one response is received
    // If blocking=false, returns immediately after processing available responses
    //
    // The latter mode uses a short timeout (1us) to avoid busy-waiting
    // It seems the current implementation of iceoryx2 WaitSet does not support non-blocking waits
    void receive_pending_responses(bool blocking);

  public:

    // iceoryx1 only: background thread enqueues results here
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
