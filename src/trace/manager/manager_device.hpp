#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include <flatbuffers/flatbuffers.h>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <iox/signal_watcher.hpp>
#include <spdlog/spdlog.h>

#include <mignificient/ipc/config.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include "iox2/iceoryx2.hpp"
#endif

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

  // IPC backend, polling, and buffer configuration
  mignificient::ipc::IPCBackend _ipc_backend = mignificient::ipc::IPCBackend::ICEORYX_V1;
  mignificient::ipc::PollingMode _polling_mode = mignificient::ipc::PollingMode::WAIT;
  uint32_t _poll_interval_us = 100;
  mignificient::ipc::BufferConfig _buffer_config;  // gpuless-server channel sizes

  std::unique_ptr<iox::popo::UntypedPublisher> client_publisher;
  std::unique_ptr<iox::popo::UntypedSubscriber> client_subscriber;

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  std::optional<iox2::Node<iox2::ServiceType::Ipc>> iox2_node;

  std::optional<iox2::Publisher<iox2::ServiceType::Ipc, iox2::bb::Slice<uint8_t>, int>> iox2_client_publisher;
  std::optional<iox2::Subscriber<iox2::ServiceType::Ipc, iox2::bb::Slice<uint8_t>, int>> iox2_client_subscriber;
  std::optional<iox2::Listener<iox2::ServiceType::Ipc>> iox2_client_listener;
  std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> iox2_client_notifier;
  std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> iox2_client_event_notify;
  std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> iox2_client_event_listen;

  std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> iox2_orchestrator_event_notify;
  std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> iox2_orchestrator_event_listen;
  std::optional<iox2::Listener<iox2::ServiceType::Ipc>> iox2_orchestrator_listener;
  std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> iox2_orchestrator_notifier;

#endif

  void setup(const std::string app_name);
  void loop(const char*);
  void loop_wait(const char*);
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  void loop_wait_v2(const char*);
#endif
  void finish();

  void* take();
  void release(void*);

  bool _process_client(const void* payload);
  bool _process_remainder();

  void _send_response(const flatbuffers::FlatBufferBuilder& builder, const void* requestPayload);
  void _release_request(const void* requestPayload);

  double _sum = 0;

  double serialization_time = 0;

  std::optional<iox::SignalGuard> sigint;
  std::optional<iox::SignalGuard> sigterm;
};

#endif // __MANAGER_DEVICE_HPP__
