#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <iox/signal_watcher.hpp>
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

  std::unique_ptr<iox::popo::UntypedPublisher> client_publisher;
  std::unique_ptr<iox::popo::UntypedSubscriber> client_subscriber;

  void setup(const std::string app_name);
  void loop(const char*);
  void loop_wait(const char*);
  void finish();

  void* take();
  void release(void*);

  bool _process_client(const void* payload);
  bool _process_remainder();
  double _sum = 0;

  double serialization_time = 0;

  std::optional<iox::SignalGuard> sigint;
  std::optional<iox::SignalGuard> sigterm;
};

#endif // __MANAGER_DEVICE_HPP__
