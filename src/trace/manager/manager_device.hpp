#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include <iceoryx_posh/popo/untyped_server.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_watcher.hpp>

void manage_device(const std::string& device, uint16_t port);
void manage_device_shmem(const std::string& device, const std::string& app_name, const std::string& poll_type, const char* user_name);

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

struct ShmemServer {

  std::unique_ptr<iox::popo::UntypedServer> server;

  void setup(const std::string app_name);
  void loop(const char*);
  void loop_wait(const char*);
  void finish();

  void* take();
  void release(void*);

  void _process_client(const void* payload);
  double _sum = 0;

  std::optional<iox::posix::SignalGuard> sigint;
  std::optional<iox::posix::SignalGuard> sigterm;
};

#endif // __MANAGER_DEVICE_HPP__
