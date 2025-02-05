
#include <cstdlib>
#include <string>

#include <sys/prctl.h>

#include "manager_device.hpp"

#include <spdlog/spdlog.h>

extern const int BACKLOG = 5;

int main(int argc, char **argv) {

  spdlog::set_level(spdlog::level::trace);
  std::string device{argv[1]};
  std::string manager_type{argv[2]};

  char* cpu_idx = std::getenv("CPU_BIND_IDX");
  if(cpu_idx) {
    int idx = std::atoi(cpu_idx);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(idx, &set);
    pid_t pid = getpid();

    spdlog::info("Setting CPU to: {}", idx);
    if(sched_setaffinity(pid, sizeof(set), &set) == -1) {
			spdlog::error("Couldn't set the CPU affinity! Error {}", strerror(errno));
			exit(EXIT_FAILURE);
    }
  }

  prctl(PR_SET_PDEATHSIG, SIGHUP);

  int cpu = sched_getcpu();
  spdlog::info("Running on CPU: {}", cpu);

  if (manager_type == "tcp") {
    if (argc != 4)
      return 1;

    int port = std::atoi(argv[3]);

    manage_device(device, port);
  } else {
    if (argc != 7)
      return 1;

    std::string app_name{argv[3]};
    std::string poll_type{argv[4]};
    const char* client_name{argv[5]};
    bool use_vmm = false;
    if(argc == 7) {
      use_vmm = std::stoi(argv[6]);
    }

    manage_device_shmem(device, app_name, poll_type, client_name, use_vmm);
  }

  return 0;
}
