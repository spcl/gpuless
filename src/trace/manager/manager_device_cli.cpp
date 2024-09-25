
#include <cstdlib>
#include <string>

#include "manager_device.hpp"

extern const int BACKLOG = 5;

int main(int argc, char **argv) {

  std::string device{argv[1]};
  std::string manager_type{argv[2]};

  if (manager_type == "tcp") {
    if (argc != 4)
      return 1;

    int port = std::atoi(argv[3]);

    manage_device(device, port);
  } else {
    if (argc != 6)
      return 1;

    std::string app_name{argv[3]};
    std::string poll_type{argv[4]};
    const char* client_name{argv[5]};

    manage_device_shmem(device, app_name, poll_type, client_name);
  }

  return 0;
}
