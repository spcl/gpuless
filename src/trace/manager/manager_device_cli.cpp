
#include <cstdlib>
#include <string>

#include "manager_device.hpp"

extern const int BACKLOG = 5;

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;

  std::string device{argv[1]};
  int port = std::atoi(argv[2]);

  manage_device(device, port);

  return 0;
}
