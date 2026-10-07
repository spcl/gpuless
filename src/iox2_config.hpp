#ifndef GPULESS_IOX2_CONFIG_HPP
#define GPULESS_IOX2_CONFIG_HPP

#include <cstdio>
#include <cstdlib>

#include <iox2/iceoryx2.hpp>

// The global iceoryx2 config with global.root-path = $MIGNIFICIENT_IOX2_ROOT, the client's own
// iceoryx2 directory set by the orchestrator (unset: the global config unchanged). Same as
// mignificient's executor/include/mignificient/executor/iox2_config.hpp; aborts on a bad path.
inline iox2::Config gpuless_iox2_config() {
  auto cfg = iox2::Config::global_config().to_owned();
  if (const char *root = std::getenv("MIGNIFICIENT_IOX2_ROOT")) {
    auto str = iox2::bb::StaticString<iox2::bb::platform::IOX2_MAX_PATH_LENGTH>::
        from_utf8_null_terminated_unchecked(root);
    if (!str.has_value()) {
      std::fprintf(stderr, "gpuless: iceoryx2 root path too long: %s\n", root);
      std::abort();
    }
    auto path = iox2::bb::Path::create(str.value());
    if (!path.has_value()) {
      std::fprintf(stderr, "gpuless: invalid iceoryx2 root path: %s\n", root);
      std::abort();
    }
    cfg.global().set_root_path(path.value());
  }
  return cfg;
}

#endif
