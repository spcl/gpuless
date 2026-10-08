#ifndef GPULESS_IOX2_CONFIG_HPP
#define GPULESS_IOX2_CONFIG_HPP

#include <cstdio>
#include <cstdlib>

#include <iox2/iceoryx2.hpp>

// The iceoryx2 config from $MIGNIFICIENT_IOX2_CONFIG (unset: the global one) with
// global.root-path = $MIGNIFICIENT_IOX2_ROOT, the client's own iceoryx2 directory set by the
// orchestrator (unset: root unchanged). Same as
// mignificient's executor/include/mignificient/executor/iox2_config.hpp; aborts on a bad path.
inline iox2::Config gpuless_iox2_config() {
  auto cfg = iox2::Config::global_config().to_owned();
  if (const char *file = std::getenv("MIGNIFICIENT_IOX2_CONFIG")) {
    auto str = iox2::bb::StaticString<iox2::bb::platform::IOX2_MAX_PATH_LENGTH>::
        from_utf8_null_terminated_unchecked(file);
    if (!str.has_value()) {
      std::fprintf(stderr, "gpuless: iceoryx2 config path too long: %s\n", file);
      std::abort();
    }
    auto path = iox2::bb::FilePath::create(str.value());
    if (!path.has_value()) {
      std::fprintf(stderr, "gpuless: invalid iceoryx2 config path: %s\n", file);
      std::abort();
    }
    auto loaded = iox2::Config::from_file(path.value());
    if (!loaded.has_value()) {
      std::fprintf(stderr, "gpuless: cannot load iceoryx2 config %s\n", file);
      std::abort();
    }
    cfg = std::move(loaded.value());
  }
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
