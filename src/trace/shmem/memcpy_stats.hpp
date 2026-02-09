#ifndef __GPULESS_MEMCPY_STATS_HPP__
#define __GPULESS_MEMCPY_STATS_HPP__

#include <chrono>
#include <cstddef>
#include <spdlog/spdlog.h>

struct MemcpyCopyStats {
  static MemcpyCopyStats& instance() {
    static MemcpyCopyStats s;
    return s;
  }

  size_t h2d_count = 0;
  size_t h2d_bytes = 0;
  double h2d_time_us = 0;

  size_t d2h_count = 0;
  size_t d2h_bytes = 0;
  double d2h_time_us = 0;

  void record_h2d(size_t bytes, double time_us) {
    h2d_count++;
    h2d_bytes += bytes;
    h2d_time_us += time_us;
  }

  void record_d2h(size_t bytes, double time_us) {
    d2h_count++;
    d2h_bytes += bytes;
    d2h_time_us += time_us;
  }

  void print_and_reset() {
    spdlog::info("[MemcpyStats] H2D: count={} bytes={} time={:.1f}us | D2H: count={} bytes={} time={:.1f}us",
      h2d_count, h2d_bytes, h2d_time_us,
      d2h_count, d2h_bytes, d2h_time_us);
    h2d_count = h2d_bytes = d2h_count = d2h_bytes = 0;
    h2d_time_us = d2h_time_us = 0;
  }
};

// C-linkage function exported from libgpuless.so so that the executor
// (which lives in a separate binary but same process via LD_PRELOAD)
// can access the singleton that libgpuless records into.
extern "C" void mignificient_memcpy_stats_print_and_reset();

#endif
