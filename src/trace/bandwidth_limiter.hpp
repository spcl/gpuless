#ifndef GPULESS_BANDWIDTH_LIMITER_HPP
#define GPULESS_BANDWIDTH_LIMITER_HPP

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

// 
// Fair share of the host-device link: with n clients with an invocation in flight on the GPU (sent by the
// orchestrator), this server's copies get link/n per direction (the link is full duplex). Off for n <= 1 and
// without $GPULESS_LINK_GBPS.
// 
// Token bucket that may go into debt: a copy is admitted whole and charged, the next one waits until the budget is
// back to zero. The average rate holds without splitting copies; only copies above CHUNK are split, which bounds
// the burst of a single call. Unused budget accumulates for at most BURST_S.
//
// The overall algorithm:
// - sending data consumes tokens (bytes)
// - tokens regenerate by the rate of: (now() - last_comm) * rate
// - if the client is idle, it won't receive more than tokens it get can over 10 ms
// - if a large copy is executed, tokens can go negative
//
class BandwidthLimiter {
  using clock = std::chrono::steady_clock;

  struct Bucket {
    double tokens = 0;
    clock::time_point last;
  };

public:
  enum Direction { H2D = 0, D2H = 1 };

  static constexpr size_t CHUNK = 512ul << 20;
  static constexpr double BURST_S = 0.01;

  static BandwidthLimiter& instance()
  {
    static BandwidthLimiter limiter;
    return limiter;
  }

  void set_clients(int n)
  {
    _rate = (n > 1 && _link > 0) ? _link / n : 0;
    spdlog::info("[BandwidthLimiter] {} clients, limit {:.2f} GB/s per direction", n, _rate / 1e9);
  }

  // copy_part(offset, size) -> cudaError_t: one call, or CHUNK-sized parts each admitted by the limiter.
  template<typename CopyPart>
  cudaError_t copy(Direction dir, size_t size, CopyPart&& copy_part)
  {
    if(_rate <= 0) {
      return copy_part(0, size);
    }
    for(size_t offset = 0; offset < size; offset += CHUNK) {
      size_t part = std::min(CHUNK, size - offset);
      _acquire(dir, part);
      if(cudaError_t err = copy_part(offset, part); err != cudaSuccess) {
        return err;
      }
    }
    return cudaSuccess;
  }

private:
  BandwidthLimiter()
  {
    if(const char* link = std::getenv("GPULESS_LINK_GBPS")) {
      _link = std::atof(link) * 1e9;
    }
    _buckets[H2D].last = _buckets[D2H].last = clock::now();
  }

  void _acquire(Direction dir, size_t bytes)
  {
    Bucket& b = _buckets[dir];
    _refill(b);
    if(b.tokens < 0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(-b.tokens / _rate));
      _refill(b);
    }
    b.tokens -= static_cast<double>(bytes);
  }

  void _refill(Bucket& b)
  {
    auto now = clock::now();
    b.tokens = std::min(b.tokens + std::chrono::duration<double>(now - b.last).count() * _rate, _rate * BURST_S);
    b.last = now;
  }

  double _link = 0;  // bytes/s
  double _rate = 0;  // bytes/s per direction, 0: unlimited
  Bucket _buckets[2];
};

#endif
