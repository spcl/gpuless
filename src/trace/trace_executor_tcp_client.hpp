#ifndef GPULESS_TRACE_EXECUTOR_TCP_H
#define GPULESS_TRACE_EXECUTOR_TCP_H

#include "../TcpClient.hpp"
#include "tcp_gpu_session.hpp"
#include "trace_executor.hpp"
#include <stdexcept>

namespace gpuless {

class TraceExecutorTcp : public TraceExecutor {
  private:
    std::unique_ptr<TcpGpuSession> m_gpusession;
    sockaddr_in m_manager_addr;

    uint64_t synchronize_counter_ = 0;
    double synchronize_total_time_ = 0;

  private:
    std::unique_ptr<TcpGpuSession>
    negotiateSession(const char *ip, const short port,
                                   manager::instance_profile profile);
    bool getDeviceAttributes();

  public:
  TraceExecutorTcp();
    ~TraceExecutorTcp();

    bool init(const char *ip, short port,
            manager::instance_profile profile) override;
    bool synchronize(gpuless::CudaTrace &cuda_trace) override;
    bool deallocate() override;
    bool send_only(gpuless::CudaTrace &cuda_trace) override { throw std::runtime_error("unimplemented"); }

    double getSynchronizeTotalTime() const override;
};

} // namespace gpuless

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
