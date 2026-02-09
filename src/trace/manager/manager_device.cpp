#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>
#include <spdlog/spdlog.h>

#include <iceoryx_hoofs/cxx/string.hpp>
#include <iceoryx_posh/popo/untyped_server.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#endif

#include "../../schemas/trace_execution_protocol_generated.h"
#include "../../utils.hpp"
#include "../cuda_trace.hpp"
#include "../cuda_trace_converter.hpp"
#include "../shmem/mempool.hpp"
#include "flatbuffers/flatbuffers.h"
#include "iceoryx_posh/internal/popo/base_subscriber.hpp"
#include "iceoryx_posh/popo/publisher.hpp"
#include "iceoryx_posh/popo/subscriber.hpp"
#include "manager_device.hpp"
#include "memory_store.hpp"
#include "../cudnn_api_calls.hpp"
#include "../cublas_api_calls.hpp"

double serialization_time = 0.0;

extern const int BACKLOG;

static bool g_device_initialized = false;
static int64_t g_sync_counter = 0;

static gpuless::CudaTrace &getCudaTrace() {
  static gpuless::CudaTrace cuda_trace;
  return cuda_trace;
}

static CudaVirtualDevice &getCudaVirtualDevice() {
  static CudaVirtualDevice cuda_virtual_device;
  if (!g_device_initialized) {
    g_device_initialized = true;
    cuda_virtual_device.initRealDevice();
  }
  return cuda_virtual_device;
}

flatbuffers::FlatBufferBuilder
handle_attributes_request(const gpuless::FBProtocolMessage *msg,
                          int socket_fd) {
  SPDLOG_INFO("Handling device attributes request");

  auto &vdev = getCudaVirtualDevice();

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<CUdeviceAttributeValue>> attrs_vec;
  for (unsigned a = 0; a < vdev.device_attributes.size(); a++) {

    auto fb_attr = CreateCUdeviceAttributeValue(
        builder, static_cast<CUdeviceAttribute>(a), vdev.device_attributes[a]);
    attrs_vec.push_back(fb_attr);
  }

  auto attrs = gpuless::CreateFBTraceAttributeResponse(
      builder, gpuless::FBStatus_OK, vdev.device_total_mem,
      builder.CreateVector(attrs_vec));

  auto response = gpuless::CreateFBProtocolMessage(
      builder, gpuless::FBMessage_FBTraceAttributeResponse, attrs.Union());
  builder.Finish(response);

  if (socket_fd >= 0) {
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
  }

  SPDLOG_DEBUG("FBTraceAttributesResponse sent");

  return builder;
}

std::optional<flatbuffers::FlatBufferBuilder>
handle_execute_request(const gpuless::FBProtocolMessage *msg, int socket_fd) {

  SPDLOG_INFO("Handling trace execution request");
  auto &cuda_trace = getCudaTrace();
  auto &vdev = getCudaVirtualDevice();

  auto s = std::chrono::high_resolution_clock::now();

  // load new modules
  auto new_modules = msg->message_as_FBTraceExecRequest()->new_modules();
  SPDLOG_INFO("Loading {} new modules", new_modules->size());
  for (const auto &m : *new_modules) {
    CUmodule mod;
    checkCudaErrors(cuModuleLoadData(&mod, m->buffer()->data()));
    vdev.module_registry_.emplace(m->module_id(), mod);
    SPDLOG_DEBUG("Loaded module {}", m->module_id());
  }

  // load new functions
  auto new_functions = msg->message_as_FBTraceExecRequest()->new_functions();
  SPDLOG_INFO("Loading {} new functions", new_functions->size());
  for (const auto &m : *new_functions) {
    auto mod_reg_it = vdev.module_registry_.find(m->module_id());
    if (mod_reg_it == vdev.module_registry_.end()) {
      SPDLOG_ERROR("Module {} not in registry", m->module_id());
    }
    CUmodule mod = mod_reg_it->second;
    CUfunction func;
    checkCudaErrors(cuModuleGetFunction(&func, mod, m->symbol()->c_str()));
    vdev.function_registry_.emplace(m->symbol()->str(), func);
    SPDLOG_DEBUG("Function loaded: {}", m->symbol()->str());
  }

  // execute CUDA api calls
  auto p = msg->message_as_FBTraceExecRequest();
  auto call_stack = gpuless::CudaTraceConverter::execRequestToTrace(p);
  cuda_trace.setCallStack(call_stack);
  SPDLOG_INFO("Execution trace of size {}", call_stack.size());

  auto e = std::chrono::high_resolution_clock::now();
  auto d1 =
      std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
      1000000.0;

  serialization_time += d1;

  auto &instance = ExecutionStatus::instance();
  auto [begin, end] = cuda_trace.callStack();

  // SPDLOG_DEBUG("Execute callstack of size {} ", callstack.size());

  // for(size_t idx = 0; idx < callstack.size(); ++idx)
  int idx = 0;
  bool has_likely_call = false;
  for (; begin != end; ++begin) {
    // auto &apiCall = callstack[idx];
    auto &apiCall = *begin;

    SPDLOG_DEBUG("Callstack pos {}, is memop {} ", idx, apiCall->is_memop());

    if (apiCall->is_memop() && !instance.can_memcpy()) {
      spdlog::error("Blocking memory operation! Saving position {} from {}",
                    idx, cuda_trace.sizeCallStack());

      instance.save(idx);

      return std::optional<flatbuffers::FlatBufferBuilder>{};
    }

    if (apiCall->is_kernel() && !instance.can_exec_kernels()) {
      spdlog::error("Blocking kernel execution! Saving position {} from {}",
                    idx, cuda_trace.sizeCallStack());

      instance.save(idx);

      return std::optional<flatbuffers::FlatBufferBuilder>{};
    }

    SPDLOG_DEBUG("Executing: {}", apiCall->typeName());
    uint64_t err = apiCall->executeNative(vdev);
    if (err != 0) {
      SPDLOG_ERROR("Failed to execute call trace: {} ({})",
                   apiCall->nativeErrorToString(err), err);
      std::exit(EXIT_FAILURE);
    }

#if defined(MIGNIFICIENT_WITH_MEMORY_PROFILING)
    // Profiling mode: check after every API call
    {
      auto mem_result = MemoryStore::get_instance().check_memory(typeid(*apiCall).name());
      if (mem_result == MemoryCheckResult::OOM) {
        return std::nullopt;
      }
    }
#else
    if (MemoryStore::is_must_call(apiCall.get())) {
      auto mem_result = MemoryStore::get_instance().check_memory(typeid(*apiCall).name());
      if (mem_result == MemoryCheckResult::OOM) {
        return std::nullopt;
      }
    } else if (MemoryStore::is_likely_call(apiCall.get())) {
      has_likely_call = true;
    }
#endif

    ++idx;
  }

  if (has_likely_call) {
    MemoryStore::get_instance().signal_likely_check();
  }

  cuda_trace.markSynchronized();
  g_sync_counter++;
  SPDLOG_INFO("Number of synchronizations: {}", g_sync_counter);

  s = std::chrono::high_resolution_clock::now();
  flatbuffers::FlatBufferBuilder builder;
  auto top = cuda_trace.historyTop()->fbSerialize(builder);

  auto fb_trace_exec_response =
      gpuless::CreateFBTraceExecResponse(builder, gpuless::FBStatus_OK, top);
  auto fb_protocol_message = gpuless::CreateFBProtocolMessage(
      builder, gpuless::FBMessage_FBTraceExecResponse,
      fb_trace_exec_response.Union());
  builder.Finish(fb_protocol_message);

  e = std::chrono::high_resolution_clock::now();
  d1 =
      std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
      1000000.0;
  serialization_time += d1;

  instance.save(-1);

  if (socket_fd >= 0) {
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
  }

  return builder;
}

std::optional<flatbuffers::FlatBufferBuilder>
finish_trace_execution(int last_idx) {
  auto &cuda_trace = getCudaTrace();
  auto &vdev = getCudaVirtualDevice();

  auto &instance = ExecutionStatus::instance();
  // auto& callstack = cuda_trace.callStack();
  auto [begin, end] = cuda_trace.callStack();
  spdlog::error("Finishing trace execution from position {} from {}, {}",
                last_idx, cuda_trace.sizeCallStack(),
                std::distance(begin, end));
  std::advance(begin, last_idx);
  size_t idx = 0;
  bool has_likely_call = false;
  // for(size_t idx = last_idx; idx < callstack.size(); ++idx)
  for (; begin != end; ++begin) {
    // auto &apiCall = callstack[idx];
    auto &apiCall = *begin;

    if (apiCall->is_memop() && !instance.can_memcpy()) {
      spdlog::error("Blocking memory operation! Saving position {} from {}",
                    idx, cuda_trace.sizeCallStack());

      instance.save(idx);

      return std::optional<flatbuffers::FlatBufferBuilder>{};
    }

    if (apiCall->is_kernel() && !instance.can_exec_kernels()) {
      spdlog::error("Blocking kernel execution! Saving position {} from {}",
                    idx, cuda_trace.sizeCallStack());

      instance.save(idx);

      return std::optional<flatbuffers::FlatBufferBuilder>{};
    }

    SPDLOG_DEBUG("Executing: {}", apiCall->typeName());
    uint64_t err = apiCall->executeNative(vdev);
    if (err != 0) {
      SPDLOG_ERROR("Failed to execute call trace: {} ({})",
                   apiCall->nativeErrorToString(err), err);
      std::exit(EXIT_FAILURE);
    }

#if defined(MIGNIFICIENT_WITH_MEMORY_PROFILING)
    // Profiling mode: check after every API call
    {
      auto mem_result = MemoryStore::get_instance().check_memory(typeid(*apiCall).name());
      if (mem_result == MemoryCheckResult::OOM) {
        return std::nullopt;
      }
    }
#else
    if (MemoryStore::is_must_call(apiCall.get())) {
      auto mem_result = MemoryStore::get_instance().check_memory(typeid(*apiCall).name());
      if (mem_result == MemoryCheckResult::OOM) {
        return std::nullopt;
      }
    } else if (MemoryStore::is_likely_call(apiCall.get())) {
      has_likely_call = true;
    }
#endif
  }

  if (has_likely_call) {
    MemoryStore::get_instance().signal_likely_check();
  }

  cuda_trace.markSynchronized();
  g_sync_counter++;
  SPDLOG_INFO("Number of synchronizations: {}", g_sync_counter);

  auto s = std::chrono::high_resolution_clock::now();
  flatbuffers::FlatBufferBuilder builder;
  auto top = cuda_trace.historyTop()->fbSerialize(builder);

  auto fb_trace_exec_response =
      gpuless::CreateFBTraceExecResponse(builder, gpuless::FBStatus_OK, top);
  auto fb_protocol_message = gpuless::CreateFBProtocolMessage(
      builder, gpuless::FBMessage_FBTraceExecResponse,
      fb_trace_exec_response.Union());
  builder.Finish(fb_protocol_message);

  auto e = std::chrono::high_resolution_clock::now();
  auto d1 =
      std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
      1000000.0;
  serialization_time += d1;

  instance.save(-1);

  return builder;
}

void handle_request(int socket_fd) {
  while (true) {
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    if (buffer.size() == 0) {
      break;
    }

    auto msg = gpuless::GetFBProtocolMessage(buffer.data());

    if (msg->message_type() == gpuless::FBMessage_FBTraceExecRequest) {
      handle_execute_request(msg, socket_fd);
    } else if (msg->message_type() ==
               gpuless::FBMessage_FBTraceAttributeRequest) {
      handle_attributes_request(msg, socket_fd);
    } else {
      SPDLOG_ERROR("Invalid request type");
      return;
    }

  }
}

void ShmemServer::setup(const std::string app_name) {
  iox::runtime::PoshRuntime::initRuntime(
      iox::RuntimeName_t{iox::TruncateToCapacity_t{}, app_name.c_str()});
}

void *ShmemServer::take() {
  auto ptr = this->client_subscriber->take();
  if (ptr.has_error()) {
    return nullptr;
  } else {
    return const_cast<void *>(ptr.value());
  }
}

void ShmemServer::release(void *ptr) { this->client_subscriber->release(ptr); }

bool ShmemServer::_process_remainder() {
  auto &instance = ExecutionStatus::instance();
  std::optional<flatbuffers::FlatBufferBuilder> builder =
      finish_trace_execution(instance.load());

  if (builder.has_value()) {

    const void *requestPayload = instance.load_payload();

    _send_response(builder.value(), requestPayload);
    _release_request(requestPayload);

    return true;
  } else {

    // Check if this was an OOM condition
    if (MemoryStore::get_instance().is_oom()) {
      _oom_detected.store(true, std::memory_order_release);
      const void *requestPayload = instance.load_payload();
      _release_request(requestPayload);
    }

    return false;
  }
}

void ShmemServer::_send_response(const flatbuffers::FlatBufferBuilder& builder,
                                  const void* requestPayload) {
  if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
    client_publisher->loan(builder.GetSize(), alignof(1), sizeof(int), alignof(int))
        .and_then([&](auto &responsePayload) {
          memcpy(responsePayload, builder.GetBufferPointer(),
                 builder.GetSize());

          auto header = static_cast<const int*>(
              iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());
          SPDLOG_DEBUG("Reply_request {}", *header);

          auto new_header = static_cast<int*>(
              iox::mepoo::ChunkHeader::fromUserPayload(responsePayload)->userHeader());
          *new_header = *header;

          client_publisher->publish(responsePayload);
        })
        .or_else([&](auto &error) {
          std::cout << "Could not allocate Response! Error: " << error
                    << std::endl;
        });
  }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  else if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {

    auto sample = iox2_client_publisher->loan_slice(builder.GetSize());
    if (!sample.has_value()) {
      spdlog::error("Could not allocate response sample: {}", static_cast<uint64_t>(sample.error()));
      return;
    }

    SPDLOG_DEBUG("iceoryx2: Sending response size {}", builder.GetSize());

    auto payload = sample.value().payload_mut();
    if (payload.number_of_elements() >= builder.GetSize()) {
      std::memcpy(payload.data(), builder.GetBufferPointer(), builder.GetSize());
    } else {
      spdlog::error("Critical error! Sample not enough {} for {}", payload.number_of_elements(), builder.GetSize());
    }

    auto send_result = iox2::send(std::move(sample.value()));
    if (!send_result.has_value()) {
      SPDLOG_ERROR("iceoryx2: Failed to send response: {}", static_cast<uint64_t>(send_result.error()));
    }

    auto notify_result = iox2_client_notifier->notify();
    if (!notify_result.has_value()) {
      std::cout << "Could not send Request! Error: " << static_cast<uint64_t>(notify_result.error()) << std::endl;
    }

  }
#endif
  else {
    abort();
  }
}

void ShmemServer::_release_request(const void* requestPayload) {
  if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
    client_subscriber->release(requestPayload);
  }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  else {
    // iceoryx2 subscriber release - samples are auto-released after processing
    // No explicit release needed, samples are RAII-managed
    SPDLOG_DEBUG("iceoryx2: Request sample auto-released");
  }
#endif
}

bool ShmemServer::_process_client(const void *requestPayload) {
  // auto request = static_cast<const AddRequest*>(requestPayload);
  // std::cout << APP_NAME << " Got Request: " << request->augend << " + " <<
  // request->addend << std::endl;

  auto s = std::chrono::high_resolution_clock::now();
  // handle_request(s_new);
  auto msg = gpuless::GetFBProtocolMessage(requestPayload);
  auto e1 = std::chrono::high_resolution_clock::now();

   auto d1 =
       std::chrono::duration_cast<std::chrono::microseconds>(e1 - s).count() /
       1000000.0;

  // auto& instance = ExecutionStatus::status();

  // if(!instance.can_exec()) {

  //  spdlog::error("Device is locked, not executing!");

  //  instance.save(msg, requestPayload);
  //  return;
  //}

  std::optional<flatbuffers::FlatBufferBuilder> builder;
  if (msg->message_type() == gpuless::FBMessage_FBTraceExecRequest) {
    builder = handle_execute_request(msg, -1);
  } else if (msg->message_type() ==
             gpuless::FBMessage_FBTraceAttributeRequest) {
    builder = handle_attributes_request(msg, -1);
  } else {
    SPDLOG_ERROR("Invalid request type");
    return false;
  }
  // auto e = std::chrono::high_resolution_clock::now();
  // auto d =
  //     std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
  //     1000000.0;
  // auto d1 =
  //     std::chrono::duration_cast<std::chrono::microseconds>(e1 - s).count() /
  //     1000000.0;
  //_sum += d;
  // std::cerr << "replied " << d << " , " << d1 << " , total " << _sum <<
  // std::endl;

  if (builder.has_value()) {

    _send_response(builder.value(), requestPayload);
    _release_request(requestPayload);

    return true;
  } else {

    // Check if this was an OOM condition vs a blocked call
    if (MemoryStore::get_instance().is_oom()) {
      _oom_detected.store(true, std::memory_order_release);
      _release_request(requestPayload);
      return false;
    }

    ExecutionStatus::instance().save_payload(requestPayload);

    return false;
  }
}

iox::popo::WaitSet<> *SigHandler::waitset_ptr;
bool SigHandler::quit = false;

void ShmemServer::loop_wait(const char *user_name) {

  client_publisher.reset(new iox::popo::UntypedPublisher(
      {iox::RuntimeName_t{iox::TruncateToCapacity_t{}, user_name},
       "Gpuless", "Response"}));

  client_subscriber.reset(new iox::popo::UntypedSubscriber(
      {iox::RuntimeName_t{iox::TruncateToCapacity_t{}, user_name},
       "Gpuless", "Request"}));

  iox::popo::Publisher<int> orchestrator_send{iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::TruncateToCapacity_t{},
                         fmt::format("gpuless-{}", user_name).c_str()},
      "Orchestrator", "Send"}};

  iox::popo::Subscriber<int> orchestrator_recv{iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::TruncateToCapacity_t{},
                         fmt::format("gpuless-{}", user_name).c_str()},
      "Orchestrator", "Receive"}};

  iox::popo::Publisher<SwapResult> swap_result_publisher{iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::TruncateToCapacity_t{},
                         fmt::format("gpuless-{}", user_name).c_str()},
      "Orchestrator", "SwapResult"}};

  iox::popo::WaitSet<> waitset;

  SigHandler::waitset_ptr = &waitset;
  sigint.emplace(iox::registerSignalHandler(iox::PosixSignal::INT,
                                                   SigHandler::sigHandler).expect(""));
  sigterm.emplace(iox::registerSignalHandler(iox::PosixSignal::TERM,
                                                    SigHandler::sigHandler).expect(""));

  waitset.attachState(*client_subscriber, iox::popo::SubscriberState::HAS_DATA)
      .or_else([](auto) {
        std::cerr << "failed to attach server" << std::endl;
        std::exit(EXIT_FAILURE);
      });

  waitset.attachState(orchestrator_recv, iox::popo::SubscriberState::HAS_DATA)
      .or_else([](auto) {
        std::cerr << "failed to attach orchestrator subscriber" << std::endl;
        std::exit(EXIT_FAILURE);
      });

  orchestrator_send.loan().and_then([&](auto &payload) {
    *payload = static_cast<int>(GPUlessMessage::REGISTER);
    orchestrator_send.publish(std::move(payload));
  });

  std::queue<const void *> pendingPayload;
  bool has_blocked_call = false;
  auto &instance = ExecutionStatus::instance();

  int idx = 0;
  while (!SigHandler::quit) {

    // Check if background thread detected OOM
    if (_oom_detected.load(std::memory_order_acquire)) {
      spdlog::error("OOM detected, sending OUT_OF_MEMORY to orchestrator");
      orchestrator_send.loan().and_then([&](auto &payload) {
        *payload = static_cast<int>(GPUlessMessage::OUT_OF_MEMORY);
        orchestrator_send.publish(std::move(payload));
      });
      SigHandler::quit = true;
      break;
    }

    auto notificationVector = waitset.wait();
    // const void* pendingPayload = nullptr;

    for (auto &notification : notificationVector) {

      if (notification->doesOriginateFrom(client_subscriber.get())) {

        bool no_more = false;

        while (!no_more) {

          client_subscriber->take()
              .and_then([&](auto &requestPayload) {
                ++idx;

                auto new_header = static_cast<const int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());
                SPDLOG_DEBUG("Received_request {} ", *new_header);

                if (instance.can_exec() && !has_blocked_call) {
                  // has_blocked_call = !_process_client(requestPayload);
                  has_blocked_call = !_process_client(requestPayload);
                } else {
                  // pendingPayload = requestPayload;
                  // spdlog::error("Executor is blocked, appending payload");
                  pendingPayload.push(requestPayload);
                }
              })
              .or_else([&](auto &res) { no_more = true; });
        }

        // Check OOM after processing client requests
        if (_oom_detected.load(std::memory_order_acquire)) {
          spdlog::error("OOM detected after processing client, sending OUT_OF_MEMORY to orchestrator");
          orchestrator_send.loan().and_then([&](auto &payload) {
            *payload = static_cast<int>(GPUlessMessage::OUT_OF_MEMORY);
            orchestrator_send.publish(std::move(payload));
          });
          SigHandler::quit = true;
          break;
        }

        SPDLOG_INFO("Received {} requests", idx);
        //MemoryStore::get_instance().print_stats();

      } else {

        auto val = orchestrator_recv.take();

        static int count = 0;

        while (!val.has_error()) {

          int code = *val->get();
          spdlog::error("Message from the orchestrator! Code {}", code);

          auto &instance = ExecutionStatus::instance();
          if (code == static_cast<int>(GPUlessMessage::LOCK_DEVICE)) {
            instance.lock();
          } else if (code == static_cast<int>(GPUlessMessage::BASIC_EXEC)) {
            instance.basic_exec();
          } else if (code == static_cast<int>(GPUlessMessage::MEMCPY_ONLY)) {
            instance.memcpy();
          } else if (code == static_cast<int>(GPUlessMessage::FULL_EXEC)) {
            instance.exec();
            ++count;
          } else if (code == static_cast<int>(GPUlessMessage::INVOCATION_FINISH)) {
            spdlog::info("Received INVOCATION_FINISH, running final memory check");
            MemoryStore::get_instance().check_memory_final();
            MemoryStore::get_instance().print_memory_report();
          } else if (code == static_cast<int>(GPUlessMessage::SWAP_OFF)) {
            spdlog::info("[Gpuless] Received SWAP_OFF, swapping out GPU memory");
            auto mem_before = MemoryStore::get_instance().current_bytes();
            auto start = std::chrono::high_resolution_clock::now();
            MemoryStore::get_instance().swap_out();
            auto end = std::chrono::high_resolution_clock::now();
            double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            spdlog::info("[Gpuless] SWAP_OFF completed: {} bytes in {} us", mem_before, time_us);

            swap_result_publisher.loan().and_then([&](auto &sample) {
              sample->time_us = time_us;
              sample->memory_bytes = mem_before;
              sample->status = 0;
              swap_result_publisher.publish(std::move(sample));
            });
            orchestrator_send.loan().and_then([&](auto &payload) {
              *payload = static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM);
              orchestrator_send.publish(std::move(payload));
            });
          } else if (code == static_cast<int>(GPUlessMessage::SWAP_IN)) {
            spdlog::info("[Gpuless] Received SWAP_IN, swapping in GPU memory");
            auto start = std::chrono::high_resolution_clock::now();
            MemoryStore::get_instance().swap_in();
            auto end = std::chrono::high_resolution_clock::now();
            double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            auto mem_after = MemoryStore::get_instance().current_bytes();
            spdlog::info("[Gpuless] SWAP_IN completed: {} bytes in {} us", mem_after, time_us);

            swap_result_publisher.loan().and_then([&](auto &sample) {
              sample->time_us = time_us;
              sample->memory_bytes = mem_after;
              sample->status = 0;
              swap_result_publisher.publish(std::move(sample));
            });
            orchestrator_send.loan().and_then([&](auto &payload) {
              *payload = static_cast<int>(GPUlessMessage::SWAP_IN_CONFIRM);
              orchestrator_send.publish(std::move(payload));
            });
          }

          val = orchestrator_recv.take();
        }

        // spdlog::error("Has pending payload? {}", pendingPayload != nullptr);

        // spdlog::error("Has unfinished trace? {}",
        // instance.has_unfinished_trace());
        if (instance.has_unfinished_trace()) {
          // spdlog::error("Process unfinished trace from pos {}",
          // instance.load());
          has_blocked_call = !_process_remainder();
        }

        // spdlog::error("Has pending payload? {}", !pendingPayload.empty());
        //  Check if there is anything to start
        if (!pendingPayload.empty() && instance.can_exec()) {

          while (!pendingPayload.empty() && !has_blocked_call) {

            // spdlog::error("Process pending payload!");
            auto payload = pendingPayload.front();
            pendingPayload.pop();
            has_blocked_call = !_process_client(payload);
            // pendingPayload = nullptr;
          }
        }
      }
    }
  }
  spdlog::info("Received {} requests", idx);
}

void ShmemServer::loop(const char *user_name) {

  client_publisher.reset(new iox::popo::UntypedPublisher(
      {iox::RuntimeName_t{iox::TruncateToCapacity_t{}, user_name},
       "Gpuless", "Response"}));

  client_subscriber.reset(new iox::popo::UntypedSubscriber(
      {iox::RuntimeName_t{iox::TruncateToCapacity_t{}, user_name},
       "Gpuless", "Request"}));

  iox::popo::Publisher<int> orchestrator_send{iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::TruncateToCapacity_t{},
                         fmt::format("gpuless-{}", user_name).c_str()},
      "Orchestrator", "Send"}};

  iox::popo::Subscriber<int> orchestrator_recv{iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::TruncateToCapacity_t{},
                         fmt::format("gpuless-{}", user_name).c_str()},
      "Orchestrator", "Receive"}};

  iox::popo::Publisher<SwapResult> swap_result_publisher{iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::TruncateToCapacity_t{},
                         fmt::format("gpuless-{}", user_name).c_str()},
      "Orchestrator", "SwapResult"}};

  SigHandler::quit = false;
  sigint.emplace(iox::registerSignalHandler(iox::PosixSignal::INT,
                                                   SigHandler::sigHandler).expect(""));
  sigterm.emplace(iox::registerSignalHandler(iox::PosixSignal::TERM,
                                                    SigHandler::sigHandler).expect(""));

  orchestrator_send.loan().and_then([&](auto &payload) {
    *payload = static_cast<int>(GPUlessMessage::REGISTER);
    orchestrator_send.publish(std::move(payload));
  });

  std::queue<const void *> pendingPayload;
  bool has_blocked_call = false;
  auto &instance = ExecutionStatus::instance();

  while (!SigHandler::quit) {

    // Check if background thread detected OOM
    if (_oom_detected.load(std::memory_order_acquire)) {
      spdlog::error("OOM detected, sending OUT_OF_MEMORY to orchestrator");
      orchestrator_send.loan().and_then([&](auto &payload) {
        *payload = static_cast<int>(GPUlessMessage::OUT_OF_MEMORY);
        orchestrator_send.publish(std::move(payload));
      });
      SigHandler::quit = true;
      break;
    }

    bool had_activity = false;

    // Poll for client requests
    auto val = client_subscriber->take();
    while(!val.has_error()) {
      had_activity = true;
      auto requestPayload = val.value();

      auto new_header = static_cast<const int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());
      SPDLOG_DEBUG("Received_request {} ", *new_header);

      if (instance.can_exec() && !has_blocked_call) {
        has_blocked_call = !_process_client(requestPayload);
      } else {
        pendingPayload.push(requestPayload);
      }

      val = client_subscriber->take();
    }

    // Check OOM after processing client requests
    if (_oom_detected.load(std::memory_order_acquire)) {
      spdlog::error("OOM detected after processing client, sending OUT_OF_MEMORY to orchestrator");
      orchestrator_send.loan().and_then([&](auto &payload) {
        *payload = static_cast<int>(GPUlessMessage::OUT_OF_MEMORY);
        orchestrator_send.publish(std::move(payload));
      });
      SigHandler::quit = true;
      break;
    }

    // Poll for orchestrator messages
    auto orch_val = orchestrator_recv.take();
    while (!orch_val.has_error()) {
      had_activity = true;

      int code = *orch_val->get();
      spdlog::error("Message from the orchestrator! Code {}", code);

      if (code == static_cast<int>(GPUlessMessage::LOCK_DEVICE)) {
        instance.lock();
      } else if (code == static_cast<int>(GPUlessMessage::BASIC_EXEC)) {
        instance.basic_exec();
      } else if (code == static_cast<int>(GPUlessMessage::MEMCPY_ONLY)) {
        instance.memcpy();
      } else if (code == static_cast<int>(GPUlessMessage::FULL_EXEC)) {
        instance.exec();
      } else if (code == static_cast<int>(GPUlessMessage::INVOCATION_FINISH)) {
        spdlog::info("Received INVOCATION_FINISH, running final memory check");
        MemoryStore::get_instance().check_memory_final();
        MemoryStore::get_instance().print_memory_report();
      } else if (code == static_cast<int>(GPUlessMessage::SWAP_OFF)) {
        spdlog::info("[Gpuless] Received SWAP_OFF, swapping out GPU memory");
        auto mem_before = MemoryStore::get_instance().current_bytes();
        auto start = std::chrono::high_resolution_clock::now();
        MemoryStore::get_instance().swap_out();
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        spdlog::info("[Gpuless] SWAP_OFF completed: {} bytes in {} us", mem_before, time_us);

        swap_result_publisher.loan().and_then([&](auto &sample) {
          sample->time_us = time_us;
          sample->memory_bytes = mem_before;
          sample->status = 0;
          swap_result_publisher.publish(std::move(sample));
        });
        orchestrator_send.loan().and_then([&](auto &payload) {
          *payload = static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM);
          orchestrator_send.publish(std::move(payload));
        });
      } else if (code == static_cast<int>(GPUlessMessage::SWAP_IN)) {
        spdlog::info("[Gpuless] Received SWAP_IN, swapping in GPU memory");
        auto start = std::chrono::high_resolution_clock::now();
        MemoryStore::get_instance().swap_in();
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        auto mem_after = MemoryStore::get_instance().current_bytes();
        spdlog::info("[Gpuless] SWAP_IN completed: {} bytes in {} us", mem_after, time_us);

        swap_result_publisher.loan().and_then([&](auto &sample) {
          sample->time_us = time_us;
          sample->memory_bytes = mem_after;
          sample->status = 0;
          swap_result_publisher.publish(std::move(sample));
        });
        orchestrator_send.loan().and_then([&](auto &payload) {
          *payload = static_cast<int>(GPUlessMessage::SWAP_IN_CONFIRM);
          orchestrator_send.publish(std::move(payload));
        });
      }

      orch_val = orchestrator_recv.take();
    }

    // Process unfinished traces and pending payloads
    if (instance.has_unfinished_trace()) {
      has_blocked_call = !_process_remainder();
    }

    if (!pendingPayload.empty() && instance.can_exec()) {
      while (!pendingPayload.empty() && !has_blocked_call) {
        auto payload = pendingPayload.front();
        pendingPayload.pop();
        has_blocked_call = !_process_client(payload);
      }
    }

    // Sleep briefly if no activity to avoid busy-waiting
    if (!had_activity) {
      std::this_thread::sleep_for(std::chrono::microseconds(_poll_interval_us));
    }
  }
}

#ifdef MIGNIFICIENT_WITH_ICEORYX2
void ShmemServer::loop_wait_v2(const char *user_name) {
  spdlog::info("Starting iceoryx2 server loop for user {}", user_name);

  auto node_result = iox2::NodeBuilder()
    .name(iox2::NodeName::create(user_name).value())
    .create<iox2::ServiceType::Ipc>();

  if (!node_result.has_value()) {
    spdlog::error("Failed to create iceoryx2 node: {}", static_cast<uint64_t>(node_result.error()));
    std::exit(EXIT_FAILURE);
  }
  iox2_node = std::move(node_result.value());

  auto& node = iox2_node.value();
  auto& buf_cfg = _buffer_config;

  {
    auto service_name = std::string(user_name) + ".Gpuless.Recv";
    auto service_result = node.service_builder(
      iox2::ServiceName::create(service_name.c_str()).value())
      .publish_subscribe<iox2::bb::Slice<uint8_t>>()
      .user_header<int>()
      .max_publishers(1)
      .max_subscribers(1)
      .enable_safe_overflow(false)
      .subscriber_max_buffer_size(buf_cfg.queue_capacity)
      .open_or_create().value();

    //auto pub_result = service_result.publisher_builder().create();
    auto pub_result = service_result
      .publisher_builder()
      .allocation_strategy(iox2::AllocationStrategy::BestFit)
      .initial_max_slice_len(4096)
      .create();
    if (!pub_result.has_value()) {
      spdlog::error("Failed to create client publisher: {}", static_cast<uint64_t>(pub_result.error()));
      std::exit(EXIT_FAILURE);
    }
    iox2_client_publisher = std::move(pub_result.value());
  }

  {
    auto service_name = std::string(user_name) + ".Gpuless.Send";
    auto service_result = node.service_builder(
      iox2::ServiceName::create(service_name.c_str()).value())
      .publish_subscribe<iox2::bb::Slice<uint8_t>>()
      .user_header<int>()
      .max_publishers(1)
      .max_subscribers(1)
      .enable_safe_overflow(false)
      .subscriber_max_buffer_size(buf_cfg.queue_capacity)
      .open_or_create().value();

    auto pub_result = service_result.subscriber_builder().buffer_size(buf_cfg.queue_capacity).create();
    if (!pub_result.has_value()) {
      spdlog::error("Failed to create client publisher: {}", static_cast<uint64_t>(pub_result.error()));
      std::exit(EXIT_FAILURE);
    }
    iox2_client_subscriber = std::move(pub_result.value());
  }

  {
    auto exec_event_service = node.service_builder(
        iox2::ServiceName::create(fmt::format("{}.Gpuless.Listener", user_name).c_str()).value())
    .event().open_or_create();
    if (exec_event_service.has_value()) {
      iox2_client_event_notify = std::move(exec_event_service.value());
    }
  }
  {
    auto exec_event_service = node.service_builder(
        iox2::ServiceName::create(fmt::format("{}.Gpuless.Notify", user_name).c_str()).value())
    .event().open_or_create();
    if (exec_event_service.has_value()) {
      iox2_client_event_listen = std::move(exec_event_service.value());
    }
  }

  iox2_client_listener = iox2_client_event_listen->listener_builder().create().value();
  iox2_client_notifier = iox2_client_event_notify->notifier_builder().create().value();

  {
    auto exec_event_service = node.service_builder(
        iox2::ServiceName::create(fmt::format("{}.Orchestrator.Gpuless.Notify", user_name).c_str()).value())
    .event().open_or_create();
    if (exec_event_service.has_value()) {
      iox2_orchestrator_event_listen = std::move(exec_event_service.value());
    }
  }
  {
    auto exec_event_service = node.service_builder(
        iox2::ServiceName::create(fmt::format("{}.Orchestrator.Gpuless.Listen", user_name).c_str()).value())
    .event().open_or_create();
    if (exec_event_service.has_value()) {
      iox2_orchestrator_event_notify = std::move(exec_event_service.value());
    }
  }

  iox2_orchestrator_listener = iox2_orchestrator_event_listen->listener_builder().create().value();
  iox2_orchestrator_notifier = iox2_orchestrator_event_notify->notifier_builder().create().value();
  if(!iox2_orchestrator_listener.has_value() || !iox2_orchestrator_notifier.has_value()) {
    spdlog::error("Failed to create iceoryx2 listener/notifier");
    throw std::runtime_error("Failed to create iceoryx2 listener/notifier");
  }

  auto swap_result_service = node.service_builder(
      iox2::ServiceName::create(fmt::format("{}.Orchestrator.Gpuless.SwapResult", user_name).c_str()).value())
  .publish_subscribe<SwapResult>()
  .max_publishers(1)
  .max_subscribers(1)
  .open_or_create();
  if(!swap_result_service.has_value()) {
    spdlog::error("Failed to create iceoryx2 service: {}", static_cast<uint64_t>(swap_result_service.error()));
    spdlog::error("{}", iox2::PublishSubscribeOpenOrCreateError::OpenIncompatibleTypes);
    throw std::runtime_error("Failed to create iceoryx2 service");
  }

  auto iox2_swap_result_publisher = std::move(swap_result_service.value().publisher_builder().create().value());

  auto res = iox2::WaitSetBuilder()
  .signal_handling_mode(iox2::SignalHandlingMode::HandleTerminationRequests)
  .create<iox2::ServiceType::Ipc>();
  if(!res.has_value()) {
    spdlog::error("Failed to create iceoryx2 WaitSet: {}", static_cast<uint64_t>(res.error()));
    throw std::runtime_error("Failed to create iceoryx2 WaitSet");
  }
  iox2::WaitSet<iox2::ServiceType::Ipc> iox2_waitset = std::move(res.value());

  auto orchestrator_guard_res = iox2_waitset.attach_notification(iox2_orchestrator_listener.value());
  if(!orchestrator_guard_res.has_value()) {
    spdlog::error("Failed to attach orchestrator listener to waitset: {}", static_cast<uint64_t>(orchestrator_guard_res.error()));
    throw std::runtime_error("Failed to attach orchestrator listener to waitset");
  }
  auto orchestrator_guard = std::move(orchestrator_guard_res.value());

  auto client_guard_res = iox2_waitset.attach_notification(iox2_client_listener.value());
  if(!client_guard_res.has_value()) {
    spdlog::error("Failed to attach executor listener to waitset: {}", static_cast<uint64_t>(orchestrator_guard_res.error()));
    throw std::runtime_error("Failed to attach executor listener to waitset");
  }
  auto client_guard = std::move(client_guard_res.value());

  {
    // Send REGISTER message to orchestrator
    auto res = iox2_orchestrator_notifier->notify_with_custom_event_id(iox2::EventId{static_cast<int>(GPUlessMessage::REGISTER)});
    if(!res.has_value()) {
      spdlog::error("Failed to send REGISTER notification to orchestrator: {}", static_cast<uint64_t>(res.error()));
      abort();
    }
  }

  std::queue<const void*> pendingPayload;
  bool has_blocked_call = false;
  auto& instance = ExecutionStatus::instance();

  int idx = 0;

  auto loop_res = iox2_waitset.wait_and_process(
    [&](iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc> attachment_id) {

      // Check if background thread detected OOM
      if (_oom_detected.load(std::memory_order_acquire)) {
        spdlog::error("OOM detected, sending OUT_OF_MEMORY to orchestrator");
        auto res = iox2_orchestrator_notifier->notify_with_custom_event_id(
          iox2::EventId{static_cast<int>(GPUlessMessage::OUT_OF_MEMORY)});
        if(!res.has_value()) {
          spdlog::error("Failed to send OUT_OF_MEMORY: {}", static_cast<uint64_t>(res.error()));
        }
        return iox2::CallbackProgression::Stop;
      }

      if(attachment_id.has_event_from(client_guard)) {

        bool no_more = false;

        auto event_res = iox2_client_listener->try_wait_one();
        while(event_res.has_value() && event_res.value().has_value()) {

          ++idx;
          auto res = iox2_client_subscriber->receive();
          while(res.has_value() && res.value().has_value()) {

            auto payload = res.value()->payload();

            auto new_header = res.value()->user_header();
            SPDLOG_DEBUG("Received_request {} ", new_header);

            if (instance.can_exec() && !has_blocked_call) {
              has_blocked_call = !_process_client(payload.data());
            } else {
              pendingPayload.push(payload.data());
            }

            res = iox2_client_subscriber->receive();
          }

          event_res = iox2_client_listener->try_wait_one();
        }

        // Check OOM after processing client requests
        if (_oom_detected.load(std::memory_order_acquire)) {
          spdlog::error("OOM detected after processing client, sending OUT_OF_MEMORY to orchestrator");
          auto res = iox2_orchestrator_notifier->notify_with_custom_event_id(
            iox2::EventId{static_cast<int>(GPUlessMessage::OUT_OF_MEMORY)});
          if(!res.has_value()) {
            spdlog::error("Failed to send OUT_OF_MEMORY: {}", static_cast<uint64_t>(res.error()));
          }
          return iox2::CallbackProgression::Stop;
        }

        SPDLOG_INFO("Received {} requests", idx);

      } else if(attachment_id.has_event_from(orchestrator_guard)) {

        auto event_res = iox2_orchestrator_listener->try_wait_one();

        while(event_res.has_value() && event_res.value().has_value()) {

          int code = event_res.value()->as_value();
          spdlog::error("Message from the orchestrator! Code {}", code);

            auto &instance = ExecutionStatus::instance();
            if (code == static_cast<int>(GPUlessMessage::LOCK_DEVICE)) {
              instance.lock();
            } else if (code == static_cast<int>(GPUlessMessage::BASIC_EXEC)) {
              instance.basic_exec();
              MemoryStore::get_instance().print_memory_report();
            } else if (code == static_cast<int>(GPUlessMessage::MEMCPY_ONLY)) {
              instance.memcpy();
            } else if (code == static_cast<int>(GPUlessMessage::FULL_EXEC)) {
              instance.exec();
            } else if (code == static_cast<int>(GPUlessMessage::INVOCATION_FINISH)) {
              spdlog::info("Received INVOCATION_FINISH, running final memory check");
              MemoryStore::get_instance().check_memory_final();
              MemoryStore::get_instance().print_memory_report();
            } else if (code == static_cast<int>(GPUlessMessage::SWAP_OFF)) {
              spdlog::info("[Gpuless] Received SWAP_OFF, swapping out GPU memory");
              auto mem_before = MemoryStore::get_instance().current_bytes();
              auto start = std::chrono::high_resolution_clock::now();
              MemoryStore::get_instance().swap_out();
              auto end = std::chrono::high_resolution_clock::now();
              double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
              spdlog::info("[Gpuless] SWAP_OFF completed: {} bytes in {} us", mem_before, time_us);

              {
                auto sample = iox2_swap_result_publisher.loan_uninit().value();
                sample.payload_mut().time_us = time_us;
                sample.payload_mut().memory_bytes = mem_before;
                sample.payload_mut().status = 0;
                auto initialized = iox2::assume_init(std::move(sample));
                iox2::send(std::move(initialized));
              }
              auto res = iox2_orchestrator_notifier->notify_with_custom_event_id(
                iox2::EventId{static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM)});
              if(!res.has_value()) {
                spdlog::error("Failed to send SWAP_OFF_CONFIRM: {}", static_cast<uint64_t>(res.error()));
              }
            } else if (code == static_cast<int>(GPUlessMessage::SWAP_IN)) {
              spdlog::info("[Gpuless] Received SWAP_IN, swapping in GPU memory");
              auto start = std::chrono::high_resolution_clock::now();
              MemoryStore::get_instance().swap_in();
              auto end = std::chrono::high_resolution_clock::now();
              double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
              auto mem_after = MemoryStore::get_instance().current_bytes();
              spdlog::info("[Gpuless] SWAP_IN completed: {} bytes in {} us", mem_after, time_us);

              {
                auto sample = iox2_swap_result_publisher.loan_uninit().value();
                sample.payload_mut().time_us = time_us;
                sample.payload_mut().memory_bytes = mem_after;
                sample.payload_mut().status = 0;
                auto initialized = iox2::assume_init(std::move(sample));
                iox2::send(std::move(initialized));
              }
              auto res = iox2_orchestrator_notifier->notify_with_custom_event_id(
                iox2::EventId{static_cast<int>(GPUlessMessage::SWAP_IN_CONFIRM)});
              if(!res.has_value()) {
                spdlog::error("Failed to send SWAP_IN_CONFIRM: {}", static_cast<uint64_t>(res.error()));
              }
            }

          event_res = iox2_orchestrator_listener->try_wait_one();
        }

        if (instance.has_unfinished_trace()) {
          has_blocked_call = !_process_remainder();
        }

        if (!pendingPayload.empty() && instance.can_exec()) {

          while (!pendingPayload.empty() && !has_blocked_call) {

            auto payload = pendingPayload.front();
            pendingPayload.pop();
            has_blocked_call = !_process_client(payload);
          }
        }

        event_res = iox2_orchestrator_listener->try_wait_one();
      }

      return iox2::CallbackProgression::Continue;
    }
  );

  if(!loop_res.has_value()) {
    spdlog::error("Error in iceoryx2 event loop: {}", static_cast<uint64_t>(loop_res.error()));
  }
  spdlog::info("Finished iceoryx2 event loop with status {}", loop_res.value());
  spdlog::info("Received {} requests", idx);

  MemoryStore::get_instance().print_memory_report();
}
#endif

void manage_device(const std::string &device, uint16_t port) {
  setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);

  // start server
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) {
    SPDLOG_ERROR("failed to open socket");
    exit(EXIT_FAILURE);
  }

  int opt = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (void *)&opt, sizeof(opt));

  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = INADDR_ANY;
  sa.sin_port = htons(port);

  if (bind(s, (sockaddr *)&sa, sizeof(sa)) < 0) {
    SPDLOG_ERROR("failed to bind socket");
    close(s);
    exit(EXIT_FAILURE);
  }

  if (listen(s, BACKLOG) < 0) {
    std::cerr << "failed to listen on socket" << std::endl;
    close(s);
    exit(EXIT_FAILURE);
  }

  int s_new;
  sockaddr remote_addr{};
  socklen_t remote_addrlen = sizeof(remote_addr);
  while ((s_new = accept(s, &remote_addr, &remote_addrlen))) {
    SPDLOG_INFO("manager_device: connection from {}",
                inet_ntoa(((sockaddr_in *)&remote_addr)->sin_addr));

    // synchronous request handler
    handle_request(s_new);
    close(s_new);
  }

  close(s);
  exit(EXIT_SUCCESS);
}

void manage_device_shmem(const std::string &device, const std::string &app_name,
                         const std::string &poll_type, const char *user_name,
                         bool use_vmm) {

  setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);

  ShmemServer shm_server;

  // Determine IPC backend from environment variable
  const char* ipc_backend_env = std::getenv("IPC_BACKEND");
  shm_server._ipc_backend = mignificient::ipc::IPCConfig::convert_ipc_backend(ipc_backend_env);
  if (shm_server._ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {
    spdlog::info("GPUless server using iceoryx2 backend");
  } else if (shm_server._ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
    spdlog::info("GPUless server using iceoryx1 backend");
  } else {
    spdlog::error("Unknown IPC backend type specified! {}", ipc_backend_env);
    throw std::runtime_error("Unknown IPC backend type specified");
  }

  // Determine polling mode from poll_type parameter or environment variable
  if (std::string_view{poll_type} == "wait") {
    shm_server._polling_mode = mignificient::ipc::PollingMode::WAIT;
  } else {
    shm_server._polling_mode = mignificient::ipc::PollingMode::POLL;
  }

  // Environment variable override for poll interval
  const char* poll_interval_env = std::getenv("MIGNIFICIENT_POLL_INTERVAL_US");
  if (poll_interval_env) {
    shm_server._poll_interval_us = static_cast<uint32_t>(std::atoi(poll_interval_env));
  }

  spdlog::info("GPUless server polling mode: {}, interval: {}us",
               shm_server._polling_mode == mignificient::ipc::PollingMode::WAIT ? "wait" : "poll",
               shm_server._poll_interval_us);

  shm_server._buffer_config = mignificient::ipc::BufferConfig(52428800, 52428800, 5);
  const char* gpuless_req_size = std::getenv("GPULESS_REQUEST_SIZE");
  if (gpuless_req_size) {
    shm_server._buffer_config.request_size = std::stoull(gpuless_req_size);
  }
  const char* gpuless_resp_size = std::getenv("GPULESS_RESPONSE_SIZE");
  if (gpuless_resp_size) {
    shm_server._buffer_config.response_size = std::stoull(gpuless_resp_size);
  }
  const char* gpuless_queue_cap = std::getenv("GPULESS_QUEUE_CAPACITY");
  if (gpuless_queue_cap) {
    shm_server._buffer_config.queue_capacity = std::stoull(gpuless_queue_cap);
  }

  spdlog::info("GPUless server buffer config: request={}B, response={}B, capacity={}",
               shm_server._buffer_config.request_size,
               shm_server._buffer_config.response_size,
               shm_server._buffer_config.queue_capacity);

  shm_server.setup(app_name);

  if (use_vmm) {
    spdlog::error("Enabling use of VMM-based allocations in CUDA!");
    MemoryStore::get_instance().enable_vmm();
  } else {
    spdlog::error("Using traditional memory allocations in CUDA!");
  }

  // Read max GPU memory from environment
  const char* max_mem_env = std::getenv("MIGNIFICIENT_MAX_GPU_MEMORY");
  if (max_mem_env) {
    float max_memory_mb = std::stof(max_mem_env);
    size_t max_memory_bytes = static_cast<size_t>(max_memory_mb * 1024 * 1024);
    MemoryStore::get_instance().set_max_memory(max_memory_bytes);
  }

  // initialize cuda device pre-emptively
  getCudaVirtualDevice().initRealDevice();

  MemoryStore::get_instance().nvml_used_memory();

#if defined(MIGNIFICIENT_WITH_PROFILING)
  spdlog::info("Memory consumption after initializing CUDA context.");
  MemoryStore::get_instance().print_memory_report();
#endif

  if (shm_server._ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
    if (std::string_view{poll_type} == "wait") {
      shm_server.loop_wait(user_name);
    } else {
      shm_server.loop(user_name);
    }
  }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  else if (shm_server._ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {
    shm_server.loop_wait_v2(user_name);
  }
#endif
  else {
    spdlog::error("Unknown backend type!");
    abort();
  }

  gpuless::MemPoolRead::get_instance().close();

  return;
}
