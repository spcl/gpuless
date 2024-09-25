#include <iostream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>
#include <spdlog/spdlog.h>

#include <iceoryx_hoofs/cxx/string.hpp>
#include <iceoryx_posh/popo/untyped_server.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "../../schemas/trace_execution_protocol_generated.h"
#include "../../utils.hpp"
#include "../cuda_trace.hpp"
#include "../cuda_trace_converter.hpp"
#include "flatbuffers/flatbuffers.h"
#include "iceoryx_posh/internal/popo/base_subscriber.hpp"
#include "iceoryx_posh/popo/publisher.hpp"
#include "iceoryx_posh/popo/subscriber.hpp"
#include "manager_device.hpp"
#include "../shmem/mempool.hpp"

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

flatbuffers::FlatBufferBuilder handle_attributes_request(
  const gpuless::FBProtocolMessage *msg, int socket_fd
)
{
  SPDLOG_INFO("Handling device attributes request");

  auto &vdev = getCudaVirtualDevice();

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<CUdeviceAttributeValue>> attrs_vec;
  for (unsigned a = 0; a < vdev.device_attributes.size(); a++) {

    auto fb_attr = CreateCUdeviceAttributeValue(
      builder, static_cast<CUdeviceAttribute>(a), vdev.device_attributes[a]
    );
    attrs_vec.push_back(fb_attr);
  }

  auto attrs = gpuless::CreateFBTraceAttributeResponse(
    builder, gpuless::FBStatus_OK, vdev.device_total_mem,
    builder.CreateVector(attrs_vec)
  );

  auto response = gpuless::CreateFBProtocolMessage(
    builder, gpuless::FBMessage_FBTraceAttributeResponse, attrs.Union()
  );
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

    auto& instance = ExecutionStatus::instance();
    auto& callstack = cuda_trace.callStack();

    for(size_t idx = 0; idx < callstack.size(); ++idx)
    {
      auto &apiCall = callstack[idx];

      if(apiCall->is_memop() && !instance.can_memcpy()) {
        spdlog::error("Blocking memory operation!");

        instance.save(idx);

        return std::optional<flatbuffers::FlatBufferBuilder>{};
      }

      if(apiCall->is_kernel() && !instance.can_exec_kernels()) {
        spdlog::error("Blocking kernel execution!");

        instance.save(idx);

        return std::optional<flatbuffers::FlatBufferBuilder>{};
      }

      SPDLOG_DEBUG("Executing: {}", apiCall->typeName());
      uint64_t err = apiCall->executeNative(vdev);
      if (err != 0) {
        SPDLOG_ERROR("Failed to execute call trace: {} ({})", apiCall->nativeErrorToString(err), err);
        std::exit(EXIT_FAILURE);
      }
    }

    cuda_trace.markSynchronized();
    g_sync_counter++;
    SPDLOG_INFO("Number of synchronizations: {}", g_sync_counter);

    flatbuffers::FlatBufferBuilder builder;
    auto top = cuda_trace.historyTop()->fbSerialize(builder);

    auto fb_trace_exec_response =
        gpuless::CreateFBTraceExecResponse(builder, gpuless::FBStatus_OK, top);
    auto fb_protocol_message = gpuless::CreateFBProtocolMessage(
        builder, gpuless::FBMessage_FBTraceExecResponse,
        fb_trace_exec_response.Union());
    builder.Finish(fb_protocol_message);

    instance.save(-1);

    if(socket_fd >= 0) {
      send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
    }

    return builder;
}

std::optional<flatbuffers::FlatBufferBuilder> finish_trace_execution(int last_idx)
{
  auto &cuda_trace = getCudaTrace();
  auto &vdev = getCudaVirtualDevice();

  auto& instance = ExecutionStatus::instance();
  auto& callstack = cuda_trace.callStack();
  for(size_t idx = last_idx; idx < callstack.size(); ++idx)
  {
    auto &apiCall = callstack[idx];

    if(apiCall->is_memop() && !instance.can_memcpy()) {
      spdlog::error("Blocking memory operation!");

      instance.save(idx);

      return std::optional<flatbuffers::FlatBufferBuilder>{};
    }

    if(apiCall->is_kernel() && !instance.can_exec_kernels()) {
      spdlog::error("Blocking kernel execution!");

      instance.save(idx);

      return std::optional<flatbuffers::FlatBufferBuilder>{};
    }

    SPDLOG_DEBUG("Executing: {}", apiCall->typeName());
    uint64_t err = apiCall->executeNative(vdev);
    if (err != 0) {
      SPDLOG_ERROR("Failed to execute call trace: {} ({})", apiCall->nativeErrorToString(err), err);
      std::exit(EXIT_FAILURE);
    }
  }

  cuda_trace.markSynchronized();
  g_sync_counter++;
  SPDLOG_INFO("Number of synchronizations: {}", g_sync_counter);

  flatbuffers::FlatBufferBuilder builder;
  auto top = cuda_trace.historyTop()->fbSerialize(builder);

  auto fb_trace_exec_response =
      gpuless::CreateFBTraceExecResponse(builder, gpuless::FBStatus_OK, top);
  auto fb_protocol_message = gpuless::CreateFBProtocolMessage(
      builder, gpuless::FBMessage_FBTraceExecResponse,
      fb_trace_exec_response.Union());
  builder.Finish(fb_protocol_message);

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
      iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, app_name});
}

void *ShmemServer::take() {
  auto ptr = this->server->take();
  if (ptr.has_error()) {
    return nullptr;
  } else {
    return const_cast<void *>(ptr.value());
  }
}

void ShmemServer::release(void *ptr) { this->server->releaseRequest(ptr); }

void ShmemServer::_process_remainder()
{
  auto& instance = ExecutionStatus::instance();
  std::optional<flatbuffers::FlatBufferBuilder> builder = finish_trace_execution(instance.load());

  if(builder.has_value()) {

    const void* requestPayload = instance.load_payload();

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    server->loan(requestHeader, sizeof(builder->GetSize()), alignof(1))
        .and_then([&](auto &responsePayload) {
          memcpy(responsePayload, builder->GetBufferPointer(), builder->GetSize());
          // auto response = static_cast<AddResponse*>(responsePayload);
          // response->sum = request->augend + request->addend;
          // std::cout << APP_NAME << " Send Response: " << response->sum <<
          // std::endl;
          server->send(responsePayload).or_else([&](auto &error) {
            std::cout << "Could not send Response! Error: " << error << std::endl;
          });
        })
        .or_else([&](auto &error) {
          std::cout << "Could not allocate Response! Error: " << error
                    << std::endl;
        });

    server->releaseRequest(requestPayload);
  }
}

void ShmemServer::_process_client(const void *requestPayload) {
  // auto request = static_cast<const AddRequest*>(requestPayload);
  // std::cout << APP_NAME << " Got Request: " << request->augend << " + " <<
  // request->addend << std::endl;

  // auto s = std::chrono::high_resolution_clock::now();
  // handle_request(s_new);
  auto msg = gpuless::GetFBProtocolMessage(requestPayload);
  // auto e1 = std::chrono::high_resolution_clock::now();

  // std::cerr << "Request" << std::endl;


  //auto& instance = ExecutionStatus::status();

  //if(!instance.can_exec()) {

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
    return;
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

  if(builder.has_value()) {

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    server->loan(requestHeader, sizeof(builder->GetSize()), alignof(1))
        .and_then([&](auto &responsePayload) {
          memcpy(responsePayload, builder->GetBufferPointer(), builder->GetSize());
          // auto response = static_cast<AddResponse*>(responsePayload);
          // response->sum = request->augend + request->addend;
          // std::cout << APP_NAME << " Send Response: " << response->sum <<
          // std::endl;
          server->send(responsePayload).or_else([&](auto &error) {
            std::cout << "Could not send Response! Error: " << error << std::endl;
          });
        })
        .or_else([&](auto &error) {
          std::cout << "Could not allocate Response! Error: " << error
                    << std::endl;
        });

    server->releaseRequest(requestPayload);
  } else {

    ExecutionStatus::instance().save_payload(requestPayload);
  }
}

iox::popo::WaitSet<>* SigHandler::waitset_ptr;
bool SigHandler::quit = false;

void ShmemServer::loop_wait(const char* user_name)
{

  server.reset(new iox::popo::UntypedServer({
    iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, user_name},
    "Gpuless",
    "Client"
  }));

  iox::popo::Publisher<int> orchestrator_send{
    iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, fmt::format("gpuless-{}", user_name)},
      "Orchestrator",
      "Send"
    }
  };

  iox::popo::Subscriber<int> orchestrator_recv{
    iox::capro::ServiceDescription{
      iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, fmt::format("gpuless-{}", user_name)},
      "Orchestrator",
      "Receive"
    }
  };

  iox::popo::WaitSet<> waitset;

  SigHandler::waitset_ptr = &waitset;
  sigint.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::INT, SigHandler::sigHandler));
  sigterm.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::TERM, SigHandler::sigHandler));

  waitset.attachState(*server, iox::popo::ServerState::HAS_REQUEST)
      .or_else([](auto) {
        std::cerr << "failed to attach server" << std::endl;
        std::exit(EXIT_FAILURE);
      });

  waitset.attachEvent(orchestrator_recv, iox::popo::SubscriberEvent::DATA_RECEIVED)
      .or_else([](auto) {
        std::cerr << "failed to attach orchestrator subscriber" << std::endl;
        std::exit(EXIT_FAILURE);
      });

  orchestrator_send.loan().and_then(
    [&](auto& payload) {

      *payload = static_cast<int>(GPUlessMessage::REGISTER);
      orchestrator_send.publish(std::move(payload));
    }
  );

  while (!SigHandler::quit) {

    auto notificationVector = waitset.wait();

    auto& instance = ExecutionStatus::instance();
    const void* pendingPayload = nullptr;

    for (auto &notification : notificationVector) {

      if (notification->doesOriginateFrom(server.get())) {

        server->take().and_then(
          [&](auto &requestPayload) {

            if(instance.can_exec()) {
              _process_client(requestPayload);
            } else {
              pendingPayload = requestPayload;
            }
          }
        );

      } else {

        int code = *orchestrator_recv.take()->get();
        spdlog::error("Message from the orchestrator! Code {}", code);

        auto& instance = ExecutionStatus::instance();
        if(code == static_cast<int>(GPUlessMessage::LOCK_DEVICE)) {
          instance.lock();
        } else if(code == static_cast<int>(GPUlessMessage::BASIC_EXEC)) {
          instance.basic_exec();
        } else if(code == static_cast<int>(GPUlessMessage::MEMCPY_ONLY)) {
          instance.memcpy();
        } else if(code == static_cast<int>(GPUlessMessage::FULL_EXEC)) {
          instance.exec();
        }

        // Check if there is anything to start
        if(pendingPayload && instance.can_exec()) {
          spdlog::error("Process pending payload!");
          _process_client(pendingPayload);
          pendingPayload = nullptr;
        }

        if(instance.has_unfinished_trace()) {
          spdlog::error("Process unfinished trace from pos {}", instance.load());
          _process_remainder();
        }

      }
    }
  }
}

void ShmemServer::loop(const char* user_name)
{

  // FIXME: add here communication with orchestrato
  server.reset(new iox::popo::UntypedServer({
    iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, user_name},
    "Gpuless",
    "Client"
  }));

  double sum = 0;
  while (!iox::posix::hasTerminationRequested()) {

    server->take().and_then(
      [&](auto &requestPayload) {
        _process_client(requestPayload);
      }
    );
  }
}

void manage_device(const std::string& device, uint16_t port) {
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
                         const std::string &poll_type, const char* user_name)
{

  setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);

  ShmemServer shm_server;

  shm_server.setup(app_name);

  // initialize cuda device pre-emptively
  getCudaVirtualDevice().initRealDevice();

  if (std::string_view{poll_type} == "wait") {
    shm_server.loop_wait(user_name);
  } else {
    shm_server.loop(user_name);
  }

  gpuless::MemPoolRead::get_instance().close();

  return;
}
