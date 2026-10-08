#include "trace_executor_shmem_client.hpp"
#include "../schemas/allocation_protocol_generated.h"
#include "cuda_trace_converter.hpp"
#include "../iox2_config.hpp"

#include <iox2/waitset_enums.hpp>
#include <spdlog/spdlog.h>

#include <iceoryx_posh/runtime/posh_runtime.hpp>
#include <iceoryx_hoofs/cxx/expected.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include "iox2/iceoryx2.hpp"
#endif
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <chrono>

namespace gpuless {

// FIXME: Runtime factory uses protected iceoryx APIs - commented out for now
// This was only needed for LD_PRELOAD scenarios without executor
//iox::runtime::PoshRuntime* TraceExecutorShmem::runtime_factory_impl(iox::cxx::optional<const iox::RuntimeName_t*> var, TraceExecutorShmem* ptr)
//{
//    static TraceExecutorShmem* obj_ptr = nullptr;
//    if(ptr) {
//        obj_ptr = ptr;
//        return nullptr;
//    } else if (var.has_value()) {
//        obj_ptr->_impl = std::make_unique<iox::runtime::PoshRuntimeImpl>(var);
//        return obj_ptr->_impl.get();
//    } else {
//        return obj_ptr->_impl.get();
//    }
//}
//
//iox::runtime::PoshRuntime& runtime_factory(iox::cxx::optional<const iox::RuntimeName_t*> var)
//{
//    return *TraceExecutorShmem::runtime_factory_impl(var, nullptr);
//}

TraceExecutorShmem::TraceExecutorShmem():
    _ipc_backend(mignificient::ipc::IPCBackend::ICEORYX_V1),
    _polling_mode(mignificient::ipc::PollingMode::WAIT),
    _poll_interval_us(100)
{
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%H:%M:%S:%e:%f] %v");

  // This is useful when we do not have executor, i.e., we just do LD_PRELOAD on existing app.
  const char* app_name = std::getenv("SHMEM_APP_NAME");
  const char* user_name = std::getenv("CONTAINER_NAME");

  _pool.set_user_name(user_name);

  // FIXME: Runtime factory uses protected iceoryx APIs - commented out for now
  // This was only needed for LD_PRELOAD scenarios without executor
  //if(app_name) {
  //  iox::runtime::PoshRuntime::setRuntimeFactory(runtime_factory);
  //  runtime_factory_impl(nullptr, this);
  //  iox::runtime::PoshRuntime::initRuntime(
  //  iox::RuntimeName_t{iox::TruncateToCapacity_t{}, app_name}
  //  );
  //}

  // Determine IPC backend from environment variable
  const char* ipc_backend_env = std::getenv("IPC_BACKEND");
  if (ipc_backend_env && std::string_view{ipc_backend_env} == "iceoryx2") {
    _ipc_backend = mignificient::ipc::IPCBackend::ICEORYX_V2;
  } else  if (ipc_backend_env && std::string_view{ipc_backend_env} == "iceoryx1") {
    _ipc_backend = mignificient::ipc::IPCBackend::ICEORYX_V1;
  } else {
    spdlog::error("Unknown backend type! {}", ipc_backend_env ? ipc_backend_env : "null");
    abort();
  }

  // Check environment variable for polling mode (backwards compatibility with POLL_TYPE)
  const char* poll_type_env = std::getenv("POLL_TYPE");
  if(poll_type_env && std::string_view{poll_type_env} == "wait") {
      _polling_mode = mignificient::ipc::PollingMode::WAIT;
  } else {
      _polling_mode = mignificient::ipc::PollingMode::POLL;
  }

  if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {

    request_publisher.reset(new iox::popo::UntypedPublisher({
          iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, user_name},
          "Gpuless",
          "Request"
          }
          }));

    request_subscriber.reset(new iox::popo::UntypedSubscriber({
          iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, user_name},
          "Gpuless",
          "Response"
          }
          }));

    waitset.emplace();

    // For wait mode, attach subscriber to waitset
    if(_polling_mode == mignificient::ipc::PollingMode::WAIT) {
      waitset.value().attachState(*request_subscriber, iox::popo::SubscriberState::HAS_DATA).or_else([](auto) {
          spdlog::error("failed to attach server");
          std::exit(EXIT_FAILURE);
      });
    }
  }

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  // Create iceoryx2 publishers/subscribers if backend is iceoryx2
  else if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {

    //iox2::set_log_level(iox2::LogLevel::Trace);

    // Environment variable override for poll interval
    const char* poll_interval_env = std::getenv("MIGNIFICIENT_POLL_INTERVAL_US");
    if (poll_interval_env) {
        _poll_interval_us = static_cast<uint32_t>(std::atoi(poll_interval_env));
    }

    _buffer_config = mignificient::ipc::BufferConfig(52428800, 52428800, 5);
    const char* gpuless_req_size = std::getenv("GPULESS_REQUEST_SIZE");
    if (gpuless_req_size) {
      _buffer_config.request_size = std::stoull(gpuless_req_size);
    }
    const char* gpuless_resp_size = std::getenv("GPULESS_RESPONSE_SIZE");
    if (gpuless_resp_size) {
      _buffer_config.response_size = std::stoull(gpuless_resp_size);
    }
    const char* gpuless_queue_cap = std::getenv("GPULESS_QUEUE_CAPACITY");
    if (gpuless_queue_cap) {
      _buffer_config.queue_capacity = std::stoull(gpuless_queue_cap);
    }

    {
      // libgpuless is preloaded into someone else's process: never install
      // SIGINT/SIGTERM handlers (they would replace e.g. Python's KeyboardInterrupt).
      iox2_config.emplace(gpuless_iox2_config());
      auto node_result_res = iox2::NodeBuilder()
        .config(*iox2_config)
        .signal_handling_mode(iox2::SignalHandlingMode::Disabled)
        .create<iox2::ServiceType::Ipc>();
      if (!node_result_res.has_value()) {
        spdlog::error("Cannot allocate iceoryx2 node! Error: {}", node_result_res.error());
      }
      iox2_node = std::move(node_result_res.value());
    }

    auto& node = iox2_node.value();
    auto& buf_cfg = _buffer_config;

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
        .open_or_create();

      if (!service_result.has_value()) {

        if(service_result.error() == iox2::PublishSubscribeOpenOrCreateError::OpenDoesNotSupportRequestedMinBufferSize) {
          spdlog::error("Failed to create client service for subscriber - cannot suport buffer size {}", buf_cfg.queue_capacity);
        } else {
          spdlog::error("Failed to create client service for subscriber: {}", static_cast<uint64_t>(service_result.error()));
        }

        std::exit(EXIT_FAILURE);
      }

      iox2_request_service = std::move(service_result.value());
      auto pub_result = iox2_request_service
        ->publisher_builder()
        .allocation_strategy(iox2::AllocationStrategy::BestFit)
        .initial_max_slice_len(4096)
        .create();

      if (!pub_result.has_value()) {
        spdlog::error("Failed to create publisher: {}", static_cast<uint64_t>(pub_result.error()));
      }
      iox2_request_publisher = std::move(pub_result.value());
      auto sample = iox2_request_publisher.value().loan_slice_uninit(1024);
    }

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
        .open_or_create();

      if (!service_result.has_value()) {

        if(service_result.error() == iox2::PublishSubscribeOpenOrCreateError::OpenDoesNotSupportRequestedMinBufferSize) {
          spdlog::error("Failed to create client service for subscriber - cannot suport buffer size {}", buf_cfg.queue_capacity);
        } else {
          spdlog::error("Failed to create client service for subscriber: {}", static_cast<uint64_t>(service_result.error()));
        }

        std::exit(EXIT_FAILURE);
      }

      auto sub_result = service_result.value().subscriber_builder().buffer_size(buf_cfg.queue_capacity).create();
      if (!sub_result.has_value()) {
        spdlog::error("Failed to create publisher: {}", static_cast<uint64_t>(sub_result.error()));
      }
      iox2_response_subscriber = std::move(sub_result.value());
    }

    // Events are needed in both modes: every request path notifies and waits on
    // them, and the iceoryx2 server always waits for events.
    //
    // TODO: for the future - do we need proper poll instead of wait?
    // Note: on iceoryx2, POLL currently behaves like WAIT; a busy-poll receive path is not implemented.
    {

      {
        auto exec_event_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Gpuless.Notify", user_name).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          iox2_event_notifier = std::move(exec_event_service.value());
        }
      }

      {
        auto exec_event_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Gpuless.Listener", user_name).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          iox2_event_listener = std::move(exec_event_service.value());
        }
      }

      iox2_response_listener = iox2_event_listener->listener_builder().create().value();
      iox2_request_notifier = iox2_event_notifier->notifier_builder().create().value();

      auto waitset_result = iox2::WaitSetBuilder()
        .signal_handling_mode(iox2::SignalHandlingMode::Disabled)
        .create<iox2::ServiceType::Ipc>();
      if (waitset_result.has_value()) {
        iox2_waitset = std::move(waitset_result.value());

        iox2_waitset_guard = iox2_waitset.value().attach_notification(iox2_response_listener.value()).value();
      }
    }
  }
#endif
  else {
    throw std::runtime_error{"Unknown backend type!"};
  }

}

TraceExecutorShmem::~TraceExecutorShmem()
{
  spdlog::debug("Total serialize_total_time {}", serialize_total_time);
}

namespace {
// How long the client waits for the gpuless server's endpoints to appear.
constexpr auto SERVER_WAIT_LIMIT = std::chrono::seconds(30);

[[noreturn]] void fatal(const std::string &msg) {
  spdlog::error("gpuless: {}", msg);
  spdlog::default_logger()->flush();
  std::abort();
}
} // namespace

// The server may start after the client (e.g. a Python function importing torch
// at module level). A request published before the server's subscriber and
// listener exist is dropped, and the client would then wait forever. So wait
// until they exist, polling with backoff (1 ms doubling to 100 ms).
bool TraceExecutorShmem::server_connected() {
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {
    // A SIGKILLed server's ports stay registered until someone cleans up its dead node; iceoryx2 knows
    // it is dead (its monitor lock is gone), so remove its stale resources before counting.
    iox2::Node<iox2::ServiceType::Ipc>::list(iox2_config->view(), [](auto state) {
      state.dead([](auto& view) { (void)view.remove_stale_resources(); });
      return iox2::CallbackProgression::Continue;
    });
    return iox2_request_service->dynamic_config().number_of_subscribers() > 0 &&
           iox2_event_notifier->dynamic_config().number_of_listeners() > 0;
  }
#endif
  return request_publisher->hasSubscribers();
}

bool TraceExecutorShmem::wait_for_server() {
  auto start = std::chrono::steady_clock::now();
  auto backoff = std::chrono::milliseconds(1);
  while (!server_connected()) {
    if (std::chrono::steady_clock::now() - start >= SERVER_WAIT_LIMIT) {
      spdlog::error("gpuless: no gpuless server for {} after {} s; is manager_device running for this client?",
                    std::getenv("CONTAINER_NAME") ? std::getenv("CONTAINER_NAME") : "null",
                    SERVER_WAIT_LIMIT.count());
      return false;
    }
    std::this_thread::sleep_for(backoff);
    backoff = std::min(backoff * 2, std::chrono::milliseconds(100));
  }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {
    iox2_request_publisher->update_connections();
  }
#endif
  auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();
  if (waited > 0) {
    spdlog::info("gpuless: server ready after {} ms", waited);
  }
  return true;
}

#ifdef MIGNIFICIENT_WITH_ICEORYX2
void TraceExecutorShmem::send_request_iox2(flatbuffers::FlatBufferBuilder &builder) {
  auto sample = iox2_request_publisher->loan_slice_uninit(builder.GetSize());
  if (!sample.has_value()) {
    spdlog::error("iceoryx2: Could not allocate Request! Error: {}", static_cast<uint64_t>(sample.error()));
    throw std::runtime_error{"could not allocate sample"};
  }

  auto payload = sample.value().payload_mut();
  std::memcpy(payload.data(), builder.GetBufferPointer(), builder.GetSize());

  SPDLOG_DEBUG("iceoryx2: Submit_request {}, size {}", last_sent, builder.GetSize());

  auto initialized_sample = iox2::assume_init(std::move(sample.value()));
  initialized_sample.user_header_mut() = ++last_sent;

  // The server's subscriber existed before the first request (wait_for_server).
  // No recipient now means it is gone and no response will ever come.
  auto send_result = iox2::send(std::move(initialized_sample));
  if (!send_result.has_value()) {
    fatal(fmt::format("could not send request {}: error {}", last_sent, static_cast<uint64_t>(send_result.error())));
  }
  if (send_result.value() == 0) {
    fatal(fmt::format("request {} reached no gpuless server", last_sent));
  }

  // A notification with no listener is lost; wait for the listener and notify again.
  auto notify_result = iox2_request_notifier->notify();
  if (notify_result.has_value() && notify_result.value() == 0) {
    spdlog::warn("gpuless: notification for request {} reached no listener, waiting for the server", last_sent);
    if (!wait_for_server()) {
      fatal(fmt::format("no gpuless server listener for request {}", last_sent));
    }
    notify_result = iox2_request_notifier->notify();
  }
  if (!notify_result.has_value()) {
    fatal(fmt::format("could not notify gpuless server of request {}: error {}", last_sent,
                      static_cast<uint64_t>(notify_result.error())));
  }
}
#endif

bool TraceExecutorShmem::init(const char *ip, const short port,
                            manager::instance_profile profile) {

    if (!this->getDeviceAttributes()) {
        return false;
    }

    if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
        // iceoryx1: spawn background thread that waits on waitset and enqueues results
        auto last_synchronized = this->last_synchronized;
        std::thread t{
          [this, last_synchronized]() {

            auto last_synchronized_local = last_synchronized;

            auto process = [&](iox::cxx::expected<const void*, iox::popo::ChunkReceiveResult> val) {

                auto responsePayload = val.value();
                auto header = static_cast<const int*>(iox::mepoo::ChunkHeader::fromUserPayload(responsePayload)->userHeader());
                SPDLOG_DEBUG("Received_reply {}", *header);
                int seq_id = *header;
                if (seq_id == last_synchronized_local + 1)
                {

                    SPDLOG_INFO("Trace execution response received");
                    auto s1 = std::chrono::high_resolution_clock::now();
                    auto fb_protocol_message_response =
                        GetFBProtocolMessage(responsePayload);
                    auto fb_trace_exec_response =
                        fb_protocol_message_response->message_as_FBTraceExecResponse();

                    last_synchronized_local++;
                    bool succ = results.enqueue(
                        std::make_pair(
                          CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response),
                          last_synchronized_local
                        )
                    );
                    assert(succ);
                    auto e1 = std::chrono::high_resolution_clock::now();
                    auto d1 =
                        std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count() /
                        1000000.0;
                    this->serialize_total_time += d1;
                    request_subscriber->release(responsePayload);
                }
                else
                {
                    std::cout << "Got Response with outdated sequence ID! Expected = " << last_synchronized_local
                              << "; Actual = " << seq_id << "! -> skip" << std::endl;
                }
            };

            while(true) {

              auto notificationVector = waitset.value().wait();

              SPDLOG_DEBUG("responses! {}", notificationVector.size());

              for (auto& notification : notificationVector)
              {

                  if(notification->doesOriginateFrom(request_subscriber.get())) {

                    auto val = request_subscriber->take();
                    if(val.has_error() && val.get_error() != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
                      spdlog::error("Failure when polling messages, error {}", val.get_error());
                    }
                    while(!val.has_error()) {
                      process(val);
                      val = request_subscriber->take();
                      if(val.has_error() && val.get_error() != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
                        spdlog::error("Failure when polling messages, error {}", val.get_error());
                      }
                    }

                  } else {
                    spdlog::error("This should not have happened!");
                  }

              }
              SPDLOG_DEBUG("responses done! {}", notificationVector.size());
            }
          }
        };
        t.detach();
    }

    // iceoryx2: no background thread, responses handled directly in main thread
    // via receive_pending_responses()

    return true;
}

bool TraceExecutorShmem::deallocate() {
    return true;
}

void TraceExecutorShmem::receive_pending_responses(bool blocking) {
  if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
    // iceoryx1: background thread handles responses and enqueues them
    // This function just dequeues from the queue
    std::pair<std::shared_ptr<AbstractCudaApiCall>, int> result;
    while (results.try_dequeue(result)) {
      auto& api_call = result.first;
      if (api_call) {
        _last_api_call = api_call;
      }
    }
  }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
  else if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {

    auto& subscriber = iox2_response_subscriber.value();

    auto on_event = [&](iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc> attachment_id) -> iox2::CallbackProgression {
      if (attachment_id.has_event_from(iox2_waitset_guard.value())) {
        SPDLOG_DEBUG("iceoryx2: Response data available");

        // Receive all pending samples
        auto event = iox2_response_listener->try_wait_one();
        while (event.has_value() && event.value().has_value()) {

          auto sample = subscriber.receive();
          if (!sample.has_value() || !sample.value().has_value()) {
            break;
          }

          const auto& received_sample = sample.value();
          auto* responsePayload = received_sample.value().payload().data();

          SPDLOG_DEBUG("receive_pending_responses[iceoryx2]: Received response");

          auto fb_protocol_message_response = GetFBProtocolMessage(responsePayload);
          auto fb_trace_exec_response = fb_protocol_message_response->message_as_FBTraceExecResponse();

          _last_api_call = CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response);

          last_synchronized++;

          auto e1 = std::chrono::high_resolution_clock::now();
          auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(e1 - std::chrono::high_resolution_clock::now()).count() / 1000000.0;
          this->serialize_total_time += d1;

          SPDLOG_DEBUG("receive_pending_responses[iceoryx2]: Processed response {}", last_synchronized);

          event = iox2_response_listener->try_wait_one();
        }
      } else {
        spdlog::error("Unknown event source in wait set!");
      }
      return iox2::CallbackProgression::Continue;
    };

    iox2::bb::Expected<iox2::WaitSetRunResult, iox2::WaitSetRunError> loop_result;
    if(blocking) {
      // Wake up once per second so callers can check that the server is still there.
      loop_result = iox2_waitset->wait_and_process_once_with_timeout(on_event, iox2::bb::Duration::from_secs(1));
    } else {
      loop_result = iox2_waitset->wait_and_process_once_with_timeout(on_event, iox2::bb::Duration::from_micros(1));
    }

    if(!loop_result.has_value()) {
      spdlog::error("iceoryx2: Waitset processing error: {}", static_cast<uint64_t>(loop_result.error()));
    }

  }
#endif
}

bool TraceExecutorShmem::send_only(CudaTrace &cuda_trace)
{
  // FIXME: single implementation with synchronize
    //auto s = std::chrono::high_resolution_clock::now();

  //spdlog::info("Send only, call stack of size {}", cuda_trace.sizeCallStackNotSent());

    this->synchronize_counter_++;
    //SPDLOG_INFO(
    //    "TraceExecutorTcp::synchronize() [synchronize_counter={}, size={}]",
    //    this->synchronize_counter_, cuda_trace.callStack().size());

    // collect statistics on synchronizations

    // send trace execution request
    auto sx = std::chrono::high_resolution_clock::now();
    flatbuffers::FlatBufferBuilder builder;
    CudaTraceConverter::traceToExecRequest(cuda_trace, builder);
    auto ex = std::chrono::high_resolution_clock::now();
    auto dx =
        std::chrono::duration_cast<std::chrono::microseconds>(ex - sx).count() /
        1000000.0;
    this->serialize_total_time += dx;
    //std::cerr << "Request compress " << dx << std::endl;

    //int64_t expectedResponseSequenceId = requestSequenceId;
    auto s1 = std::chrono::high_resolution_clock::now();

    if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
      request_publisher->loan(builder.GetSize(), 16, sizeof(int), alignof(int))
          .and_then([&, this](auto& requestPayload) {

            auto header = static_cast<int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());

            (*header) = ++last_sent;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            SPDLOG_DEBUG("Submit_request {}, size {}", last_sent - 1, builder.GetSize());

            request_publisher->publish(requestPayload);
          })
          .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    // iceoryx2 path
    else if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {

      send_request_iox2(builder);
    }
#endif

    SPDLOG_INFO("Trace execution request sent {}", this->serialize_total_time);

    cuda_trace.markSent();
    //// FIXME: merge implementation with synchronization
    //// FIXME: verify this works properly in all edge cases

    //std::shared_ptr<AbstractCudaApiCall> cuda_api_call = nullptr;

    //auto process = [&](iox::cxx::expected<const void*, iox::popo::ChunkReceiveResult> val) {

    //    auto responsePayload = val.value();
    //    auto responseHeader = iox::popo::ResponseHeader::fromPayload(responsePayload);
    //    SPDLOG_DEBUG("Received_reply {}", responseHeader->getSequenceId());
    //    //if (responseHeader->getSequenceId() == expectedResponseSequenceId)
    //    if (responseHeader->getSequenceId() == last_synchronized + 1)
    //    {

    //        SPDLOG_INFO("Trace execution response received");
    //        auto s1 = std::chrono::high_resolution_clock::now();
    //        auto fb_protocol_message_response =
    //            GetFBProtocolMessage(responsePayload);
    //        auto fb_trace_exec_response =
    //            fb_protocol_message_response->message_as_FBTraceExecResponse();
    //        cuda_api_call =
    //            CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response);
    //        auto e1 = std::chrono::high_resolution_clock::now();
    //        auto d1 =
    //            std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count() /
    //            1000000.0;
    //        //std::cerr << d1 << std::endl;
    //        this->serialize_total_time += d1;
    //        client->releaseResponse(responsePayload);

    //        last_synchronized++;
    //    }
    //    else
    //    {
    //        std::cout << "Got Response with outdated sequence ID! Expected = " << last_synchronized
    //                  << "; Actual = " << responseHeader->getSequenceId() << "! -> skip" << std::endl;
    //    }
    //};

    int64_t prev_synchronization = last_synchronized;

    static double duration = 0;

    std::shared_ptr<AbstractCudaApiCall> last_api_call_result;

    if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
      // iceoryx1: drain queue non-blocking (background thread enqueues results)
      std::pair<std::shared_ptr<AbstractCudaApiCall>, int> cuda_api_call{nullptr, 0};
      bool status = results.try_dequeue(cuda_api_call);
      while(status) {
        last_synchronized = cuda_api_call.second;
        SPDLOG_INFO("Opportunistic sync, now position synchronized {}", last_synchronized);
        status = results.try_dequeue(cuda_api_call);
      }
      last_api_call_result = cuda_api_call.first;

      int64_t synchronized_calls = last_synchronized - prev_synchronization;
      if(synchronized_calls > 0) {

        // Here we want to access the part of call stack that was already sent.
        auto [begin, end] = cuda_trace.fullCallStack();
        for(int i = 0; i < synchronized_calls; ++i) {

          if(auto* ptr = dynamic_cast<CudaMemcpyH2D*>((*begin).get())) {
            _pool.give(ptr->shared_name);
          }
          if(auto* ptr = dynamic_cast<CudaMemcpyAsyncH2D*>((*begin).get())) {
            _pool.give(ptr->shared_name);
          }

          ++begin;

        }

      cuda_trace.markSynchronized(synchronized_calls);
      if(last_api_call_result) {
        cuda_trace.setHistoryTop(last_api_call_result);
      }
    }
}

    //auto f = std::chrono::high_resolution_clock::now();
    //auto d3 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(f - s).count() /
    //    1000.0;
    //std::cerr << "send is done. waitset? " << d3 << std::endl;

    return true;
}

bool TraceExecutorShmem::synchronize(CudaTrace &cuda_trace)
{
    auto s = std::chrono::high_resolution_clock::now();

    this->synchronize_counter_++;
    //SPDLOG_INFO(
    //    "TraceExecutorTcp::synchronize() [synchronize_counter={}, size={}]",
    //    this->synchronize_counter_, cuda_trace.callStack().size());

    // collect statistics on synchronizations

    // send trace execution request
    auto sx = std::chrono::high_resolution_clock::now();
    flatbuffers::FlatBufferBuilder builder;
    CudaTraceConverter::traceToExecRequest(cuda_trace, builder);
    auto ex = std::chrono::high_resolution_clock::now();
    auto dx =
        std::chrono::duration_cast<std::chrono::microseconds>(ex - sx).count() /
        1000000.0;
    this->serialize_total_time += dx;
    //std::cerr << "Request compress " << dx << std::endl;

    //int64_t expectedResponseSequenceId = requestSequenceId;
    auto s1 = std::chrono::high_resolution_clock::now();
    // iceoryx1 path
    if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
      // FIXME: what should be the alignment here?
      request_publisher->loan(builder.GetSize(), 16, sizeof(int), alignof(int))
          .and_then([&, this](auto& requestPayload) {

            auto header = static_cast<int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());

            (*header) = ++last_sent;
            //auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            SPDLOG_DEBUG("Submit_request {}, size {}", last_sent - 1, builder.GetSize());

            request_publisher->publish(requestPayload);
          })
          .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    // iceoryx2 path
    else if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {

      send_request_iox2(builder);
    }
#endif

    SPDLOG_INFO("Trace execution request sent. Last synchronized {} last sent {}", last_synchronized, last_sent);

    //! [take response]
    std::shared_ptr<AbstractCudaApiCall> last_api_call_result;

    if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
      // iceoryx1: blocking dequeue from background thread's queue
      std::pair<std::shared_ptr<AbstractCudaApiCall>, int> cuda_api_call;
      while(last_synchronized != last_sent) {
        results.wait_dequeue(cuda_api_call);
        last_synchronized = cuda_api_call.second;
        SPDLOG_INFO("Synchronized, now position synchronized {}, last sent {}, size {}", last_synchronized, last_sent, results.size_approx());
        //spdlog::info("Synchronized, now position synchronized {}, last sent {}", last_synchronized, last_sent);
      }
      last_api_call_result = cuda_api_call.first;
    } else {
      // iceoryx2: direct waitset/poll in main thread
      while(last_synchronized != last_sent) {
        receive_pending_responses(true);
        if (last_synchronized != last_sent && !server_connected()) {
          fatal(fmt::format("the gpuless server disappeared; requests {}..{} are unanswered",
                            last_synchronized + 1, last_sent));
        }
        SPDLOG_INFO("Synchronized, now position synchronized {}, last sent {}", last_synchronized, last_sent);
        //spdlog::info("Synchronized, now position synchronized {}, last sent {}", last_synchronized, last_sent);

        //if(_polling_mode == mignificient::ipc::PollingMode::POLL && last_synchronized != last_sent) {
        //    std::this_thread::sleep_for(std::chrono::microseconds(_poll_interval_us));
        //}
      }
      last_api_call_result = _last_api_call;
    }

    // return chunks
    auto [begin, end] = cuda_trace.fullCallStack();

    for(; begin != end; ++begin) {

      if(auto* ptr = dynamic_cast<CudaMemcpyH2D*>((*begin).get())) {
        _pool.give(ptr->shared_name);
      }

      if(auto* ptr = dynamic_cast<CudaMemcpyAsyncH2D*>((*begin).get())) {
        _pool.give(ptr->shared_name);
      }

    }

    cuda_trace.markSynchronized();
    cuda_trace.setHistoryTop(last_api_call_result);

    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000000.0;
    this->synchronize_total_time_ += d;

    SPDLOG_INFO(
        "TraceExecutorTcp::synchronize() successful [t={}s, total_time={}s]", d,
        this->synchronize_total_time_);
    return true;
}

bool TraceExecutorShmem::getDeviceAttributes() {

    SPDLOG_INFO("TraceExecutorShmem::getDeviceAttributes()");

    flatbuffers::FlatBufferBuilder builder;
    auto attr_request =
        CreateFBProtocolMessage(builder, FBMessage_FBTraceAttributeRequest,
                                CreateFBTraceAttributeRequest(builder).Union());
    builder.Finish(attr_request);

    // First request of this client: make sure the server can receive it.
    if (!wait_for_server()) {
      return false;
    }

    // iceoryx1 path
    if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V1) {
      SPDLOG_DEBUG("FBTraceAttributeRequest sent {}", fmt::ptr(request_publisher.get()));

      // FIXME: Merge with other send functions
      request_publisher->loan(builder.GetSize(), 16, sizeof(int), alignof(int))
          .and_then([&, this](auto& requestPayload) {

            auto header = static_cast<int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());
            (*header) = ++last_sent;

            //auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            SPDLOG_INFO("Submit_request {}", last_sent - 1);

            request_publisher->publish(requestPayload);

          })
          .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });

      SPDLOG_INFO("FBTraceAttributeResponse wait for receive");

      //std::pair<std::shared_ptr<AbstractCudaApiCall>, int> cuda_api_call;
      //results.wait_dequeue(cuda_api_call);

      //last_synchronized = cuda_api_call.second;
      auto notificationVector = waitset.value().wait();
      for (auto& notification : notificationVector)
      {
        if(notification->doesOriginateFrom(request_subscriber.get())) {

          auto val = request_subscriber->take();
          if(!val) {
            spdlog::error("Failed to receive on device attributes!");
            return false;
          }

          auto responsePayload = val.value();
          auto responseHeader = iox::popo::ResponseHeader::fromPayload(responsePayload);
          SPDLOG_INFO("Received_reply {}", responseHeader->getSequenceId());

          auto fb_protocol_message_response =
              GetFBProtocolMessage(responsePayload);
          auto fb_trace_attribute_response =
              fb_protocol_message_response->message_as_FBTraceAttributeResponse();

          this->device_total_mem = fb_trace_attribute_response->total_mem();
          this->device_attributes.resize(CU_DEVICE_ATTRIBUTE_MAX);
          for (const auto &a : *fb_trace_attribute_response->device_attributes()) {
              int32_t value = a->value();
              auto dev_attr = static_cast<CUdevice_attribute>(a->device_attribute());
              // The server may be built with a newer CUDA (more attributes) than this client.
              if (static_cast<size_t>(dev_attr) < this->device_attributes.size()) {
                  this->device_attributes[dev_attr] = value;
              }
          }

          //client->releaseResponse(responsePayload);
          request_subscriber->release(responsePayload);

          last_synchronized++;

          return true;

        }
      }
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    // iceoryx2 path
    else if (_ipc_backend == mignificient::ipc::IPCBackend::ICEORYX_V2) {
      SPDLOG_DEBUG("FBTraceAttributeRequest sent (iceoryx2)");

      send_request_iox2(builder);

      SPDLOG_INFO("FBTraceAttributeResponse wait for receive (iceoryx2)");

      auto& subscriber = iox2_response_subscriber.value();
      auto& waitset = iox2_waitset.value();

      auto on_event = [&](iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc> attachment_id) -> iox2::CallbackProgression {
        if (attachment_id.has_event_from(iox2_waitset_guard.value())) {
          SPDLOG_DEBUG("iceoryx2: Response data available");

          // Receive all pending samples
          auto event = iox2_response_listener->try_wait_one();
          while (event.has_value() && event.value().has_value()) {

            auto sample = subscriber.receive();
            if (!sample.has_value() || !sample.value().has_value()) {
              break;
            }

            const auto& received_sample = sample.value();
            auto* responsePayload = received_sample.value().payload().data();

            SPDLOG_DEBUG("iceoryx2: Received response");

            auto fb_protocol_message_response =
                GetFBProtocolMessage(responsePayload);
            auto fb_trace_attribute_response =
                fb_protocol_message_response->message_as_FBTraceAttributeResponse();

            last_synchronized++;
            if (!fb_trace_attribute_response || !fb_trace_attribute_response->device_attributes()) {
              spdlog::error("gpuless: expected a device attribute response, got something else");
              event = iox2_response_listener->try_wait_one();
              continue;
            }

            this->device_total_mem = fb_trace_attribute_response->total_mem();
            this->device_attributes.resize(CU_DEVICE_ATTRIBUTE_MAX);
            for (const auto &a : *fb_trace_attribute_response->device_attributes()) {
                int32_t value = a->value();
                auto dev_attr = static_cast<CUdevice_attribute>(a->device_attribute());
                // The server may be built with a newer CUDA (more attributes) than this client.
                if (static_cast<size_t>(dev_attr) < this->device_attributes.size()) {
                    this->device_attributes[dev_attr] = value;
                }
            }

            auto e1 = std::chrono::high_resolution_clock::now();
            auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(e1 - std::chrono::high_resolution_clock::now()).count() / 1000000.0;
            this->serialize_total_time += d1;

            SPDLOG_DEBUG("iceoryx2: Processed response {}", last_synchronized);

            event = iox2_response_listener->try_wait_one();
          }
        } else {
          spdlog::error("Unknown event source in wait set!");
        }
        return iox2::CallbackProgression::Continue;
      };

      // Only a received attribute response counts as success. No fixed
      // deadline here: the server holds requests until the orchestrator
      // enables execution (first invocation), which can take arbitrarily long.
      // Fail only when the server is gone; check that once per second.
      // Note: a server that exits cleanly removes its ports and is detected;
      // a crashed (SIGKILL/SIGSEGV) server leaves stale ports in the iceoryx2
      // dynamic config, so that case still waits (the orchestrator's function
      // timeout covers it).
      auto start = std::chrono::steady_clock::now();
      auto next_warning = SERVER_WAIT_LIMIT;
      while (last_synchronized < last_sent) {
        auto loop_result = waitset.wait_and_process_once_with_timeout(
            on_event, iox2::bb::Duration::from_secs(1));
        if (!loop_result.has_value()) {
          spdlog::error("iceoryx2: WaitSet loop error: {}", static_cast<uint64_t>(loop_result.error()));
        }
        if (last_synchronized >= last_sent) {
          break;
        }
        if (!server_connected()) {
          spdlog::error("gpuless: the gpuless server disappeared before answering the device attribute request");
          return false;
        }
        if (std::chrono::steady_clock::now() - start >= next_warning) {
          spdlog::warn("gpuless: still waiting for the device attribute response after {} s",
                       next_warning.count());
          next_warning += SERVER_WAIT_LIMIT;
        }
      }

      if (this->device_attributes.empty()) {
        spdlog::error("gpuless: the device attribute response carried no attributes");
        return false;
      }
      return true;
    }
#endif

    return false;
}

double TraceExecutorShmem::getSynchronizeTotalTime() const {
    return synchronize_total_time_;
}

} // namespace gpuless
