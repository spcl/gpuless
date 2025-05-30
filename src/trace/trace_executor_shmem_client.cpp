#include "trace_executor_shmem_client.hpp"
#include "../schemas/allocation_protocol_generated.h"
#include "cuda_trace_converter.hpp"

#include <spdlog/spdlog.h>

#include <iceoryx_posh/runtime/posh_runtime.hpp>
#include <iceoryx_hoofs/cxx/expected.hpp>
#include <stdexcept>

namespace gpuless {

iox::runtime::PoshRuntime* TraceExecutorShmem::runtime_factory_impl(iox::cxx::optional<const iox::RuntimeName_t*> var, TraceExecutorShmem* ptr)
{
    static TraceExecutorShmem* obj_ptr = nullptr;
    if(ptr) {
        obj_ptr = ptr;
        return nullptr;
    } else if (var.has_value()) {
        obj_ptr->_impl = std::make_unique<iox::runtime::PoshRuntimeImpl>(var);
        return obj_ptr->_impl.get();
    } else {
        return obj_ptr->_impl.get();
    }
}

iox::runtime::PoshRuntime& runtime_factory(iox::cxx::optional<const iox::RuntimeName_t*> var)
{
    return *TraceExecutorShmem::runtime_factory_impl(var, nullptr);
}

TraceExecutorShmem::TraceExecutorShmem()
{
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%H:%M:%S:%e:%f] %v");

    // This is useful when we do not have executor, i.e., we just do LD_PRELOAD on existing app.
    const char* app_name = std::getenv("SHMEM_APP_NAME");
    const char* user_name = std::getenv("CONTAINER_NAME");

    _pool.set_user_name(user_name);

    if(app_name) {

      iox::runtime::PoshRuntime::setRuntimeFactory(runtime_factory);
      runtime_factory_impl(nullptr, this);

      iox::runtime::PoshRuntime::initRuntime(
      iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, app_name}
      );
    }

    request_publisher.reset(new iox::popo::UntypedPublisher({
        iox::capro::ServiceDescription{
            iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, user_name},
            "Gpuless",
            "Request"
        }
    }));

    request_subscriber.reset(new iox::popo::UntypedSubscriber({
        iox::capro::ServiceDescription{
            iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, user_name},
            "Gpuless",
            "Response"
        }
    }));


    waitset.emplace();
    wait_poll = std::string_view{std::getenv("POLL_TYPE")} == "wait";

    if(wait_poll) {
      waitset.value().attachState(*request_subscriber, iox::popo::SubscriberState::HAS_DATA).or_else([](auto) {
          std::cerr << "failed to attach server" << std::endl;
          std::exit(EXIT_FAILURE);
      });

    }
}

//TraceExecutorShmem::~TraceExecutorShmem() = default;
TraceExecutorShmem::~TraceExecutorShmem()
{
  std::cerr << "Total serialize_total_time " << serialize_total_time << std::endl;
}

bool TraceExecutorShmem::init(const char *ip, const short port,
                            manager::instance_profile profile) {
    this->getDeviceAttributes();

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
                //std::cerr << d1 << std::endl;
                this->serialize_total_time += d1;
                request_subscriber->release(responsePayload);
            }
            else
            {
                std::cout << "Got Response with outdated sequence ID! Expected = " << last_synchronized_local
                          << "; Actual = " << seq_id << "! -> skip" << std::endl;
            }
        };

        //while(true) {

        //  auto val = request_subscriber->take();

        //  if(val.has_error()) {

        //    if(val.get_error() == iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
        //      continue;
        //    } else {
        //      abort();
        //    }

        //  } else {

        //    SPDLOG_ERROR("poll responses!");
        //    process(val);

        //  }

        //}
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


    return true;
}

bool TraceExecutorShmem::deallocate() {
    return true;
}

bool TraceExecutorShmem::send_only(CudaTrace &cuda_trace)
{
  // FIXME: single implementation with synchronize
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
    // FIXME: what should be the alignment here?
    request_publisher->loan(builder.GetSize(), 16, sizeof(int), alignof(int))
        .and_then([&, this](auto& requestPayload) {


						auto header = static_cast<int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());

						(*header) = ++last_sent;
            //auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            //requestHeader->setSequenceId(++last_sent);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            SPDLOG_DEBUG("Submit_request {}, size {}", last_sent - 1, builder.GetSize());

            request_publisher->publish(requestPayload);
        })
        .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
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

    //std::cerr << "Add sleep " << std::endl;
    //for(int i = 0; i < 100; ++i)  {
    //std::this_thread::sleep_for(std::chrono::microseconds(1));

    // We don't wait, we check for new results.
    //auto val = client->take();
    //while(!val.has_error()) {
    //  SPDLOG_INFO("RECEIVED");
    //  SPDLOG_INFO("Send-only: process notification");
    //  process(val);
    //  val = client->take();
    //}
    ////}
    std::pair<std::shared_ptr<AbstractCudaApiCall>, int> cuda_api_call{nullptr, 0};
    bool status = results.try_dequeue(cuda_api_call);
    while(status) {

      last_synchronized = cuda_api_call.second;
      SPDLOG_INFO("Opportunistic sync, now position synchronized {}", last_synchronized);
      status = results.try_dequeue(cuda_api_call);
    }
    //results.wait_dequeue(cuda_api_call);
    //last_synchronized = cuda_api_call.second;
    //SPDLOG_INFO("Opportunistic sync, now position synchronized {}", last_synchronized);

    std::cerr << "Previous synchronization point " << prev_synchronization << " new one " << last_synchronized << std::endl;
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
      if(cuda_api_call.first) {
        cuda_trace.setHistoryTop(cuda_api_call.first);
      }
    }

    std::cerr << "send is done. waitset? " << std::endl;

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
    // FIXME: what should be the alignment here?
    request_publisher->loan(builder.GetSize(), 16, sizeof(int), alignof(1))
        .and_then([&, this](auto& requestPayload) {

						auto header = static_cast<int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());

						(*header) = ++last_sent;
            //auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            //requestHeader->setSequenceId(++last_sent);
            SPDLOG_DEBUG("Submit_request {}", last_sent - 1);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            request_publisher->publish(requestPayload);

        })
        .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
    SPDLOG_INFO("Trace execution request sent");

    //! [take response]
    //std::shared_ptr<AbstractCudaApiCall> cuda_api_call = nullptr;
    std::pair<std::shared_ptr<AbstractCudaApiCall>, int> cuda_api_call;

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
    //        this->serialize_total_time += d1;
    //        //std::cerr << d1 << std::endl;
    //        //this->synchronize_total_time_2 += d1;
    //        client->releaseResponse(responsePayload);

    //        last_synchronized++;
    //    }
    //    else
    //    {
    //        std::cout << "Got Response with outdated sequence ID! Expected = " << last_synchronized
    //                  << "; Actual = " << responseHeader->getSequenceId() << "! -> skip" << std::endl;
    //    }
    //};

    if(wait_poll) {

      while(last_synchronized != last_sent) {

        results.wait_dequeue(cuda_api_call);

        last_synchronized = cuda_api_call.second;
        SPDLOG_INFO("Synchronized, now position synchronized {}, last sent {}, size {}", last_synchronized, last_sent, results.size_approx());

        //std::cerr << "Synchronize " << last_synchronized << " " << last_sent << std::endl;

        //// FIXME here wait until we reach the final synchronization point
        //auto notificationVector = waitset.value().wait();

        //std::cerr << "responses! " << notificationVector.size() << std::endl;

        //for (auto& notification : notificationVector)
        //{

        //    if(notification->doesOriginateFrom(client.get())) {

        //      auto val = client->take();
        //      if(val.has_error() && val.get_error() != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
        //        spdlog::error("Failure when polling messages, error {}", val.get_error());
        //      }
        //      while(!val.has_error()) {
        //        process(val);
        //        val = client->take();
        //        if(val.has_error() && val.get_error() != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
        //          spdlog::error("Failure when polling messages, error {}", val.get_error());
        //        }
        //      }

        //    } else {
        //      spdlog::error("This should not have happened!");
        //    }

        //}
      }

    } else {

      // FIXME: this doesn't support the new algorithm that waits until we synchronize

      abort();
      std::cerr << "WRONG" << std::endl;
      while(true) {

        //auto val = client->take();

        //if(val.has_error()) {

        //  if(val.get_error() == iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
        //    continue;
        //  } else {
        //    abort();
        //  }

        //} else {

        //  //process(val);
        //  break;

        //}

      }

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
    //cuda_trace.setHistoryTop(cuda_api_call);
    cuda_trace.setHistoryTop(cuda_api_call.first);

    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000000.0;
    this->synchronize_total_time_ += d;

    SPDLOG_INFO(
        "TraceExecutorTcp::synchronize() successful [t={}s, total_time={}s]", d,
        this->synchronize_total_time_);
    std::cerr << "Synchronize succesful " << serialize_total_time << std::endl;
    return true;
}

bool TraceExecutorShmem::getDeviceAttributes() {
    SPDLOG_INFO("TraceExecutorTcp::getDeviceAttributes()");


    flatbuffers::FlatBufferBuilder builder;
    auto attr_request =
        CreateFBProtocolMessage(builder, FBMessage_FBTraceAttributeRequest,
                                CreateFBTraceAttributeRequest(builder).Union());
    builder.Finish(attr_request);
    SPDLOG_DEBUG("FBTraceAttributeRequest sent {}", fmt::ptr(request_publisher.get()));

    // FIXME: Merge with other send functions
    request_publisher->loan(builder.GetSize(), 16, sizeof(int), alignof(int))
        .and_then([&, this](auto& requestPayload) {

						auto header = static_cast<int*>(iox::mepoo::ChunkHeader::fromUserPayload(requestPayload)->userHeader());

						(*header) = ++last_sent;

            //auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            //requestHeader->setSequenceId(++last_sent);
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
          spdlog::error("Failed to receive the response on device attributes!");
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
            this->device_attributes[dev_attr] = value;
        }

        //client->releaseResponse(responsePayload);
				request_subscriber->release(responsePayload);

        last_synchronized++;

        return true;

      }

    }

    return false;
}

double TraceExecutorShmem::getSynchronizeTotalTime() const {
    return synchronize_total_time_;
}

} // namespace gpuless
