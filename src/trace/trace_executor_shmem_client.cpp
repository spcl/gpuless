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

    client.reset(new iox::popo::UntypedClient({
      iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, user_name},
      "Gpuless",
      "Client"
    }));

    waitset.emplace();
    wait_poll = std::string_view{std::getenv("POLL_TYPE")} == "wait";

    if(wait_poll) {
      waitset.value().attachState(*client, iox::popo::ClientState::HAS_RESPONSE).or_else([](auto) {
          std::cerr << "failed to attach server" << std::endl;
          std::exit(EXIT_FAILURE);
      });
    }
}

TraceExecutorShmem::~TraceExecutorShmem() = default;

bool TraceExecutorShmem::init(const char *ip, const short port,
                            manager::instance_profile profile) {
    this->getDeviceAttributes();
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
    //auto sx = std::chrono::high_resolution_clock::now();
    flatbuffers::FlatBufferBuilder builder;
    CudaTraceConverter::traceToExecRequest(cuda_trace, builder);
    //auto ex = std::chrono::high_resolution_clock::now();
    //auto dx =
    //    std::chrono::duration_cast<std::chrono::microseconds>(ex - sx).count() /
    //    1000000.0;
    //std::cerr << "Request compress " << dx << std::endl;

    //int64_t expectedResponseSequenceId = requestSequenceId;
    auto s1 = std::chrono::high_resolution_clock::now();
    // FIXME: what should be the alignment here?
    client->loan(builder.GetSize(), 16)
        .and_then([&, this](auto& requestPayload) {


            auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            requestHeader->setSequenceId(++last_sent);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            SPDLOG_DEBUG("Submit_request {}", last_sent - 1);

            client->send(requestPayload).or_else(
                [&](auto& error) { std::cout << "Could not send Request! Error: " << error << std::endl; });

        })
        .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
    SPDLOG_INFO("Trace execution request sent");

    cuda_trace.markSent();
    // FIXME: merge implementation with synchronization
    // FIXME: verify this works properly in all edge cases

    std::shared_ptr<AbstractCudaApiCall> cuda_api_call = nullptr;

    auto process = [&](iox::cxx::expected<const void*, iox::popo::ChunkReceiveResult> val) {

        auto responsePayload = val.value();
        auto responseHeader = iox::popo::ResponseHeader::fromPayload(responsePayload);
        SPDLOG_DEBUG("Received_reply {}", responseHeader->getSequenceId());
        //if (responseHeader->getSequenceId() == expectedResponseSequenceId)
        if (responseHeader->getSequenceId() == last_synchronized + 1)
        {

            SPDLOG_INFO("Trace execution response received");
            auto fb_protocol_message_response =
                GetFBProtocolMessage(responsePayload);
            auto fb_trace_exec_response =
                fb_protocol_message_response->message_as_FBTraceExecResponse();
            cuda_api_call =
                CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response);
            //auto e1 = std::chrono::high_resolution_clock::now();
            //auto d1 =
            //    std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count() /
            //    1000000.0;
            //std::cerr << d1 << std::endl;
            //this->synchronize_total_time_2 += d1;
            client->releaseResponse(responsePayload);

            last_synchronized++;
        }
        else
        {
            std::cout << "Got Response with outdated sequence ID! Expected = " << last_synchronized
                      << "; Actual = " << responseHeader->getSequenceId() << "! -> skip" << std::endl;
        }
    };

    int64_t prev_synchronization = last_synchronized;

    // We don't wait, we check for new results.
    auto val = client->take();
    while(!val.has_error()) {
      SPDLOG_INFO("Send-only: process notification");
      process(val);
      val = client->take();
    }

    //std::cerr << "Previous synchronization point " << prev_synchronization << " new one " << last_synchronized << std::endl;
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
      if(cuda_api_call)
        cuda_trace.setHistoryTop(cuda_api_call);

    }

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
    //auto sx = std::chrono::high_resolution_clock::now();
    flatbuffers::FlatBufferBuilder builder;
    CudaTraceConverter::traceToExecRequest(cuda_trace, builder);
    //auto ex = std::chrono::high_resolution_clock::now();
    //auto dx =
    //    std::chrono::duration_cast<std::chrono::microseconds>(ex - sx).count() /
    //    1000000.0;
    //std::cerr << "Request compress " << dx << std::endl;

    //int64_t expectedResponseSequenceId = requestSequenceId;
    auto s1 = std::chrono::high_resolution_clock::now();
    // FIXME: what should be the alignment here?
    client->loan(builder.GetSize(), 16)
        .and_then([&, this](auto& requestPayload) {

            auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            requestHeader->setSequenceId(++last_sent);
            SPDLOG_DEBUG("Submit_request {}", last_sent - 1);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            client->send(requestPayload).or_else(
                [&](auto& error) { std::cout << "Could not send Request! Error: " << error << std::endl; });


        })
        .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
    SPDLOG_INFO("Trace execution request sent");

    //! [take response]
    std::shared_ptr<AbstractCudaApiCall> cuda_api_call = nullptr;

    auto process = [&](iox::cxx::expected<const void*, iox::popo::ChunkReceiveResult> val) {

        auto responsePayload = val.value();
        auto responseHeader = iox::popo::ResponseHeader::fromPayload(responsePayload);
        SPDLOG_DEBUG("Received_reply {}", responseHeader->getSequenceId());
        //if (responseHeader->getSequenceId() == expectedResponseSequenceId)
        if (responseHeader->getSequenceId() == last_synchronized + 1)
        {

            SPDLOG_INFO("Trace execution response received");
            auto fb_protocol_message_response =
                GetFBProtocolMessage(responsePayload);
            auto fb_trace_exec_response =
                fb_protocol_message_response->message_as_FBTraceExecResponse();
            cuda_api_call =
                CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response);
            //auto e1 = std::chrono::high_resolution_clock::now();
            //auto d1 =
            //    std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count() /
            //    1000000.0;
            //std::cerr << d1 << std::endl;
            //this->synchronize_total_time_2 += d1;
            client->releaseResponse(responsePayload);

            last_synchronized++;
        }
        else
        {
            std::cout << "Got Response with outdated sequence ID! Expected = " << last_synchronized
                      << "; Actual = " << responseHeader->getSequenceId() << "! -> skip" << std::endl;
        }
    };

    if(wait_poll) {

      while(last_synchronized != last_sent) {

        //std::cerr << "Synchronize " << last_synchronized << " " << last_sent << std::endl;

        // FIXME here wait until we reach the final synchronization point
        auto notificationVector = waitset.value().wait();

        //std::cerr << "responses! " << notificationVector.size() << std::endl;

        for (auto& notification : notificationVector)
        {

            if(notification->doesOriginateFrom(client.get())) {

              auto val = client->take();
              if(val.has_error() && val.get_error() != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
                spdlog::error("Failure when polling messages, error {}", val.get_error());
              }
              while(!val.has_error()) {
                process(val);
                val = client->take();
                if(val.has_error() && val.get_error() != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
                  spdlog::error("Failure when polling messages, error {}", val.get_error());
                }
              }

            } else {
              spdlog::error("This should not have happened!");
            }

        }
      }

    } else {

      // FIXME: this doesn't support the new algorithm that waits until we synchronize

      while(true) {

        auto val = client->take();

        if(val.has_error()) {

          if(val.get_error() == iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
            continue;
          } else {
            abort();
          }

        } else {

          process(val);
          break;

        }

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
    cuda_trace.setHistoryTop(cuda_api_call);

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
    SPDLOG_INFO("TraceExecutorTcp::getDeviceAttributes()");


    flatbuffers::FlatBufferBuilder builder;
    auto attr_request =
        CreateFBProtocolMessage(builder, FBMessage_FBTraceAttributeRequest,
                                CreateFBTraceAttributeRequest(builder).Union());
    builder.Finish(attr_request);
    SPDLOG_DEBUG("FBTraceAttributeRequest sent");

    // FIXME: Merge with other send functions
    client->loan(builder.GetSize(), 16)
        .and_then([&, this](auto& requestPayload) {

            auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            //requestHeader->setSequenceId(requestSequenceId);
            requestHeader->setSequenceId(++last_sent);
            //expectedResponseSequenceId = requestSequenceId;
            //requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            SPDLOG_INFO("Submit_request {}", last_sent - 1);

            client->send(requestPayload).or_else(
                [&](auto& error) { std::cout << "Could not send Request! Error: " << error << std::endl; });

        })
        .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });

    SPDLOG_INFO("FBTraceAttributeResponse wait for receive");

    auto notificationVector = waitset.value().wait();
    for (auto& notification : notificationVector)
    {
      if(notification->doesOriginateFrom(client.get())) {

        auto val = client->take();
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

        client->releaseResponse(responsePayload);

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
