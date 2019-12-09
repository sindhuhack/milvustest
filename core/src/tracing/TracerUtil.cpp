#include "tracing/TracerUtil.h"

#include <opentracing/dynamic_load.h>
#include <opentracing/tracer.h>

#include <fstream>
#include <iostream>

#include "thirdparty/nlohmann/json.hpp"

namespace milvus {
namespace tracing {

std::string TracerUtil::tracer_context_header_name_;

void
TracerUtil::InitGlobal(const std::string& config_path) {
    if (!config_path.empty())
        LoadConfig(config_path);
}

void
TracerUtil::LoadConfig(const std::string& config_path) {
    // Parse JSON config
    std::ifstream tracer_config(config_path);
    if (!tracer_config.good()) {
        std::cerr << "Failed to open tracer config file " << config_path << ": " << std::strerror(errno) << std::endl;
        return;
    }
    using json = nlohmann::json;
    json tracer_config_json;
    tracer_config >> tracer_config_json;
    std::string tracing_shared_lib = tracer_config_json[TRACER_LIBRARY_CONFIG_NAME];
    std::string tracer_config_str = tracer_config_json[TRACER_CONFIGURATION_CONFIG_NAME].dump();
    tracer_context_header_name_ = tracer_config_json[TRACE_CONTEXT_HEADER_CONFIG_NAME].dump();

    // Load the tracer library.
    std::string error_message;
    auto handle_maybe = opentracing::DynamicallyLoadTracingLibrary(tracing_shared_lib.c_str(), error_message);
    if (!handle_maybe) {
        std::cerr << "Failed to load tracer library: " << error_message << std::endl;
        return;
    }

    // Construct a tracer.
    auto& tracer_factory = handle_maybe->tracer_factory();
    auto tracer_maybe = tracer_factory.MakeTracer(tracer_config_str.c_str(), error_message);
    if (!tracer_maybe) {
        std::cerr << "Failed to create tracer: " << error_message << std::endl;
        return;
    }
    auto& tracer = *tracer_maybe;

    opentracing::Tracer::InitGlobal(tracer);
}

const std::string&
TracerUtil::GetTraceContextHeaderName() {
    return tracer_context_header_name_;
}

}  // namespace tracing
}  // namespace milvus