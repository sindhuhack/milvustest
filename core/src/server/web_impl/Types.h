// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <map>
#include <string>

#include <oatpp/core/data/mapping/type/Object.hpp>

#include "db/engine/ExecutionEngine.h"

#include "server/web_impl/Constants.h"

namespace milvus {
namespace server {
namespace web {

using OString = oatpp::data::mapping::type::String;
using OInt64 = oatpp::data::mapping::type::Int64;
using OFloat32 = oatpp::data::mapping::type::Float32;
template <class T>
using OList = oatpp::data::mapping::type::List<T>;

using OQueryParams = oatpp::web::protocol::http::QueryParams;

enum StatusCode : int {
    SUCCESS = 0,
    UNEXPECTED_ERROR = 1,
    CONNECT_FAILED = 2,  // reserved.
    PERMISSION_DENIED = 3,
    TABLE_NOT_EXISTS = 4,  // DB_NOT_FOUND || TABLE_NOT_EXISTS
    ILLEGAL_ARGUMENT = 5,
    ILLEGAL_RANGE = 6,
    ILLEGAL_DIMENSION = 7,
    ILLEGAL_INDEX_TYPE = 8,
    ILLEGAL_TABLE_NAME = 9,
    ILLEGAL_TOPK = 10,
    ILLEGAL_ROWRECORD = 11,
    ILLEGAL_VECTOR_ID = 12,
    ILLEGAL_SEARCH_RESULT = 13,
    FILE_NOT_FOUND = 14,
    META_FAILED = 15,
    CACHE_FAILED = 16,
    CANNOT_CREATE_FOLDER = 17,
    CANNOT_CREATE_FILE = 18,
    CANNOT_DELETE_FOLDER = 19,
    CANNOT_DELETE_FILE = 20,
    BUILD_INDEX_ERROR = 21,
    ILLEGAL_NLIST = 22,
    ILLEGAL_METRIC_TYPE = 23,
    OUT_OF_MEMORY = 24,

    //
    PATH_PARAM_LOSS = 31,
    QUERY_PARAM_LOSS = 32,
    BODY_FIELD_LOSS = 33,
};

static const std::unordered_map<engine::EngineType, std::string> IndexMap = {
    {engine::EngineType::FAISS_IDMAP, NAME_ENGINE_TYPE_FLAT},
    {engine::EngineType::FAISS_IVFFLAT, NAME_ENGINE_TYPE_IVFFLAT},
    {engine::EngineType::FAISS_IVFSQ8, NAME_ENGINE_TYPE_IVFSQ8},
    {engine::EngineType::FAISS_IVFSQ8H, NAME_ENGINE_TYPE_IVFSQ8H},
    {engine::EngineType::NSG_MIX, NAME_ENGINE_TYPE_RNSG},
    {engine::EngineType::FAISS_PQ, NAME_ENGINE_TYPE_IVFPQ},
};

static const std::unordered_map<std::string, engine::EngineType> IndexNameMap = {
    {NAME_ENGINE_TYPE_FLAT, engine::EngineType::FAISS_IDMAP},
    {NAME_ENGINE_TYPE_IVFFLAT, engine::EngineType::FAISS_IVFFLAT},
    {NAME_ENGINE_TYPE_IVFSQ8, engine::EngineType::FAISS_IVFSQ8},
    {NAME_ENGINE_TYPE_IVFSQ8H, engine::EngineType::FAISS_IVFSQ8H},
    {NAME_ENGINE_TYPE_RNSG, engine::EngineType::NSG_MIX},
    {NAME_ENGINE_TYPE_IVFPQ, engine::EngineType::FAISS_PQ},
};

static const std::unordered_map<engine::MetricType, std::string> MetricMap = {
    {engine::MetricType::L2, NAME_METRIC_TYPE_L2},
    {engine::MetricType::IP, NAME_METRIC_TYPE_IP},
};

}  // namespace web
}  // namespace server
}  // namespace milvus
