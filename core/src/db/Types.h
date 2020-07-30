// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <faiss/Index.h>

#include <src/segment/Segment.h>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "db/meta/MetaTypes.h"
#include "utils/Json.h"

namespace milvus {
namespace engine {

typedef int64_t IDNumber;
typedef IDNumber* IDNumberPtr;
typedef std::vector<IDNumber> IDNumbers;

typedef faiss::Index::distance_t VectorDistance;
typedef std::vector<VectorDistance> VectorDistances;

typedef std::vector<faiss::Index::idx_t> ResultIds;
typedef std::vector<faiss::Index::distance_t> ResultDistances;

struct CollectionIndex {
    std::string index_name_;
    std::string metric_name_;
    milvus::json extra_params_ = {{"nlist", 2048}};
};

struct VectorsData {
    uint64_t vector_count_ = 0;
    std::vector<float> float_data_;
    std::vector<uint8_t> binary_data_;
    IDNumbers id_array_;
};

struct Entity {
    int64_t entity_count_ = 0;
    std::vector<uint8_t> attr_value_;
    std::unordered_map<std::string, VectorsData> vector_data_;
    IDNumbers id_array_;
};

struct AttrsData {
    uint64_t attr_count_ = 0;
    std::unordered_map<std::string, engine::meta::DataType> attr_type_;
    std::unordered_map<std::string, std::vector<uint8_t>> attr_data_;
    IDNumbers id_array_;
};

struct QueryResult {
    uint64_t row_num_;
    engine::ResultIds result_ids_;
    engine::ResultDistances result_distances_;
    engine::DataChunkPtr data_chunk_;
};
using QueryResultPtr = std::shared_ptr<QueryResult>;

using File2ErrArray = std::map<std::string, std::vector<std::string>>;
using Table2FileErr = std::map<std::string, File2ErrArray>;

extern const char* DEFAULT_UID_NAME;

extern const char* DEFAULT_RAW_DATA_NAME;
extern const char* DEFAULT_BLOOM_FILTER_NAME;
extern const char* DEFAULT_DELETED_DOCS_NAME;
extern const char* DEFAULT_INDEX_COMPRESS_NAME;
extern const char* DEFAULT_STRUCTURED_INDEX_NAME;

extern const char* PARAM_DIMENSION;
extern const char* PARAM_INDEX_METRIC_TYPE;
extern const char* PARAM_INDEX_EXTRA_PARAMS;
extern const char* PARAM_SEGMENT_ROW_COUNT;

constexpr int64_t MAX_NAME_LENGTH = 255;
constexpr int64_t MAX_DIMENSION = 32768;
constexpr int32_t MAX_SEGMENT_ROW_COUNT = 4 * 1024 * 1024;
constexpr int64_t DEFAULT_SEGMENT_ROW_COUNT = 100000;
constexpr int64_t M_BYTE = 1024 * 1024;
constexpr int64_t MAX_INSERT_DATA_SIZE = 256 * M_BYTE;

using FieldType = meta::DataType;

enum FieldElementType {
    FET_NONE = 0,
    FET_RAW = 1,
    FET_BLOOM_FILTER = 2,
    FET_DELETED_DOCS = 3,
    FET_INDEX = 4,
    FET_COMPRESS_SQ8 = 5,
};

}  // namespace engine
}  // namespace milvus
