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

#include "server/delivery/hybrid_request/CreateCollectionRequest.h"
#include "db/Utils.h"
#include "server/DBWrapper.h"
#include "server/delivery/request/BaseRequest.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <fiu-local.h>
#include <memory>
#include <string>

namespace milvus {
namespace server {

CreateCollectionRequest::CreateCollectionRequest(const std::shared_ptr<Context>& context,
                                                 const std::string& collection_name,
                                                 std::vector<std::pair<std::string, engine::meta::hybrid::DataType>>& field_types,
                                                 std::vector<std::pair<std::string, uint64_t>>& vector_dimensions,
                                                 std::vector<std::pair<std::string, std::string>>& field_params)
    : BaseRequest(context, DDL_DML_REQUEST_GROUP),
      collection_name_(collection_name),
      field_types_(field_types),
      vector_dimensions_(vector_dimensions),
      field_params_(field_params) {
}

BaseRequestPtr
CreateCollectionRequest::Create(const std::shared_ptr<Context>& context,
                                const std::string& collection_name,
                                std::vector<std::pair<std::string, engine::meta::hybrid::DataType>>& field_types,
                                std::vector<std::pair<std::string, uint64_t>>& vector_dimensions,
                                std::vector<std::pair<std::string, std::string>>& field_params) {
    return std::shared_ptr<BaseRequest>(
        new CreateCollectionRequest(context, collection_name, field_types, vector_dimensions, field_params));
}

Status
CreateCollectionRequest::OnExecute() {
    std::string hdr = "CreateCollectionRequest(collection=" + collection_name_ + ")";
    TimeRecorderAuto rc(hdr);

    try {
        // step 1: check arguments
        auto status = ValidationUtil::ValidateTableName(collection_name_);
        if (!status.ok()) {
            return status;
        }

        rc.RecordSection("check validation");

        // step 2: construct collection schema and vector schema
        engine::meta::hybrid::CollectionSchema collection_info;
        engine::meta::hybrid::FieldsSchema fields_schema;

        auto size = field_types_.size();
        collection_info.collection_id_ = collection_name_;
        collection_info.field_num = size;
        fields_schema.fields_schema_.resize(size);
        for (uint64_t i = 0; i < size; ++i) {
            fields_schema.fields_schema_[i].field_name_ =  field_types_[i].first;
            fields_schema.fields_schema_[i].field_type_ = (int32_t)field_types_[i].second;
            fields_schema.fields_schema_[i].field_params_ = field_params_[i].second;
        }

        // TODO(yukun): check dimension, metric_type, and assign engine_type

        // step 3: create collection
        status = DBWrapper::DB()->CreateHybridCollection(collection_info, fields_schema);
        if (!status.ok()) {
            // collection could exist
            if (status.code() == DB_ALREADY_EXIST) {
                return Status(SERVER_INVALID_TABLE_NAME, status.message());
            }
            return status;
        }
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
