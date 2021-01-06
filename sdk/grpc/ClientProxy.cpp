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

#include "grpc/ClientProxy.h"
#include "thirdparty/nlohmann/json.hpp"

#include <memory>
#include <string>
#include <vector>

#include "grpc-gen/gen-milvus/milvus.grpc.pb.h"

#define MILVUS_SDK_VERSION "0.9.0";

namespace milvus {

using JSON = nlohmann::json;
static const char* EXTRA_PARAM_KEY = "params";

#define CLIENT_NULL_CHECK(client_ptr)                                                            \
    do {                                                                                         \
        if (client_ptr == nullptr) {                                                             \
            return Status(StatusCode::NotConnected, "Client is not connected to milvus server"); \
        }                                                                                        \
    } while (false)

bool
UriCheck(const std::string& uri) {
    size_t index = uri.find_first_of(':', 0);
    return (index != std::string::npos);
}

int64_t
GetRandomNumber() {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 64);
    int64_t place_number = dist(rng);
    return place_number;
}

template <typename T>
void
ConstructSearchParam(const std::string& collection_name, const std::vector<std::string>& partition_tag_array,
                     int64_t topk, const std::string& extra_params, T& search_param) {
    search_param.set_collection_name(collection_name);
    milvus::grpc::KeyValuePair* kv = search_param.add_extra_params();
    kv->set_key(EXTRA_PARAM_KEY);
    kv->set_value(extra_params);

    for (auto& tag : partition_tag_array) {
        search_param.add_partition_tag_array(tag);
    }
}

void
CopyRowRecord(::milvus::grpc::VectorRowRecord* target, const VectorData& src) {
    if (!src.float_data.empty()) {
        auto vector_data = target->mutable_float_data();
        vector_data->Resize(static_cast<int>(src.float_data.size()), 0.0);
        memcpy(vector_data->mutable_data(), src.float_data.data(), src.float_data.size() * sizeof(float));
    }

    if (!src.binary_data.empty()) {
        target->set_binary_data(src.binary_data.data(), src.binary_data.size());
    }
}

void
ConstructTopkQueryResult(const ::milvus::grpc::QueryResult& grpc_result, TopKQueryResult& topk_query_result) {
    int64_t nq = grpc_result.row_num();
    if (nq == 0) {
        return;
    }
    topk_query_result.reserve(nq);

    const auto& grpc_entity = grpc_result.entities();
    int64_t topk = grpc_entity.ids_size() / nq;
    int64_t offset = 0;
    for (int64_t i = 0; i < grpc_result.row_num(); i++) {
        milvus::QueryResult one_result = milvus::QueryResult();
        one_result.ids.resize(topk);
        one_result.distances.resize(topk);
        memcpy(one_result.ids.data(), grpc_entity.ids().data() + topk * i, topk * sizeof(int64_t));
        memcpy(one_result.distances.data(), grpc_result.distances().data() + topk * i, topk * sizeof(float));

        int valid_size = one_result.ids.size();
        while (valid_size > 0 && one_result.ids[valid_size - 1] == -1) {
            valid_size--;
        }
        if (valid_size != topk) {
            one_result.ids.resize(valid_size);
            one_result.distances.resize(valid_size);
        }

        for (int64_t k = 0; k < topk; k++) {
            std::unordered_map<std::string, std::any> scalar_data;
            std::unordered_map<std::string, milvus::VectorData> vector_data;
            if (grpc_entity.valid_row(i * topk + k)) {
                for (int64_t j = 0; j < grpc_entity.fields_size(); j++) {
                    const auto& grpc_field = grpc_entity.fields(j);
                    if (grpc_field.has_attr_record()) {
                        if (grpc_field.attr_record().int32_value_size() > 0) {
                            scalar_data.insert({grpc_field.field_name(), grpc_field.attr_record().int32_value(offset)});
                        } else if (grpc_field.attr_record().int64_value_size() > 0) {
                            scalar_data.insert({grpc_field.field_name(), grpc_field.attr_record().int64_value(offset)});
                        } else if (grpc_field.attr_record().float_value_size() > 0) {
                            scalar_data.insert({grpc_field.field_name(), grpc_field.attr_record().float_value(offset)});
                        } else {
                            scalar_data.insert(
                                {grpc_field.field_name(), grpc_field.attr_record().double_value(offset)});
                        }
                    } else {
                        auto float_size = grpc_field.vector_record().records(offset).float_data_size();
                        auto bin_size = grpc_field.vector_record().records(offset).binary_data().size();
                        milvus::VectorData vectors;
                        if (float_size > 0) {
                            std::vector<float> float_data(float_size);
                            memcpy(float_data.data(), grpc_field.vector_record().records(offset).float_data().data(),
                                   sizeof(float) * float_size);
                            vectors.float_data = float_data;
                        } else if (bin_size > 0) {
                            std::vector<uint8_t> bin_data(bin_size);
                            memcpy(bin_data.data(), grpc_field.vector_record().records(offset).binary_data().data(),
                                   bin_size);
                            vectors.binary_data = bin_data;
                        }
                        vector_data.insert({grpc_field.field_name(), vectors});
                    }
                }
                if (!scalar_data.empty() || !vector_data.empty()) {
                    Entity entity;
                    entity.entity_id = grpc_entity.ids(i * topk + k);
                    entity.scalar_data = scalar_data;
                    entity.vector_data = vector_data;
                    one_result.entities.emplace_back(entity);
                    offset++;
                }
            }
        }

        topk_query_result.emplace_back(one_result);
    }
}

void
CopyFieldValue(const FieldValue& field_value, ::milvus::grpc::InsertParam& insert_param) {
    if (!field_value.int32_value.empty()) {
        for (auto& field_it : field_value.int32_value) {
            auto grpc_field = insert_param.add_fields();
            grpc_field->set_field_name(field_it.first);
            auto grpc_attr_record = grpc_field->mutable_attr_record();
            auto grpc_int32_value = grpc_attr_record->mutable_int32_value();
            auto field_data = field_it.second;
            auto data_size = field_data.size();

            grpc_int32_value->Resize(static_cast<int>(data_size), 0);
            memcpy(grpc_int32_value->mutable_data(), field_data.data(), data_size * sizeof(int32_t));
        }
    }
    if (!field_value.int64_value.empty()) {
        for (auto& field_it : field_value.int64_value) {
            auto grpc_field = insert_param.add_fields();
            grpc_field->set_field_name(field_it.first);
            auto grpc_attr_record = grpc_field->mutable_attr_record();
            auto grpc_int64_value = grpc_attr_record->mutable_int64_value();
            auto field_data = field_it.second;
            auto data_size = field_data.size();

            grpc_int64_value->Resize(static_cast<int>(data_size), 0);
            memcpy(grpc_int64_value->mutable_data(), field_data.data(), data_size * sizeof(int64_t));
        }
    }
    if (!field_value.float_value.empty()) {
        for (auto& field_it : field_value.float_value) {
            auto grpc_field = insert_param.add_fields();
            grpc_field->set_field_name(field_it.first);
            auto grpc_attr_record = grpc_field->mutable_attr_record();
            auto grpc_float_value = grpc_attr_record->mutable_float_value();
            auto field_data = field_it.second;
            auto data_size = field_data.size();

            grpc_float_value->Resize(static_cast<int>(data_size), 0.0);
            memcpy(grpc_float_value->mutable_data(), field_data.data(), data_size * sizeof(float));
        }
    }
    if (!field_value.double_value.empty()) {
        for (auto& field_it : field_value.double_value) {
            auto grpc_field = insert_param.add_fields();
            grpc_field->set_field_name(field_it.first);
            auto grpc_attr_record = grpc_field->mutable_attr_record();
            auto grpc_double_value = grpc_attr_record->mutable_double_value();
            auto field_data = field_it.second;
            auto data_size = field_data.size();

            grpc_double_value->Resize(static_cast<int>(data_size), 0.0);
            memcpy(grpc_double_value->mutable_data(), field_data.data(), data_size * sizeof(double));
        }
    }
    if (!field_value.vector_value.empty()) {
        for (auto& field_it : field_value.vector_value) {
            auto grpc_field = insert_param.add_fields();
            grpc_field->set_field_name(field_it.first);
            auto grpc_vector_record = grpc_field->mutable_vector_record();
            for (const auto& vector_data : field_it.second) {
                auto row_record = grpc_vector_record->add_records();
                CopyRowRecord(row_record, vector_data);
            }
        }
    }
}

void
CopyEntities(::milvus::grpc::Entities& grpc_entities, Entities& entities) {
    auto grpc_field_size = grpc_entities.fields_size();
    std::vector<std::string> field_names(grpc_field_size);
    for (int64_t i = 0; i < grpc_field_size; i++) {
        field_names[i] = grpc_entities.fields(i).field_name();
    }

    int row_num = grpc_entities.ids_size();
    int64_t offset = 0;
    for (int64_t i = 0; i < row_num; i++) {
        if (!grpc_entities.valid_row(i)) {
            continue;
        }
        milvus::Entity entity = milvus::Entity();
        entity.entity_id = grpc_entities.ids(i);
        for (int64_t j = 0; j < grpc_field_size; j++) {
            const auto& grpc_field = grpc_entities.fields(j);
            auto field_name = grpc_field.field_name();
            if (grpc_field.has_attr_record()) {
                const auto& grpc_attr_record = grpc_field.attr_record();
                if (grpc_attr_record.int32_value_size() > 0) {
                    entity.scalar_data.insert({field_name, grpc_attr_record.int32_value(offset)});
                } else if (grpc_attr_record.int64_value_size() > 0) {
                    entity.scalar_data.insert({field_name, grpc_attr_record.int64_value(offset)});
                } else if (grpc_attr_record.float_value_size() > 0) {
                    entity.scalar_data.insert({field_name, grpc_attr_record.float_value(offset)});
                } else if (grpc_attr_record.double_value_size() > 0) {
                    entity.scalar_data.insert({field_name, grpc_attr_record.double_value(offset)});
                }
            } else if (grpc_field.has_vector_record()) {
                const auto& grpc_vector_record = grpc_field.vector_record();
                const auto& record = grpc_vector_record.records(offset);
                milvus::VectorData vector_data;
                if (record.float_data_size() > 0) {
                    std::vector<float> data(record.float_data_size());
                    memcpy(data.data(), record.float_data().data(), record.float_data_size() * sizeof(float));
                    vector_data.float_data = data;
                } else if (record.binary_data().size() > 0) {
                    std::vector<uint8_t> data(record.binary_data().size());
                    memcpy(data.data(), record.binary_data().data(), record.binary_data().size());
                    vector_data.binary_data = data;
                }
                entity.vector_data.insert({field_name, vector_data});
            }
        }
        entities.emplace_back(entity);
        offset++;
    }
}

void
CopyEntityToJson(::milvus::grpc::Entities& grpc_entities, JSON& json_entity) {
    int i;
    auto grpc_field_size = grpc_entities.fields_size();
    std::vector<std::string> field_names(grpc_field_size);
    for (i = 0; i < grpc_field_size; i++) {
        field_names[i] = grpc_entities.fields(i).field_name();
    }

    std::unordered_map<std::string, std::vector<int32_t>> int32_data;
    std::unordered_map<std::string, std::vector<int64_t>> int64_data;
    std::unordered_map<std::string, std::vector<float>> float_data;
    std::unordered_map<std::string, std::vector<double>> double_data;
    std::unordered_map<std::string, std::vector<milvus::VectorData>> vector_data;

    int row_num = grpc_entities.ids_size();
    for (i = 0; i < grpc_field_size; i++) {
        const auto& grpc_field = grpc_entities.fields(i);
        const auto& grpc_attr_record = grpc_field.attr_record();
        const auto& grpc_vector_record = grpc_field.vector_record();
        switch (grpc_field.type()) {
            case ::milvus::grpc::INT8:
            case ::milvus::grpc::INT16:
            case ::milvus::grpc::INT32: {
                std::vector<int32_t> data(row_num, 0);
                int64_t offset = 0;
                for (int64_t j = 0; j < row_num; j++) {
                    if (grpc_entities.valid_row(j)) {
                        data[j] = grpc_attr_record.int32_value(offset);
                        offset++;
                    }
                }
                // memcpy(data.data(), grpc_attr_record.int32_value().data(), row_num * sizeof(int32_t));
                int32_data.insert(std::make_pair(grpc_field.field_name(), data));
                break;
            }
            case ::milvus::grpc::INT64: {
                std::vector<int64_t> data(row_num, 0);
                int64_t offset = 0;
                for (int64_t j = 0; j < row_num; j++) {
                    if (grpc_entities.valid_row(j)) {
                        data[j] = grpc_attr_record.int64_value(offset);
                        offset++;
                    }
                }
                // memcpy(data.data(), grpc_attr_record.int64_value().data(), row_num * sizeof(int64_t));
                int64_data.insert(std::make_pair(grpc_field.field_name(), data));
                break;
            }
            case ::milvus::grpc::FLOAT: {
                std::vector<float> data(row_num, 0);
                int64_t offset = 0;
                for (int64_t j = 0; j < row_num; j++) {
                    if (grpc_entities.valid_row(j)) {
                        data[j] = grpc_attr_record.float_value(offset);
                        offset++;
                    }
                }
                // memcpy(data.data(), grpc_attr_record.float_value().data(), row_num * sizeof(float));
                float_data.insert(std::make_pair(grpc_field.field_name(), data));
                break;
            }
            case ::milvus::grpc::DOUBLE: {
                std::vector<double> data(row_num, 0);
                int64_t offset = 0;
                for (int64_t j = 0; j < row_num; j++) {
                    if (grpc_entities.valid_row(j)) {
                        data[j] = grpc_attr_record.double_value(offset);
                        offset++;
                    }
                }
                // memcpy(data.data(), grpc_attr_record.double_value().data(), row_num * sizeof(double));
                double_data.insert(std::make_pair(grpc_field.field_name(), data));
                break;
            }
            case ::milvus::grpc::VECTOR_FLOAT: {
                std::vector<milvus::VectorData> data(row_num);
                for (int j = 0; j < row_num; j++) {
                    size_t dim = grpc_vector_record.records(j).float_data_size();
                    data[j].float_data.resize(dim);
                    memcpy(data[j].float_data.data(), grpc_vector_record.records(j).float_data().data(),
                           dim * sizeof(float));
                }
                vector_data.insert(std::make_pair(grpc_field.field_name(), data));
                break;
            }
            case ::milvus::grpc::VECTOR_BINARY: {
                // TODO (yukun)
            }
            default: {}
        }
    }

    for (i = 0; i < row_num; i++) {
        JSON one_json;
        one_json["id"] = grpc_entities.ids(i);
        if (grpc_entities.valid_row(i)) {
            for (const auto& name : field_names) {
                if (int32_data.find(name) != int32_data.end()) {
                    one_json[name] = int32_data.at(name)[i];
                } else if (int64_data.find(name) != int64_data.end()) {
                    one_json[name] = int64_data.at(name)[i];
                } else if (float_data.find(name) != float_data.end()) {
                    one_json[name] = float_data.at(name)[i];
                } else if (double_data.find(name) != double_data.end()) {
                    one_json[name] = double_data.at(name)[i];
                } else if (vector_data.find(name) != vector_data.end()) {
                    if (!(vector_data.at(name)[i].float_data.empty())) {
                        one_json[name] = vector_data.at(name)[i].float_data;
                    } else if (!(vector_data.at(name)[i].binary_data.empty())) {
                        one_json[name] = vector_data.at(name)[i].binary_data;
                    }
                }
            }
        }
        json_entity.emplace_back(one_json);
    }
}

Status
ClientProxy::Connect(const ConnectParam& param) {
    std::string uri = param.ip_address + ":" + param.port;

    ::grpc::ChannelArguments args;
    args.SetMaxSendMessageSize(-1);
    args.SetMaxReceiveMessageSize(-1);
    channel_ = ::grpc::CreateCustomChannel(uri, ::grpc::InsecureChannelCredentials(), args);
    if (channel_ != nullptr) {
        connected_ = true;
        client_ptr_ = std::make_shared<GrpcClient>(channel_);
        return Status::OK();
    }

    std::string reason = "Connect failed!";
    connected_ = false;
    return Status(StatusCode::NotConnected, reason);
}

Status
ClientProxy::Connect(const std::string& uri) {
    if (!UriCheck(uri)) {
        return Status(StatusCode::InvalidAgument, "Invalid uri");
    }
    size_t index = uri.find_first_of(':', 0);

    ConnectParam param;
    param.ip_address = uri.substr(0, index);
    param.port = uri.substr(index + 1);

    return Connect(param);
}

Status
ClientProxy::Connected() const {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        std::string info;
        return client_ptr_->Cmd("", info);
    } catch (std::exception& ex) {
        return Status(StatusCode::NotConnected, "Connection lost: " + std::string(ex.what()));
    }
}

Status
ClientProxy::Disconnect() {
    try {
        Status status = client_ptr_->Disconnect();
        connected_ = false;
        channel_.reset();
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to disconnect: " + std::string(ex.what()));
    }
}

Status
ClientProxy::CreateCollection(const Mapping& mapping) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::Mapping grpc_mapping;
        grpc_mapping.set_collection_name(mapping.collection_name);
        for (auto& field : mapping.fields) {
            auto grpc_field = grpc_mapping.add_fields();
            grpc_field->set_name(field->name);
            grpc_field->set_type((::milvus::grpc::DataType)field->type);

            auto grpc_extra_param = grpc_field->add_extra_params();
            grpc_extra_param->set_key(EXTRA_PARAM_KEY);
            grpc_extra_param->set_value(field->params);
        }
        auto grpc_param = grpc_mapping.add_extra_params();
        grpc_param->set_key(EXTRA_PARAM_KEY);
        grpc_param->set_value(mapping.extra_params);

        return client_ptr_->CreateCollection(grpc_mapping);
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to create collection: " + std::string(ex.what()));
    }
}

Status
ClientProxy::DropCollection(const std::string& collection_name) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::CollectionName grpc_collection_name;
        grpc_collection_name.set_collection_name(collection_name);
        return client_ptr_->DropCollection(grpc_collection_name);
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to drop collection: " + std::string(ex.what()));
    }
}

bool
ClientProxy::HasCollection(const std::string& collection_name) {
    try {
        Status status = Status::OK();
        ::milvus::grpc::CollectionName grpc_collection_name;
        grpc_collection_name.set_collection_name(collection_name);
        return client_ptr_->HasCollection(grpc_collection_name, status);
    } catch (std::exception& ex) {
        return false;
    }
}

Status
ClientProxy::ListCollections(std::vector<std::string>& collection_array) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        Status status;
        milvus::grpc::CollectionNameList collection_name_list;
        status = client_ptr_->ListCollections(collection_name_list);

        collection_array.resize(collection_name_list.collection_names_size());
        for (uint64_t i = 0; i < collection_name_list.collection_names_size(); ++i) {
            collection_array[i] = collection_name_list.collection_names(i);
        }
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to list collections: " + std::string(ex.what()));
    }
}

Status
ClientProxy::GetCollectionInfo(const std::string& collection_name, Mapping& mapping) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::Mapping grpc_mapping;

        Status status = client_ptr_->GetCollectionInfo(collection_name, grpc_mapping);

        mapping.collection_name = collection_name;
        for (int64_t i = 0; i < grpc_mapping.fields_size(); i++) {
            const auto& grpc_field = grpc_mapping.fields(i);
            FieldPtr field_ptr = std::make_shared<Field>();
            field_ptr->name = grpc_field.name();
            JSON json_index_params;
            for (int64_t j = 0; j < grpc_field.index_params_size(); j++) {
                JSON json_param;
                json_param[grpc_field.index_params(j).key()] = grpc_field.index_params(j).value();
                json_index_params.emplace_back(json_param);
            }
            field_ptr->index_params = json_index_params.dump();
            JSON json_extra_params;
            for (int64_t j = 0; j < grpc_field.extra_params_size(); j++) {
                JSON json_param;
                json_param = JSON::parse(grpc_field.extra_params(j).value());
                json_extra_params.emplace_back(json_param);
            }
            field_ptr->params = json_extra_params.dump();
            field_ptr->type = (DataType)grpc_field.type();
            mapping.fields.emplace_back(field_ptr);
        }
        if (!grpc_mapping.extra_params().empty()) {
            mapping.extra_params = grpc_mapping.extra_params(0).value();
        }
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to get collection info: " + std::string(ex.what()));
    }
}

Status
ClientProxy::GetCollectionStats(const std::string& collection_name, std::string& collection_stats) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        Status status;
        ::milvus::grpc::CollectionName grpc_collection_name;
        grpc_collection_name.set_collection_name(collection_name);
        milvus::grpc::CollectionInfo grpc_collection_stats;
        status = client_ptr_->GetCollectionStats(grpc_collection_name, grpc_collection_stats);

        collection_stats = grpc_collection_stats.json_info();

        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to get collection stats: " + std::string(ex.what()));
    }
}

Status
ClientProxy::CountEntities(const std::string& collection_name, int64_t& row_count) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        Status status;
        ::milvus::grpc::CollectionName grpc_collection_name;
        grpc_collection_name.set_collection_name(collection_name);
        row_count = client_ptr_->CountEntities(grpc_collection_name, status);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to count collection: " + std::string(ex.what()));
    }
}

Status
ClientProxy::CreatePartition(const PartitionParam& partition_param) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::PartitionParam grpc_partition_param;
        grpc_partition_param.set_collection_name(partition_param.collection_name);
        grpc_partition_param.set_tag(partition_param.partition_tag);
        Status status = client_ptr_->CreatePartition(grpc_partition_param);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to create partition: " + std::string(ex.what()));
    }
}

Status
ClientProxy::DropPartition(const PartitionParam& partition_param) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::PartitionParam grpc_partition_param;
        grpc_partition_param.set_collection_name(partition_param.collection_name);
        grpc_partition_param.set_tag(partition_param.partition_tag);
        Status status = client_ptr_->DropPartition(grpc_partition_param);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to drop partition: " + std::string(ex.what()));
    }
}

bool
ClientProxy::HasPartition(const std::string& collection_name, const std::string& partition_tag) const {
    try {
        Status status = Status::OK();
        ::milvus::grpc::PartitionParam grpc_partition_param;
        grpc_partition_param.set_collection_name(collection_name);
        grpc_partition_param.set_tag(partition_tag);
        return client_ptr_->HasPartition(grpc_partition_param, status);
    } catch (std::exception& ex) {
        return false;
    }
}

Status
ClientProxy::ListPartitions(const std::string& collection_name, PartitionTagList& partition_tag_array) const {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::CollectionName grpc_collection_name;
        grpc_collection_name.set_collection_name(collection_name);
        ::milvus::grpc::PartitionList grpc_partition_list;
        Status status = client_ptr_->ListPartitions(grpc_collection_name, grpc_partition_list);
        partition_tag_array.resize(grpc_partition_list.partition_tag_array_size());
        for (uint64_t i = 0; i < grpc_partition_list.partition_tag_array_size(); ++i) {
            partition_tag_array[i] = grpc_partition_list.partition_tag_array(i);
        }
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to show partitions: " + std::string(ex.what()));
    }
}

Status
ClientProxy::CreateIndex(const IndexParam& index_param) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::IndexParam grpc_index_param;
        grpc_index_param.set_collection_name(index_param.collection_name);
        grpc_index_param.set_field_name(index_param.field_name);
        JSON json_param = JSON::parse(index_param.index_params);
        for (auto& item : json_param.items()) {
            milvus::grpc::KeyValuePair* kv = grpc_index_param.add_extra_params();
            kv->set_key(item.key());
            if (item.value().is_object()) {
                kv->set_value(item.value().dump());
            } else {
                kv->set_value(item.value());
            }
        }
        return client_ptr_->CreateIndex(grpc_index_param);
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to build index: " + std::string(ex.what()));
    }
}

Status
ClientProxy::DropIndex(const std::string& collection_name, const std::string& field_name,
                       const std::string& index_name) const {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::IndexParam grpc_index_param;
        grpc_index_param.set_collection_name(collection_name);
        grpc_index_param.set_field_name(field_name);
        grpc_index_param.set_index_name(index_name);
        Status status = client_ptr_->DropIndex(grpc_index_param);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to drop index: " + std::string(ex.what()));
    }
}

Status
ClientProxy::Insert(const std::string& collection_name, const std::string& partition_tag, const FieldValue& field_value,
                    std::vector<int64_t>& id_array) {
    CLIENT_NULL_CHECK(client_ptr_);
    Status status = Status::OK();
    try {
        ::milvus::grpc::InsertParam insert_param;
        insert_param.set_collection_name(collection_name);
        insert_param.set_partition_tag(partition_tag);

        CopyFieldValue(field_value, insert_param);

        // Single thread
        ::milvus::grpc::EntityIds entity_ids;
        if (!id_array.empty()) {
            /* set user's ids */
            auto row_ids = insert_param.mutable_entity_id_array();
            row_ids->Resize(static_cast<int>(id_array.size()), -1);
            memcpy(row_ids->mutable_data(), id_array.data(), id_array.size() * sizeof(int64_t));
            status = client_ptr_->Insert(insert_param, entity_ids);
        } else {
            status = client_ptr_->Insert(insert_param, entity_ids);
            /* return Milvus generated ids back to user */
            id_array.insert(id_array.end(), entity_ids.entity_id_array().begin(), entity_ids.entity_id_array().end());
        }
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to add entities: " + std::string(ex.what()));
    }

    return status;
}

Status
ClientProxy::GetEntityByID(const std::string& collection_name, const std::vector<int64_t>& id_array,
                           Entities& entities) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::EntityIdentity entity_identity;
        entity_identity.set_collection_name(collection_name);
        for (auto id : id_array) {
            entity_identity.add_id_array(id);
        }
        ::milvus::grpc::Entities grpc_entities;

        Status status = client_ptr_->GetEntityByID(entity_identity, grpc_entities);
        if (!status.ok()) {
            return status;
        }

        CopyEntities(grpc_entities, entities);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to get entity by id: " + std::string(ex.what()));
    }
}

Status
ClientProxy::DeleteEntityByID(const std::string& collection_name, const std::vector<int64_t>& id_array) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::DeleteByIDParam delete_by_id_param;
        delete_by_id_param.set_collection_name(collection_name);
        for (auto id : id_array) {
            delete_by_id_param.add_id_array(id);
        }

        return client_ptr_->DeleteEntityByID(delete_by_id_param);
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to delete entity id: " + std::string(ex.what()));
    }
}

void
ParseDsl(const milvus::Mapping& mapping, nlohmann::json& dsl_json, nlohmann::json& vector_json,
         std::vector<VectorData>& vector_records) {
    auto str = dsl_json.dump();
    if (dsl_json.is_array()) {
        for (auto& query : dsl_json) {
            //            auto aaa = query.begin().key();
            if (query.contains("vector")) {
                auto vector_query_json = query["vector"];
                auto placeholder = "placeholder" + std::to_string(GetRandomNumber());
                query["vector"] = placeholder;
                for (auto& element : vector_query_json.items()) {
                    const auto& name = element.key();
                    for (const auto& field : mapping.fields) {
                        if (field->name == name) {
                            if (field->type == milvus::DataType::VECTOR_FLOAT) {
                                auto embedding = element.value()["query"].get<std::vector<std::vector<float>>>();
                                for (const auto& data : embedding) {
                                    milvus::VectorData vector_data = {data, std::vector<uint8_t>()};
                                    vector_records.emplace_back(vector_data);
                                }
                            } else if (field->type == milvus::DataType::VECTOR_BINARY) {
                                auto embedding = element.value()["query"].get<std::vector<std::vector<uint8_t>>>();
                                for (const auto& data : embedding) {
                                    milvus::VectorData vector_data = {std::vector<float>(), data};
                                    vector_records.emplace_back(vector_data);
                                }
                            }
                        }
                    }

                    element.value().erase("query");
                    vector_json[placeholder] = vector_query_json;
                    return;
                }
            } else if (query.contains("must") || query.contains("should") || query.contains("must_not")) {
                ParseDsl(mapping, query.begin().value(), vector_json, vector_records);
            }
        }
    } else {
        for (auto& object : dsl_json.items()) {
            auto aaa = object.key();
            ParseDsl(mapping, object.value(), vector_json, vector_records);
        }
    }
}

Status
ClientProxy::Search(const std::string& collection_name, const std::vector<std::string>& partition_list,
                    nlohmann::json& dsl, const std::string& extra_params, TopKQueryResult& query_result) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        milvus::Mapping mapping;
        auto status = GetCollectionInfo(collection_name, mapping);
        if (!status.ok()) {
            return status;
        }
        nlohmann::json vector_json;
        std::vector<milvus::VectorData> vector_records;
        auto str = dsl.dump();
        ParseDsl(mapping, dsl, vector_json, vector_records);

        ::milvus::grpc::SearchParam search_param;
        search_param.set_collection_name(collection_name);
        for (const auto& partition : partition_list) {
            auto value = search_param.add_partition_tag_array();
            *value = partition;
        }
        search_param.set_dsl(dsl.dump());
        auto grpc_vector_param = search_param.add_vector_param();
        grpc_vector_param->set_json(vector_json.dump());
        auto grpc_vector_record = grpc_vector_param->mutable_row_record();
        for (auto& vector_data : vector_records) {
            auto row_record = grpc_vector_record->add_records();
            CopyRowRecord(row_record, vector_data);
        }

        if (!extra_params.empty()) {
            auto extra_param = search_param.add_extra_params();
            extra_param->set_key("params");
            extra_param->set_value(extra_params);
        }

        ::milvus::grpc::QueryResult grpc_result;
        status = client_ptr_->Search(search_param, grpc_result);
        ConstructTopkQueryResult(grpc_result, query_result);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to search entities: " + std::string(ex.what()));
    }
}

Status
ClientProxy::ListIDInSegment(const std::string& collection_name, const int64_t& segment_id,
                             std::vector<int64_t>& id_array) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::GetEntityIDsParam param;
        param.set_collection_name(collection_name);
        param.set_segment_id(segment_id);

        ::milvus::grpc::EntityIds entity_ids;
        Status status = client_ptr_->ListIDInSegment(param, entity_ids);
        if (!status.ok()) {
            return status;
        }
        id_array.insert(id_array.end(), entity_ids.entity_id_array().begin(), entity_ids.entity_id_array().end());
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to get ids from segment: " + std::string(ex.what()));
    }
}

Status
ClientProxy::LoadCollection(const std::string& collection_name) const {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::CollectionName grpc_collection_name;
        grpc_collection_name.set_collection_name(collection_name);
        Status status = client_ptr_->LoadCollection(grpc_collection_name);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to preload collection: " + std::string(ex.what()));
    }
}

Status
ClientProxy::Flush(const std::vector<std::string>& collection_name_array) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        if (collection_name_array.empty()) {
            return client_ptr_->Flush("");
        } else {
            for (auto& collection_name : collection_name_array) {
                client_ptr_->Flush(collection_name);
            }
        }
        return Status::OK();
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to flush collection");
    }
}

Status
ClientProxy::Compact(const std::string& collection_name, const double& threshold) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        ::milvus::grpc::CompactParam grpc_compact_param;
        grpc_compact_param.set_collection_name(collection_name);
        grpc_compact_param.set_threshold(threshold);
        Status status = client_ptr_->Compact(grpc_compact_param);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to compact collection: " + std::string(ex.what()));
    }
}

/*******************************New Interface**********************************/

void
WriteQueryToProto(::milvus::grpc::GeneralQuery* general_query, const BooleanQueryPtr& boolean_query) {
    if (!boolean_query->GetBooleanQueries().empty()) {
        for (const auto& query : boolean_query->GetBooleanQueries()) {
            auto grpc_boolean_query = general_query->mutable_boolean_query();
            grpc_boolean_query->set_occur((::milvus::grpc::Occur)query->GetOccur());

            for (const auto& leaf_query : query->GetLeafQueries()) {
                auto grpc_query = grpc_boolean_query->add_general_query();
                if (leaf_query->term_query_ptr != nullptr) {
                    auto term_query = grpc_query->mutable_term_query();
                    term_query->set_field_name(leaf_query->term_query_ptr->field_name);
                    term_query->set_boost(leaf_query->query_boost);
                    if (leaf_query->term_query_ptr->int_value.size() > 0) {
                        auto mutable_int_value = term_query->mutable_int_value();
                        auto size = leaf_query->term_query_ptr->int_value.size();
                        mutable_int_value->Resize(size, 0);
                        memcpy(mutable_int_value->mutable_data(), leaf_query->term_query_ptr->int_value.data(),
                               size * sizeof(int64_t));
                    } else if (leaf_query->term_query_ptr->double_value.size() > 0) {
                        auto mutable_double_value = term_query->mutable_double_value();
                        auto size = leaf_query->term_query_ptr->double_value.size();
                        mutable_double_value->Resize(size, 0);
                        memcpy(mutable_double_value->mutable_data(), leaf_query->term_query_ptr->double_value.data(),
                               size * sizeof(double));
                    }
                }
                if (leaf_query->range_query_ptr != nullptr) {
                    auto range_query = grpc_query->mutable_range_query();
                    range_query->set_boost(leaf_query->query_boost);
                    range_query->set_field_name(leaf_query->range_query_ptr->field_name);
                    for (const auto& com_expr : leaf_query->range_query_ptr->compare_expr) {
                        auto grpc_com_expr = range_query->add_operand();
                        grpc_com_expr->set_operand(com_expr.operand);
                        grpc_com_expr->set_operator_((milvus::grpc::CompareOperator)com_expr.compare_operator);
                    }
                }
                if (leaf_query->vector_query_ptr != nullptr) {
                    auto vector_query = grpc_query->mutable_vector_query();
                    vector_query->set_field_name(leaf_query->vector_query_ptr->field_name);
                    vector_query->set_query_boost(leaf_query->query_boost);
                    vector_query->set_topk(leaf_query->vector_query_ptr->topk);
                    for (const auto& record : leaf_query->vector_query_ptr->query_vector) {
                        ::milvus::grpc::VectorRowRecord* row_record = vector_query->add_records();
                        CopyRowRecord(row_record, record);
                    }
                    auto extra_param = vector_query->add_extra_params();
                    extra_param->set_key(EXTRA_PARAM_KEY);
                    extra_param->set_value(leaf_query->vector_query_ptr->extra_params);
                }
            }

            if (!query->GetBooleanQueries().empty()) {
                ::milvus::grpc::GeneralQuery* next_query = grpc_boolean_query->add_general_query();
                WriteQueryToProto(next_query, query);
            }
        }
    }
}

/*
Status
ClientProxy::SearchPB(const std::string& collection_name, const std::vector<std::string>& partition_list,
                      BooleanQueryPtr& boolean_query, const std::string& extra_params,
                      TopKQueryResult& topk_query_result) {
    CLIENT_NULL_CHECK(client_ptr_);
    try {
        // convert boolean_query to proto
        ::milvus::grpc::SearchParamPB search_param;
        search_param.set_collection_name(collection_name);
        for (const auto& partition : partition_list) {
            auto value = search_param.add_partition_tag_array();
            *value = partition;
        }
        if (!extra_params.empty()) {
            auto extra_param = search_param.add_extra_params();
            extra_param->set_key("params");
            extra_param->set_value(extra_params);
        }
        WriteQueryToProto(search_param.mutable_general_query(), boolean_query);

        // step 2: search vectors
        ::milvus::grpc::QueryResult result;
        Status status = client_ptr_->SearchPB(search_param, result);

        // step 3: convert result array
        ConstructTopkQueryResult(result, topk_query_result);
        return status;
    } catch (std::exception& ex) {
        return Status(StatusCode::UnknownError, "Failed to search entities: " + std::string(ex.what()));
    }
}
 */

}  // namespace milvus
