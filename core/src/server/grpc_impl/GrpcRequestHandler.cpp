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

#include "server/grpc_impl/GrpcRequestHandler.h"

#include <fiu-local.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "context/HybridSearchContext.h"
#include "query/BinaryQuery.h"
#include "tracing/TextMapCarrier.h"
#include "tracing/TracerUtil.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace server {
namespace grpc {

::milvus::grpc::ErrorCode
ErrorMap(ErrorCode code) {
    static const std::map<ErrorCode, ::milvus::grpc::ErrorCode> code_map = {
        {SERVER_UNEXPECTED_ERROR, ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
        {SERVER_UNSUPPORTED_ERROR, ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
        {SERVER_NULL_POINTER, ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
        {SERVER_INVALID_ARGUMENT, ::milvus::grpc::ErrorCode::ILLEGAL_ARGUMENT},
        {SERVER_FILE_NOT_FOUND, ::milvus::grpc::ErrorCode::FILE_NOT_FOUND},
        {SERVER_NOT_IMPLEMENT, ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR},
        {SERVER_CANNOT_CREATE_FOLDER, ::milvus::grpc::ErrorCode::CANNOT_CREATE_FOLDER},
        {SERVER_CANNOT_CREATE_FILE, ::milvus::grpc::ErrorCode::CANNOT_CREATE_FILE},
        {SERVER_CANNOT_DELETE_FOLDER, ::milvus::grpc::ErrorCode::CANNOT_DELETE_FOLDER},
        {SERVER_CANNOT_DELETE_FILE, ::milvus::grpc::ErrorCode::CANNOT_DELETE_FILE},
        {SERVER_TABLE_NOT_EXIST, ::milvus::grpc::ErrorCode::TABLE_NOT_EXISTS},
        {SERVER_INVALID_TABLE_NAME, ::milvus::grpc::ErrorCode::ILLEGAL_TABLE_NAME},
        {SERVER_INVALID_TABLE_DIMENSION, ::milvus::grpc::ErrorCode::ILLEGAL_DIMENSION},
        {SERVER_INVALID_VECTOR_DIMENSION, ::milvus::grpc::ErrorCode::ILLEGAL_DIMENSION},

        {SERVER_INVALID_INDEX_TYPE, ::milvus::grpc::ErrorCode::ILLEGAL_INDEX_TYPE},
        {SERVER_INVALID_ROWRECORD, ::milvus::grpc::ErrorCode::ILLEGAL_ROWRECORD},
        {SERVER_INVALID_ROWRECORD_ARRAY, ::milvus::grpc::ErrorCode::ILLEGAL_ROWRECORD},
        {SERVER_INVALID_TOPK, ::milvus::grpc::ErrorCode::ILLEGAL_TOPK},
        {SERVER_INVALID_NPROBE, ::milvus::grpc::ErrorCode::ILLEGAL_ARGUMENT},
        {SERVER_INVALID_INDEX_NLIST, ::milvus::grpc::ErrorCode::ILLEGAL_NLIST},
        {SERVER_INVALID_INDEX_METRIC_TYPE, ::milvus::grpc::ErrorCode::ILLEGAL_METRIC_TYPE},
        {SERVER_INVALID_INDEX_FILE_SIZE, ::milvus::grpc::ErrorCode::ILLEGAL_ARGUMENT},
        {SERVER_ILLEGAL_VECTOR_ID, ::milvus::grpc::ErrorCode::ILLEGAL_VECTOR_ID},
        {SERVER_ILLEGAL_SEARCH_RESULT, ::milvus::grpc::ErrorCode::ILLEGAL_SEARCH_RESULT},
        {SERVER_CACHE_FULL, ::milvus::grpc::ErrorCode::CACHE_FAILED},
        {DB_META_TRANSACTION_FAILED, ::milvus::grpc::ErrorCode::META_FAILED},
        {SERVER_BUILD_INDEX_ERROR, ::milvus::grpc::ErrorCode::BUILD_INDEX_ERROR},
        {SERVER_OUT_OF_MEMORY, ::milvus::grpc::ErrorCode::OUT_OF_MEMORY},
    };

    if (code_map.find(code) != code_map.end()) {
        return code_map.at(code);
    } else {
        return ::milvus::grpc::ErrorCode::UNEXPECTED_ERROR;
    }
}

namespace {
void
CopyRowRecords(const google::protobuf::RepeatedPtrField<::milvus::grpc::RowRecord>& grpc_records,
               const google::protobuf::RepeatedField<google::protobuf::int64>& grpc_id_array,
               engine::VectorsData& vectors) {
    // step 1: copy vector data
    int64_t float_data_size = 0, binary_data_size = 0;
    for (auto& record : grpc_records) {
        float_data_size += record.float_data_size();
        binary_data_size += record.binary_data().size();
    }

    std::vector<float> float_array(float_data_size, 0.0f);
    std::vector<uint8_t> binary_array(binary_data_size, 0);
    int64_t float_offset = 0, binary_offset = 0;
    if (float_data_size > 0) {
        for (auto& record : grpc_records) {
            memcpy(&float_array[float_offset], record.float_data().data(), record.float_data_size() * sizeof(float));
            float_offset += record.float_data_size();
        }
    } else if (binary_data_size > 0) {
        for (auto& record : grpc_records) {
            memcpy(&binary_array[binary_offset], record.binary_data().data(), record.binary_data().size());
            binary_offset += record.binary_data().size();
        }
    }

    // step 2: copy id array
    std::vector<int64_t> id_array;
    if (grpc_id_array.size() > 0) {
        id_array.resize(grpc_id_array.size());
        memcpy(id_array.data(), grpc_id_array.data(), grpc_id_array.size() * sizeof(int64_t));
    }

    // step 3: contruct vectors
    vectors.vector_count_ = grpc_records.size();
    vectors.float_data_.swap(float_array);
    vectors.binary_data_.swap(binary_array);
    vectors.id_array_.swap(id_array);
}

void
ConstructResults(const TopKQueryResult& result, ::milvus::grpc::TopKQueryResult* response) {
    if (!response) {
        return;
    }

    response->set_row_num(result.row_num_);

    response->mutable_ids()->Resize(static_cast<int>(result.id_list_.size()), 0);
    memcpy(response->mutable_ids()->mutable_data(), result.id_list_.data(), result.id_list_.size() * sizeof(int64_t));

    response->mutable_distances()->Resize(static_cast<int>(result.distance_list_.size()), 0.0);
    memcpy(response->mutable_distances()->mutable_data(), result.distance_list_.data(),
           result.distance_list_.size() * sizeof(float));
}

void
ConstructPartitionStat(const PartitionStat& partition_stat, ::milvus::grpc::PartitionStat* grpc_partition_stat) {
    if (!grpc_partition_stat) {
        return;
    }

    grpc_partition_stat->set_total_row_count(partition_stat.total_row_num_);
    grpc_partition_stat->set_tag(partition_stat.tag_);

    for (auto& seg_stat : partition_stat.segments_stat_) {
        ::milvus::grpc::SegmentStat* grpc_seg_stat = grpc_partition_stat->mutable_segments_stat()->Add();
        grpc_seg_stat->set_row_count(seg_stat.row_num_);
        grpc_seg_stat->set_segment_name(seg_stat.name_);
        grpc_seg_stat->set_index_name(seg_stat.index_name_);
        grpc_seg_stat->set_data_size(seg_stat.data_size_);
    }
}

void
ConstructTableInfo(const TableInfo& table_info, ::milvus::grpc::TableInfo* response) {
    if (!response) {
        return;
    }

    response->set_total_row_count(table_info.total_row_num_);

    for (auto& partition_stat : table_info.partitions_stat_) {
        ::milvus::grpc::PartitionStat* grpc_partiton_stat = response->mutable_partitions_stat()->Add();
        ConstructPartitionStat(partition_stat, grpc_partiton_stat);
    }
}

}  // namespace

GrpcRequestHandler::GrpcRequestHandler(const std::shared_ptr<opentracing::Tracer>& tracer)
    : tracer_(tracer), random_num_generator_() {
    std::random_device random_device;
    random_num_generator_.seed(random_device());
}

void
GrpcRequestHandler::OnPostRecvInitialMetaData(
    ::grpc::experimental::ServerRpcInfo* server_rpc_info,
    ::grpc::experimental::InterceptorBatchMethods* interceptor_batch_methods) {
    std::unordered_map<std::string, std::string> text_map;
    auto* metadata_map = interceptor_batch_methods->GetRecvInitialMetadata();
    auto context_kv = metadata_map->find(tracing::TracerUtil::GetTraceContextHeaderName());
    if (context_kv != metadata_map->end()) {
        text_map[std::string(context_kv->first.data(), context_kv->first.length())] =
            std::string(context_kv->second.data(), context_kv->second.length());
    }
    // test debug mode
    //    if (std::string(server_rpc_info->method()).find("Search") != std::string::npos) {
    //        text_map["demo-debug-id"] = "debug-id";
    //    }

    tracing::TextMapCarrier carrier{text_map};
    auto span_context_maybe = tracer_->Extract(carrier);
    if (!span_context_maybe) {
        std::cerr << span_context_maybe.error().message() << std::endl;
        return;
    }
    auto span = tracer_->StartSpan(server_rpc_info->method(), {opentracing::ChildOf(span_context_maybe->get())});
    auto server_context = server_rpc_info->server_context();
    auto client_metadata = server_context->client_metadata();
    // TODO: request id
    std::string request_id;
    auto request_id_kv = client_metadata.find("request_id");
    if (request_id_kv != client_metadata.end()) {
        request_id = request_id_kv->second.data();
    } else {
        request_id = std::to_string(random_id()) + std::to_string(random_id());
    }
    auto trace_context = std::make_shared<tracing::TraceContext>(span);
    auto context = std::make_shared<Context>(request_id);
    context->SetTraceContext(trace_context);
    SetContext(server_rpc_info->server_context(), context);
}

void
GrpcRequestHandler::OnPreSendMessage(::grpc::experimental::ServerRpcInfo* server_rpc_info,
                                     ::grpc::experimental::InterceptorBatchMethods* interceptor_batch_methods) {
    std::lock_guard<std::mutex> lock(context_map_mutex_);
    context_map_[server_rpc_info->server_context()]->GetTraceContext()->GetSpan()->Finish();
    auto search = context_map_.find(server_rpc_info->server_context());
    if (search != context_map_.end()) {
        context_map_.erase(search);
    }
}

const std::shared_ptr<Context>&
GrpcRequestHandler::GetContext(::grpc::ServerContext* server_context) {
    std::lock_guard<std::mutex> lock(context_map_mutex_);
    return context_map_[server_context];
}

void
GrpcRequestHandler::SetContext(::grpc::ServerContext* server_context, const std::shared_ptr<Context>& context) {
    std::lock_guard<std::mutex> lock(context_map_mutex_);
    context_map_[server_context] = context;
}

uint64_t
GrpcRequestHandler::random_id() const {
    std::lock_guard<std::mutex> lock(random_mutex_);
    auto value = random_num_generator_();
    while (value == 0) {
        value = random_num_generator_();
    }
    return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

::grpc::Status
GrpcRequestHandler::CreateTable(::grpc::ServerContext* context, const ::milvus::grpc::TableSchema* request,
                                ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.CreateTable(context_map_[context], request->table_name(), request->dimension(),
                                                 request->index_file_size(), request->metric_type());
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::HasTable(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                             ::milvus::grpc::BoolReply* response) {
    CHECK_NULLPTR_RETURN(request);

    bool has_table = false;

    Status status = request_handler_.HasTable(context_map_[context], request->table_name(), has_table);
    response->set_bool_reply(has_table);
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::DropTable(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                              ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.DropTable(context_map_[context], request->table_name());

    SET_RESPONSE(response, status, context);
    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::CreateIndex(::grpc::ServerContext* context, const ::milvus::grpc::IndexParam* request,
                                ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    milvus::json json_params;
    for (int i = 0; i < request->extra_params_size(); i++) {
        const ::milvus::grpc::KeyValuePair& extra = request->extra_params(i);
        if (extra.key() == EXTRA_PARAM_KEY) {
            json_params = json::parse(extra.value());
        }
    }

    Status status =
        request_handler_.CreateIndex(context_map_[context], request->table_name(), request->index_type(), json_params);

    SET_RESPONSE(response, status, context);
    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::Insert(::grpc::ServerContext* context, const ::milvus::grpc::InsertParam* request,
                           ::milvus::grpc::VectorIds* response) {
    CHECK_NULLPTR_RETURN(request);

    // step 1: copy vector data
    engine::VectorsData vectors;
    CopyRowRecords(request->row_record_array(), request->row_id_array(), vectors);

    // step 2: insert vectors
    Status status =
        request_handler_.Insert(context_map_[context], request->table_name(), vectors, request->partition_tag());

    // step 3: return id array
    response->mutable_vector_id_array()->Resize(static_cast<int>(vectors.id_array_.size()), 0);
    memcpy(response->mutable_vector_id_array()->mutable_data(), vectors.id_array_.data(),
           vectors.id_array_.size() * sizeof(int64_t));

    SET_RESPONSE(response->mutable_status(), status, context);
    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::GetVectorByID(::grpc::ServerContext* context, const ::milvus::grpc::VectorIdentity* request,
                                  ::milvus::grpc::VectorData* response) {
    CHECK_NULLPTR_RETURN(request);

    std::vector<int64_t> vector_ids = {request->id()};
    engine::VectorsData vectors;
    Status status = request_handler_.GetVectorByID(context_map_[context], request->table_name(), vector_ids, vectors);

    if (!vectors.float_data_.empty()) {
        response->mutable_vector_data()->mutable_float_data()->Resize(vectors.float_data_.size(), 0);
        memcpy(response->mutable_vector_data()->mutable_float_data()->mutable_data(), vectors.float_data_.data(),
               vectors.float_data_.size() * sizeof(float));
    } else if (!vectors.binary_data_.empty()) {
        response->mutable_vector_data()->mutable_binary_data()->resize(vectors.binary_data_.size());
        memcpy(response->mutable_vector_data()->mutable_binary_data()->data(), vectors.binary_data_.data(),
               vectors.binary_data_.size() * sizeof(uint8_t));
    }
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::GetVectorIDs(::grpc::ServerContext* context, const ::milvus::grpc::GetVectorIDsParam* request,
                                 ::milvus::grpc::VectorIds* response) {
    CHECK_NULLPTR_RETURN(request);

    std::vector<int64_t> vector_ids;
    Status status = request_handler_.GetVectorIDs(context_map_[context], request->table_name(), request->segment_name(),
                                                  vector_ids);

    if (!vector_ids.empty()) {
        response->mutable_vector_id_array()->Resize(vector_ids.size(), -1);
        memcpy(response->mutable_vector_id_array()->mutable_data(), vector_ids.data(),
               vector_ids.size() * sizeof(int64_t));
    }
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::Search(::grpc::ServerContext* context, const ::milvus::grpc::SearchParam* request,
                           ::milvus::grpc::TopKQueryResult* response) {
    CHECK_NULLPTR_RETURN(request);

    // step 1: copy vector data
    engine::VectorsData vectors;
    CopyRowRecords(request->query_record_array(), google::protobuf::RepeatedField<google::protobuf::int64>(), vectors);

    // step 2: partition tags
    std::vector<std::string> partitions;
    for (auto& partition : request->partition_tag_array()) {
        partitions.emplace_back(partition);
    }

    // step 3: parse extra parameters
    milvus::json json_params;
    for (int i = 0; i < request->extra_params_size(); i++) {
        const ::milvus::grpc::KeyValuePair& extra = request->extra_params(i);
        if (extra.key() == EXTRA_PARAM_KEY) {
            json_params = json::parse(extra.value());
        }
    }

    // step 4: search vectors
    std::vector<std::string> file_ids;
    TopKQueryResult result;
    fiu_do_on("GrpcRequestHandler.Search.not_empty_file_ids", file_ids.emplace_back("test_file_id"));
    Status status = request_handler_.Search(context_map_[context], request->table_name(), vectors, request->topk(),
                                            json_params, partitions, file_ids, result);

    // step 5: construct and return result
    ConstructResults(result, response);

    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::SearchByID(::grpc::ServerContext* context, const ::milvus::grpc::SearchByIDParam* request,
                               ::milvus::grpc::TopKQueryResult* response) {
    CHECK_NULLPTR_RETURN(request);

    // step 1: partition tags
    std::vector<std::string> partitions;
    for (auto& partition : request->partition_tag_array()) {
        partitions.emplace_back(partition);
    }

    // step 2: parse extra parameters
    milvus::json json_params;
    for (int i = 0; i < request->extra_params_size(); i++) {
        const ::milvus::grpc::KeyValuePair& extra = request->extra_params(i);
        if (extra.key() == EXTRA_PARAM_KEY) {
            json_params = json::parse(extra.value());
        }
    }

    // step 3: search vectors
    TopKQueryResult result;
    Status status = request_handler_.SearchByID(context_map_[context], request->table_name(), request->id(),
                                                request->topk(), json_params, partitions, result);

    // step 4: construct and return result
    ConstructResults(result, response);

    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::SearchInFiles(::grpc::ServerContext* context, const ::milvus::grpc::SearchInFilesParam* request,
                                  ::milvus::grpc::TopKQueryResult* response) {
    CHECK_NULLPTR_RETURN(request);

    auto* search_request = &request->search_param();

    // step 1: copy vector data
    engine::VectorsData vectors;
    CopyRowRecords(search_request->query_record_array(), google::protobuf::RepeatedField<google::protobuf::int64>(),
                   vectors);

    // step 2: copy file id array
    std::vector<std::string> file_ids;
    for (auto& file_id : request->file_id_array()) {
        file_ids.emplace_back(file_id);
    }

    // step 3: partition tags
    std::vector<std::string> partitions;
    for (auto& partition : search_request->partition_tag_array()) {
        partitions.emplace_back(partition);
    }

    // step 4: parse extra parameters
    milvus::json json_params;
    for (int i = 0; i < search_request->extra_params_size(); i++) {
        const ::milvus::grpc::KeyValuePair& extra = search_request->extra_params(i);
        if (extra.key() == EXTRA_PARAM_KEY) {
            json_params = json::parse(extra.value());
        }
    }

    // step 5: search vectors
    TopKQueryResult result;
    Status status = request_handler_.Search(context_map_[context], search_request->table_name(), vectors,
                                            search_request->topk(), json_params, partitions, file_ids, result);

    // step 6: construct and return result
    ConstructResults(result, response);

    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::DescribeTable(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                                  ::milvus::grpc::TableSchema* response) {
    CHECK_NULLPTR_RETURN(request);

    TableSchema table_schema;
    Status status = request_handler_.DescribeTable(context_map_[context], request->table_name(), table_schema);
    response->set_table_name(table_schema.table_name_);
    response->set_dimension(table_schema.dimension_);
    response->set_index_file_size(table_schema.index_file_size_);
    response->set_metric_type(table_schema.metric_type_);

    SET_RESPONSE(response->mutable_status(), status, context);
    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::CountTable(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                               ::milvus::grpc::TableRowCount* response) {
    CHECK_NULLPTR_RETURN(request);

    int64_t row_count = 0;
    Status status = request_handler_.CountTable(context_map_[context], request->table_name(), row_count);
    response->set_table_row_count(row_count);
    SET_RESPONSE(response->mutable_status(), status, context);
    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::ShowTables(::grpc::ServerContext* context, const ::milvus::grpc::Command* request,
                               ::milvus::grpc::TableNameList* response) {
    CHECK_NULLPTR_RETURN(request);

    std::vector<std::string> tables;
    Status status = request_handler_.ShowTables(context_map_[context], tables);
    for (auto& table : tables) {
        response->add_table_names(table);
    }
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::ShowTableInfo(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                                  ::milvus::grpc::TableInfo* response) {
    CHECK_NULLPTR_RETURN(request);

    TableInfo table_info;
    Status status = request_handler_.ShowTableInfo(context_map_[context], request->table_name(), table_info);
    ConstructTableInfo(table_info, response);
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::Cmd(::grpc::ServerContext* context, const ::milvus::grpc::Command* request,
                        ::milvus::grpc::StringReply* response) {
    CHECK_NULLPTR_RETURN(request);

    std::string reply;
    Status status = request_handler_.Cmd(context_map_[context], request->cmd(), reply);
    response->set_string_reply(reply);
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::DeleteByID(::grpc::ServerContext* context, const ::milvus::grpc::DeleteByIDParam* request,
                               ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    // step 1: prepare id array
    std::vector<int64_t> vector_ids;
    for (int i = 0; i < request->id_array_size(); i++) {
        vector_ids.push_back(request->id_array(i));
    }

    // step 2: delete vector
    Status status = request_handler_.DeleteByID(context_map_[context], request->table_name(), vector_ids);
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::PreloadTable(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                                 ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.PreloadTable(context_map_[context], request->table_name());
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::DescribeIndex(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                                  ::milvus::grpc::IndexParam* response) {
    CHECK_NULLPTR_RETURN(request);

    IndexParam param;
    Status status = request_handler_.DescribeIndex(context_map_[context], request->table_name(), param);
    response->set_table_name(param.table_name_);
    response->set_index_type(param.index_type_);
    ::milvus::grpc::KeyValuePair* kv = response->add_extra_params();
    kv->set_key(EXTRA_PARAM_KEY);
    kv->set_value(param.extra_params_);
    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::DropIndex(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                              ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.DropIndex(context_map_[context], request->table_name());
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::CreatePartition(::grpc::ServerContext* context, const ::milvus::grpc::PartitionParam* request,
                                    ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.CreatePartition(context_map_[context], request->table_name(), request->tag());
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::ShowPartitions(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                                   ::milvus::grpc::PartitionList* response) {
    CHECK_NULLPTR_RETURN(request);

    std::vector<PartitionParam> partitions;
    Status status = request_handler_.ShowPartitions(context_map_[context], request->table_name(), partitions);
    for (auto& partition : partitions) {
        response->add_partition_tag_array(partition.tag_);
    }

    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::DropPartition(::grpc::ServerContext* context, const ::milvus::grpc::PartitionParam* request,
                                  ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.DropPartition(context_map_[context], request->table_name(), request->tag());
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::Flush(::grpc::ServerContext* context, const ::milvus::grpc::FlushParam* request,
                          ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    std::vector<std::string> table_names;
    for (int32_t i = 0; i < request->table_name_array().size(); i++) {
        table_names.push_back(request->table_name_array(i));
    }
    Status status = request_handler_.Flush(context_map_[context], table_names);
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::Compact(::grpc::ServerContext* context, const ::milvus::grpc::TableName* request,
                            ::milvus::grpc::Status* response) {
    CHECK_NULLPTR_RETURN(request);

    Status status = request_handler_.Compact(context_map_[context], request->table_name());
    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

/*******************************************New Interface*********************************************/

::grpc::Status
GrpcRequestHandler::CreateCollection(::grpc::ServerContext* context,
                                     const ::milvus::grpc::Mapping* request,
                                     ::milvus::grpc::Status* response) {

    CHECK_NULLPTR_RETURN(request);

    std::vector<std::pair<std::string, engine::meta::hybrid::DataType>> field_types;
    std::vector<std::pair<std::string, uint64_t>> vector_dimensions;
    std::vector<std::pair<std::string, std::string>> field_params;
    for (uint64_t i = 0; i < request->fields_size(); ++i) {
        if(request->fields(i).type().has_vector_param()){
            auto vector_dimension = std::make_pair(request->fields(i).name(), request->fields(i).type().vector_param().dimension());
            vector_dimensions.emplace_back(vector_dimension);
        } else {
            auto type = std::make_pair(request->fields(i).name(),
                                       (engine::meta::hybrid::DataType)request->fields(i).type().data_type());
            field_types.emplace_back(type);
        }
        // Currently only one extra_param
        auto extra_params = std::make_pair(request->fields(i).name(), request->fields(i).extra_params(0).value());
        field_params.emplace_back(extra_params);
    }

    Status status = request_handler_.CreateCollection(context_map_[context],
                                                      request->collection_name(),
                                                      field_types,
                                                      vector_dimensions,
                                                      field_params);

    SET_RESPONSE(response, status, context);

    return ::grpc::Status::OK;
}

::grpc::Status
GrpcRequestHandler::InsertEntity(::grpc::ServerContext* context,
                                 const ::milvus::grpc::HInsertParam* request,
                                 ::milvus::grpc::HEntityIDs* response) {
    CHECK_NULLPTR_RETURN(request);

    std::unordered_map<std::string, std::vector<std::string>> attr_values;
    std::unordered_map<std::string, engine::VectorsData> vector_datas;

    auto size = request->entities().field_names_size();
    auto attr_size = request->entities().attr_records_size();
    for (uint64_t i = 0; i < attr_size; ++i) {
        std::vector<std::string> values;
        auto record_size = request->entities().attr_records(i).value_size();
        values.resize(record_size);
        for (uint64_t j = 0; j < record_size; ++j) {
            values[j] = request->entities().attr_records(i).value(j);
        }
        attr_values.insert(std::make_pair(request->entities().field_names(i), values));
    }

    auto vector_size = request->entities().result_values_size();
    for (uint64_t i = 0; i < vector_size; ++i) {
        engine::VectorsData vectors;
        CopyRowRecords(request->entities().result_values(i).vector_value().value(), request->entity_id_array(), vectors);
        vector_datas.insert(std::make_pair(request->entities().field_names(attr_size + i), vectors));
    }

    std::string collection_name = request->collection_name();
    std::string partition_tag = request->partition_tag();
    Status status = request_handler_.InsertEntity(context_map_[context],
                                                  collection_name,
                                                  partition_tag,
                                                  attr_values,
                                                  vector_datas);

    response->mutable_entity_id_array()->Resize(static_cast<int>(vector_datas.begin()->second.id_array_.size()), 0);
    memcpy(response->mutable_entity_id_array()->mutable_data(), vector_datas.begin()->second.id_array_.data(),
           vector_datas.begin()->second.id_array_.size() * sizeof(int64_t));

    SET_RESPONSE(response->mutable_status(), status, context);
    return ::grpc::Status::OK;
}

void
DeSerialization(const ::milvus::grpc::GeneralQuery& general_query, query::BooleanQueryPtr boolean_clause) {
    if (general_query.has_boolean_query()) {
//        boolean_clause->SetOccur((query::Occur)general_query.boolean_query().occur());

        for (uint64_t i = 0; i < general_query.boolean_query().general_query_size(); ++i) {
            if (general_query.boolean_query().general_query(i).has_boolean_query()) {
                query::BooleanQueryPtr query =
                    std::make_shared<query::BooleanQuery>((query::Occur)(general_query.boolean_query().occur()));
                DeSerialization(general_query.boolean_query().general_query(i), query);
                boolean_clause->AddBooleanQuery(query);
            } else {
                auto leaf_query = std::make_shared<query::LeafQuery>();
                auto query = general_query.boolean_query().general_query(i);
                if (query.has_term_query()) {
                    query::TermQueryPtr term_query = std::make_shared<query::TermQuery>();
                    term_query->field_name = query.term_query().field_name();
                    term_query->boost = query.term_query().boost();
                    term_query->field_value.resize(query.term_query().values_size());
                    for (uint64_t j = 0; j < query.term_query().values_size(); ++j) {
                        term_query->field_value[j] = query.term_query().values(j);
                    }
                    leaf_query->term_query = term_query;
                    boolean_clause->AddLeafQuery(leaf_query);
                }
                if (query.has_range_query()) {
                    query::RangeQueryPtr range_query = std::make_shared<query::RangeQuery>();
                    range_query->field_name = query.range_query().field_name();
                    range_query->boost = query.range_query().boost();
                    range_query->compare_expr.resize(query.range_query().operand_size());
                    for (uint64_t j = 0; j < query.range_query().operand_size(); ++j) {
                        range_query->compare_expr[j].compare_operator =
                            query::CompareOperator(query.range_query().operand(j).operator_());
                        range_query->compare_expr[j].operand = query.range_query().operand(j).operand();
                    }
                    leaf_query->range_query = range_query;
                    boolean_clause->AddLeafQuery(leaf_query);
                }
                if (query.has_vector_query()) {
                    query::VectorQueryPtr vector_query = std::make_shared<query::VectorQuery>();

                    engine::VectorsData vectors;
                    CopyRowRecords(query.vector_query().records(), google::protobuf::RepeatedField<google::protobuf::int64>(),
                                   vectors);

                    vector_query->query_vector.float_data = vectors.float_data_;
                    vector_query->query_vector.binary_data = vectors.binary_data_;

                    vector_query->boost = query.vector_query().query_boost();
                    vector_query->field_name = query.vector_query().field_name();
                    vector_query->topk = query.vector_query().topk();

                    milvus::json json_params;
                    for (int j = 0; j < query.vector_query().extra_params_size(); j++) {
                        const ::milvus::grpc::KeyValuePair& extra = query.vector_query().extra_params(j);
                        if (extra.key() == EXTRA_PARAM_KEY) {
                            json_params = json::parse(extra.value());
                        }
                    }
                    vector_query->extra_params = json_params;
                    leaf_query->vector_query = vector_query;
                    boolean_clause->AddLeafQuery(leaf_query);
                }
            }
        }
    }
}

::grpc::Status
GrpcRequestHandler::HybridSearch(::grpc::ServerContext* context,
                                 const ::milvus::grpc::HSearchParam* request,
                                 ::milvus::grpc::TopKQueryResult* response) {
    CHECK_NULLPTR_RETURN(request);

    context::HybridSearchContextPtr hybrid_search_context = std::make_shared<context::HybridSearchContext>();
    hybrid_search_context->grpc_request_ = *request;

    query::BooleanQueryPtr boolean_query = std::make_shared<query::BooleanQuery>();
    DeSerialization(request->general_query(), boolean_query);

    query::GeneralQueryPtr general_query = std::make_shared<query::GeneralQuery>();
    general_query->bin = std::make_shared<query::BinaryQuery>();
    query::GenBinaryQuery(boolean_query, general_query->bin);

    Status status;

    if (query::BinaryQueryHeight(general_query->bin) == true) {
        status = Status{SERVER_INVALID_BINARY_QUERY, "Generate wrong binary query tree"};
        SET_RESPONSE(response->mutable_status(), status, context);
        return ::grpc::Status::OK;
    }

    hybrid_search_context->general_query_ = general_query;

    std::vector<std::string> partition_list;
    partition_list.resize(request->partition_tag_array_size());
    for (uint64_t i = 0; i < request->partition_tag_array_size(); ++i) {
        partition_list[i] = request->partition_tag_array(i);
    }

    TopKQueryResult result;

    status = request_handler_.HybridSearch(context_map_[context],
                                           hybrid_search_context,
                                           request->collection_name(),
                                           partition_list,
                                           general_query,
                                           result);

    // step 6: construct and return result
    ConstructResults(result, response);

    SET_RESPONSE(response->mutable_status(), status, context);

    return ::grpc::Status::OK;
}

}  // namespace grpc
}  // namespace server
}  // namespace milvus
