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

#include "db/transcript/TranscriptProxy.h"
#include "db/transcript/ScriptRecorder.h"
#include "db/transcript/ScriptReplay.h"
#include "utils/CommonUtil.h"
#include "utils/Exception.h"

#include <experimental/filesystem>

namespace milvus {
namespace engine {

TranscriptProxy::TranscriptProxy(const DBPtr& db, const DBOptions& options) : DBProxy(db, options) {
    // db must implemented
    if (db == nullptr) {
        throw Exception(DB_ERROR, "null pointer");
    }
}

Status
TranscriptProxy::Start() {
    // let service start
    auto status = db_->Start();
    if (!status.ok()) {
        return status;
    }

    // replay script in necessary
    if (!options_.replay_script_path_.empty()) {
        ScriptReplay replay;
        STATUS_CHECK(replay.Replay(db_, options_.replay_script_path_));
    }

    // prepare for transcript
    std::experimental::filesystem::path db_path(options_.meta_.path_);
    auto transcript_path = db_path.parent_path();
    transcript_path /= "transcript";
    std::string path = transcript_path.c_str();
    status = CommonUtil::CreateDirectory(path);
    if (!status.ok()) {
        std::cerr << "Error: Failed to create transcript path: " << path << std::endl;
        kill(0, SIGUSR1);
    }

    ScriptRecorder& recorder = ScriptRecorder::GetInstance();
    recorder.SetScriptPath(path);

    return Status::OK();
}

Status
TranscriptProxy::Stop() {
    return db_->Stop();
}

Status
TranscriptProxy::CreateCollection(const snapshot::CreateCollectionContext& context) {
    ScriptRecorder::GetInstance().CreateCollection(context);
    return db_->CreateCollection(context);
}

Status
TranscriptProxy::DropCollection(const std::string& collection_name) {
    ScriptRecorder::GetInstance().DropCollection(collection_name);
    return db_->DropCollection(collection_name);
}

Status
TranscriptProxy::HasCollection(const std::string& collection_name, bool& has_or_not) {
    ScriptRecorder::GetInstance().HasCollection(collection_name, has_or_not);
    return db_->HasCollection(collection_name, has_or_not);
}

Status
TranscriptProxy::ListCollections(std::vector<std::string>& names) {
    ScriptRecorder::GetInstance().ListCollections(names);
    return db_->ListCollections(names);
}

Status
TranscriptProxy::GetCollectionInfo(const std::string& collection_name, snapshot::CollectionPtr& collection,
                                   snapshot::FieldElementMappings& fields_schema) {
    ScriptRecorder::GetInstance().GetCollectionInfo(collection_name, collection, fields_schema);
    return db_->GetCollectionInfo(collection_name, collection, fields_schema);
}

Status
TranscriptProxy::GetCollectionStats(const std::string& collection_name, milvus::json& collection_stats) {
    ScriptRecorder::GetInstance().GetCollectionStats(collection_name, collection_stats);
    return db_->GetCollectionStats(collection_name, collection_stats);
}

Status
TranscriptProxy::CountEntities(const std::string& collection_name, int64_t& row_count) {
    ScriptRecorder::GetInstance().CountEntities(collection_name, row_count);
    return db_->CountEntities(collection_name, row_count);
}

Status
TranscriptProxy::CreatePartition(const std::string& collection_name, const std::string& partition_name) {
    ScriptRecorder::GetInstance().CreatePartition(collection_name, partition_name);
    return db_->CreatePartition(collection_name, partition_name);
}

Status
TranscriptProxy::DropPartition(const std::string& collection_name, const std::string& partition_name) {
    ScriptRecorder::GetInstance().DropPartition(collection_name, partition_name);
    return db_->DropPartition(collection_name, partition_name);
}

Status
TranscriptProxy::HasPartition(const std::string& collection_name, const std::string& partition_tag, bool& exist) {
    ScriptRecorder::GetInstance().HasPartition(collection_name, partition_tag, exist);
    return db_->HasPartition(collection_name, partition_tag, exist);
}

Status
TranscriptProxy::ListPartitions(const std::string& collection_name, std::vector<std::string>& partition_names) {
    ScriptRecorder::GetInstance().ListPartitions(collection_name, partition_names);
    return db_->ListPartitions(collection_name, partition_names);
}

Status
TranscriptProxy::CreateIndex(const server::ContextPtr& context, const std::string& collection_name,
                             const std::string& field_name, const CollectionIndex& index) {
    ScriptRecorder::GetInstance().CreateIndex(context, collection_name, field_name, index);
    return db_->CreateIndex(context, collection_name, field_name, index);
}

Status
TranscriptProxy::DropIndex(const std::string& collection_name, const std::string& field_name) {
    ScriptRecorder::GetInstance().DropIndex(collection_name, field_name);
    return db_->DropIndex(collection_name, field_name);
}

Status
TranscriptProxy::DescribeIndex(const std::string& collection_name, const std::string& field_name,
                               CollectionIndex& index) {
    ScriptRecorder::GetInstance().DescribeIndex(collection_name, field_name, index);
    return db_->DescribeIndex(collection_name, field_name, index);
}

Status
TranscriptProxy::Insert(const std::string& collection_name, const std::string& partition_name, DataChunkPtr& data_chunk,
                        idx_t op_id) {
    ScriptRecorder::GetInstance().Insert(collection_name, partition_name, data_chunk, op_id);
    return db_->Insert(collection_name, partition_name, data_chunk);
}

Status
TranscriptProxy::GetEntityByID(const std::string& collection_name, const IDNumbers& id_array,
                               const std::vector<std::string>& field_names, std::vector<bool>& valid_row,
                               DataChunkPtr& data_chunk) {
    ScriptRecorder::GetInstance().GetEntityByID(collection_name, id_array, field_names, valid_row, data_chunk);
    return db_->GetEntityByID(collection_name, id_array, field_names, valid_row, data_chunk);
}

Status
TranscriptProxy::DeleteEntityByID(const std::string& collection_name, const engine::IDNumbers& entity_ids,
                                  idx_t op_id) {
    ScriptRecorder::GetInstance().DeleteEntityByID(collection_name, entity_ids, op_id);
    return db_->DeleteEntityByID(collection_name, entity_ids);
}

Status
TranscriptProxy::ListIDInSegment(const std::string& collection_name, int64_t segment_id, IDNumbers& entity_ids) {
    ScriptRecorder::GetInstance().ListIDInSegment(collection_name, segment_id, entity_ids);
    return db_->ListIDInSegment(collection_name, segment_id, entity_ids);
}

Status
TranscriptProxy::Query(const server::ContextPtr& context, const query::QueryPtr& query_ptr,
                       engine::QueryResultPtr& result) {
    ScriptRecorder::GetInstance().Query(context, query_ptr, result);
    return db_->Query(context, query_ptr, result);
}

Status
TranscriptProxy::LoadCollection(const server::ContextPtr& context, const std::string& collection_name,
                                const std::vector<std::string>& field_names, bool force) {
    ScriptRecorder::GetInstance().LoadCollection(context, collection_name, field_names, force);
    return db_->LoadCollection(context, collection_name, field_names, force);
}

Status
TranscriptProxy::Flush(const std::string& collection_name) {
    ScriptRecorder::GetInstance().Flush(collection_name);
    return db_->Flush(collection_name);
}

Status
TranscriptProxy::Flush() {
    ScriptRecorder::GetInstance().Flush();
    return db_->Flush();
}

Status
TranscriptProxy::Compact(const server::ContextPtr& context, const std::string& collection_name, double threshold) {
    ScriptRecorder::GetInstance().Compact(context, collection_name, threshold);
    return db_->Compact(context, collection_name, threshold);
}

}  // namespace engine
}  // namespace milvus
