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

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "config/handler/CacheConfigHandler.h"
#include "db/insert/SSMemTableFile.h"
#include "db/insert/VectorSource.h"
#include "utils/Status.h"

namespace milvus {
namespace engine {

class SSMemTable : public server::CacheConfigHandler {
 public:
    using SSMemTableFileList = std::vector<SSMemTableFilePtr>;

    SSMemTable(const std::string& collection_id, const DBOptions& options);

    Status
    Add(const VectorSourcePtr& source);

    Status
    AddEntities(const VectorSourcePtr& source);

    Status
    Delete(segment::doc_id_t doc_id);

    Status
    Delete(const std::vector<segment::doc_id_t>& doc_ids);

    void
    GetCurrentSSMemTableFile(SSMemTableFilePtr& mem_table_file);

    size_t
    GetTableFileCount();

    Status
    Serialize(uint64_t wal_lsn, bool apply_delete = true);

    bool
    Empty();

    const std::string&
    GetTableId() const;

    size_t
    GetCurrentMem();

    uint64_t
    GetLSN();

    void
    SetLSN(uint64_t lsn);

 protected:
    void
    OnCacheInsertDataChanged(bool value) override;

 private:
    Status
    ApplyDeletes();

 private:
    const std::string collection_id_;

    SSMemTableFileList mem_table_file_list_;

    DBOptions options_;

    std::mutex mutex_;

    std::set<segment::doc_id_t> doc_ids_to_delete_;

    std::atomic<uint64_t> lsn_;
};  // SSMemTable

using SSMemTablePtr = std::shared_ptr<SSMemTable>;

}  // namespace engine
}  // namespace milvus
