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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <MilvusApi.h>

class ClientTest {
 public:
    ClientTest(const std::string&, const std::string&);
    ~ClientTest();

    void
    Test();

 private:
    void
    ListCollections(std::vector<std::string>&);

    void
    CreateCollection();

    void
    CreateMultiVecCollection();

    void
    InsertMultiEntities();

    void
    CreatePartition();

    void
    GetCollectionInfo();

    void
    ListPartitions();

    void
    InsertEntities();

    void
    CountEntities(int64_t& count);

    void
    Flush();

    void
    GetCollectionStats();

    void
    BuildVectors(int64_t nq, int64_t dimension);

    void
    GetEntityByID(const std::vector<int64_t>& id_array);

    void
    SearchEntities();

    void
    CreateIndex(const std::string&, int64_t);

    void
    LoadCollection(const std::string&);

    void
    CompactCollection(const std::string&);

    void
    DeleteByIds(const std::vector<int64_t>& id_array);

    void
    DropPartition();

    void
    DropIndex(const std::string& collection_name, const std::string& field_name, const std::string& index_name);

    void
    DropCollection(const std::string&);

 private:
    std::shared_ptr<milvus::Connection> conn_;
    std::vector<std::pair<int64_t, milvus::VectorData>> search_entity_array_;
    std::vector<int64_t> search_id_array_;
};