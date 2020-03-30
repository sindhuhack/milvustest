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

#include "examples/utils/TimeRecorder.h"
#include "examples/utils/Utils.h"
#include "examples/utils/ThreadPool.h"
#include "examples/qps/src/ClientTest.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {
const char* COLLECTION_NAME = milvus_sdk::Utils::GenCollectionName().c_str();
constexpr int64_t BATCH_ENTITY_COUNT = 100000;
constexpr int64_t ADD_ENTITY_LOOP = 10;
}  // namespace

ClientTest::ClientTest(const std::string& address, const std::string& port)
    : server_ip_(address), server_port_(port) {
}

ClientTest::~ClientTest() {
}

std::shared_ptr<milvus::Connection>
ClientTest::Connect() {
    std::shared_ptr<milvus::Connection> conn;
    milvus::ConnectParam param = {server_ip_, server_port_};
    conn = milvus::Connection::Create();
    milvus::Status stat = conn->Connect(param);
    if (!stat.ok()) {
        std::string msg = "Connect function call status: " + stat.message();
        std::cout << "Connect function call status: " << stat.message() << std::endl;
    }
    return conn;
}

bool
ClientTest::CheckParameters(const TestParameters& parameters) {
    if (parameters.index_type_ != (int64_t)milvus::IndexType::FLAT
        && parameters.index_type_ != (int64_t)milvus::IndexType::IVFFLAT
        && parameters.index_type_ != (int64_t)milvus::IndexType::IVFSQ8
        && parameters.index_type_ != (int64_t)milvus::IndexType::IVFSQ8H) {
        std::cout << "Unsupportted index type: " << parameters.index_type_ << std::endl;
        return false;
    }

    if (parameters.metric_type_ <= 0 || parameters.metric_type_ > (int64_t)milvus::MetricType::SUPERSTRUCTURE) {
        std::cout << "Invalid metric type: " << parameters.metric_type_ << std::endl;
        return false;
    }

    if (parameters.row_count_ <= 0) {
        std::cout << "Invalid row count: " << parameters.row_count_ << std::endl;
        return false;
    }

    if (parameters.concurrency_ <= 0) {
        std::cout << "Invalid concurrency: " << parameters.concurrency_ << std::endl;
        return false;
    }

    if (parameters.query_count_ <= 0) {
        std::cout << "Invalid query count: " << parameters.query_count_ << std::endl;
        return false;
    }

    if (parameters.nq_ <= 0) {
        std::cout << "Invalid query nq: " << parameters.nq_ << std::endl;
        return false;
    }

    if (parameters.topk_ <= 0 || parameters.topk_ > 2048) {
        std::cout << "Invalid query topk: " << parameters.topk_ << std::endl;
        return false;
    }

    if (parameters.nprobe_ <= 0) {
        std::cout << "Invalid query nprobe: " << parameters.nprobe_ << std::endl;
        return false;
    }

    return true;
}

bool
ClientTest::BuildCollection() {
    std::shared_ptr<milvus::Connection> conn = Connect();
    if (conn == nullptr) {
        return false;
    }

    milvus::CollectionParam collection_param = {
        COLLECTION_NAME,
        parameters_.dimensions_,
        parameters_.index_file_size_,
        (milvus::MetricType)parameters_.metric_type_
    };
    auto stat = conn->CreateCollection(collection_param);
    std::cout << "CreateCollection function call status: " << stat.message() << std::endl;
    if (!stat.ok()) {
        return false;
    }

    InsertEntities(conn);

    conn->PreloadCollection(COLLECTION_NAME);
    milvus::Connection::Destroy(conn);
    return true;
}

bool
ClientTest::InsertEntities(std::shared_ptr<milvus::Connection>& conn) {
    int64_t batch_count = parameters_.row_count_ * ADD_ENTITY_LOOP;
    for (int i = 0; i < batch_count; i++) {
        std::vector<milvus::Entity> entity_array;
        std::vector<int64_t> record_ids;
        int64_t begin_index = i * BATCH_ENTITY_COUNT;
        {  // generate vectors
//            milvus_sdk::TimeRecorder rc("Build entities No." + std::to_string(i));
            milvus_sdk::Utils::BuildEntities(begin_index,
                                             begin_index + BATCH_ENTITY_COUNT,
                                             entity_array,
                                             record_ids,
                                             parameters_.dimensions_);
        }

        std::string title = "Insert " + std::to_string(entity_array.size()) + " entities No." + std::to_string(i);
        milvus_sdk::TimeRecorder rc(title);
        milvus::Status stat = conn->Insert(COLLECTION_NAME, "", entity_array, record_ids);
//        std::cout << "InsertEntities function call status: " << stat.message() << std::endl;
//        std::cout << "Returned id array count: " << record_ids.size() << std::endl;

        stat = conn->FlushCollection(COLLECTION_NAME);
    }

    return true;
}

void
ClientTest::CreateIndex() {
    std::shared_ptr<milvus::Connection> conn = Connect();
    if (conn == nullptr) {
        return;
    }

    std::cout << "Wait create index ..." << std::endl;
    JSON json_params = {{"nlist", parameters_.nlist_}};
    milvus::IndexParam index = {COLLECTION_NAME, (milvus::IndexType)parameters_.index_type_, json_params.dump()};
    milvus_sdk::Utils::PrintIndexParam(index);
    milvus::Status stat = conn->CreateIndex(index);
    std::cout << "CreateIndex function call status: " << stat.message() << std::endl;

    conn->PreloadCollection(COLLECTION_NAME);
    milvus::Connection::Destroy(conn);
}

void
ClientTest::DropCollection() {
    std::shared_ptr<milvus::Connection> conn = Connect();
    if (conn == nullptr) {
        return;
    }

    milvus::Status stat = conn->DropCollection(COLLECTION_NAME);
    std::cout << "DropCollection function call status: " << stat.message() << std::endl;

    milvus::Connection::Destroy(conn);
}

void
ClientTest::BuildSearchEntities(std::vector<EntityList>& entity_array) {
    entity_array.clear();
    for (int64_t i = 0; i < parameters_.query_count_; i++) {
        std::vector<milvus::Entity> entities;
        std::vector<int64_t> record_ids;

        int64_t batch_index = i % ADD_ENTITY_LOOP;
        int64_t offset = batch_index * BATCH_ENTITY_COUNT;
        milvus_sdk::Utils::BuildEntities(offset,
                                         offset + parameters_.nq_,
                                         entities,
                                         record_ids,
                                         parameters_.dimensions_);
        entity_array.emplace_back(entities);
    }

//    std::cout << "Build search entities finish" << std::endl;
}

void
ClientTest::Search() {
    std::vector<EntityList> search_entities;
    BuildSearchEntities(search_entities);

    std::list<std::future<milvus::TopKQueryResult>> query_thread_results;
    milvus_sdk::ThreadPool query_thread_pool(parameters_.concurrency_, parameters_.concurrency_ * 2);

    auto start = std::chrono::system_clock::now();
    // multi-threads query
    for (int32_t i = 0; i < parameters_.query_count_; i++) {
        query_thread_results.push_back(query_thread_pool.enqueue(&ClientTest::SearchWorker,
                                                                 this,
                                                                 search_entities[i]));
    }

    // wait all query return
    for (auto& iter : query_thread_results) {
        iter.wait();
    }

    // print result
    int64_t index = 0;
    for (auto& iter : query_thread_results) {
        milvus::TopKQueryResult result = iter.get();
        CheckSearchResult(index, result);
        PrintSearchResult(index, result);
        index++;
    }

    // calculate qps
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    int64_t span = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
    double sec = (double)span / 1000.0;
    double tps_s = parameters_.query_count_ / sec;
    int64_t tps = (int64_t)tps_s;
    int64_t qps = tps * parameters_.nq_;
    std::cout << "TPS = " << tps << " \tQPS = " << qps << std::endl;

    // print search detail statistics
    JSON search_stats = JSON();
    search_stats["index"] = milvus_sdk::Utils::IndexTypeName((milvus::IndexType)parameters_.index_type_);
    search_stats["index_file_size"] = parameters_.index_file_size_;
    search_stats["nlist"] = parameters_.nlist_;
    search_stats["metric"] = milvus_sdk::Utils::MetricTypeName((milvus::MetricType)parameters_.metric_type_);
    search_stats["dimension"] = parameters_.dimensions_;
    search_stats["row_count"] = parameters_.row_count_ * BATCH_ENTITY_COUNT * ADD_ENTITY_LOOP;
    search_stats["concurrency"] = parameters_.concurrency_;
    search_stats["query_count"] = parameters_.query_count_;
    search_stats["nq"] = parameters_.nq_;
    search_stats["topk"] = parameters_.topk_;
    search_stats["nprobe"] = parameters_.nprobe_;
    search_stats["qps"] = qps;
    search_stats["tps"] = tps;
    std::cout << search_stats.dump() << std::endl;
}

milvus::TopKQueryResult
ClientTest::SearchWorker(EntityList& entities) {
    milvus::TopKQueryResult res;

    std::shared_ptr<milvus::Connection> conn;
    milvus::ConnectParam param = {server_ip_, server_port_};
    conn = milvus::Connection::Create();
    milvus::Status stat = conn->Connect(param);
    if (!stat.ok()) {
        milvus::Connection::Destroy(conn);
        std::string msg = "Connect function call status: " + stat.message();
        std::cout << msg << std::endl;
        return res;
    }

    JSON json_params = {{"nprobe", parameters_.nprobe_}};
    std::vector<std::string> partition_tags;
    stat = conn->Search(COLLECTION_NAME,
                        partition_tags,
                        entities,
                        parameters_.topk_,
                        json_params.dump(),
                        res);
    if (!stat.ok()) {
        milvus::Connection::Destroy(conn);
        std::string msg = "Search function call status: " + stat.message();
        std::cout << msg << std::endl;
        return res;
    }

    milvus::Connection::Destroy(conn);
    return res;
}

void
ClientTest::PrintSearchResult(int64_t batch_num, const milvus::TopKQueryResult& result) {
    if (!parameters_.print_result_) {
        return;
    }

    std::cout << "No." << batch_num << " query result:" << std::endl;
    for (size_t i = 0; i < result.size(); i++) {
        std::cout << "\tNQ_" << i;
        const milvus::QueryResult& one_result = result[i];
        size_t topk = one_result.ids.size();
        for (size_t j = 0; j < topk; j++) {
            std::cout << "\t[" << one_result.ids[j] << ", " << one_result.distances[j] << "]";
        }
        std::cout << std::endl;
    }
}

void
ClientTest::CheckSearchResult(int64_t batch_num, const milvus::TopKQueryResult& result) {
    if (result.empty()) {
        std::cout << "ERROR! No." << batch_num << " query return empty result" << std::endl;
        return;
    }
    for (auto& res : result) {
        if (res.ids.empty()) {
            std::cout << "ERROR! No." << batch_num << " query return empty id" << std::endl;
            return;
        }
    }
}

void
ClientTest::Test(const TestParameters& parameters) {
    if (!CheckParameters(parameters)) {
        return;
    }
    parameters_ = parameters;
    if (!BuildCollection()) {
        return;
    }

    CreateIndex();

    // search with index
    std::cout << "Search with index: " << milvus_sdk::Utils::IndexTypeName((milvus::IndexType)parameters.index_type_)
              << std::endl;
    Search();

    DropCollection();
}
