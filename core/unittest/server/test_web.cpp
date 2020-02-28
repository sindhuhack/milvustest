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

#include <gtest/gtest.h>
#include <opentracing/mocktracer/tracer.h>

#include <boost/filesystem.hpp>
#include <thread>
#include <random>
#include <unistd.h>

#include <oatpp/web/client/HttpRequestExecutor.hpp>
#include <oatpp/network/client/SimpleTCPConnectionProvider.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/web/client/ApiClient.hpp>

#include "wrapper/VecIndex.h"

#include "server/Server.h"
#include "server/delivery/RequestScheduler.h"
#include "server/delivery/request/BaseRequest.h"
#include "server/delivery/RequestHandler.h"
#include "src/version.h"

#include "server/web_impl/handler/WebRequestHandler.h"
#include "server/web_impl/dto/TableDto.hpp"
#include "server/web_impl/dto/StatusDto.hpp"
#include "server/web_impl/dto/VectorDto.hpp"
#include "server/web_impl/dto/IndexDto.hpp"
#include "server/web_impl/component/AppComponent.hpp"
#include "server/web_impl/controller/WebController.hpp"
#include "server/web_impl/Types.h"
#include "server/web_impl/WebServer.h"

#include "scheduler/ResourceFactory.h"
#include "scheduler/SchedInst.h"
#include "server/Config.h"
#include "server/DBWrapper.h"
#include "utils/CommonUtil.h"

#include "unittest/server/utils.h"

static const char* TABLE_NAME = "test_web";
static constexpr int64_t TABLE_DIM = 256;
static constexpr int64_t INDEX_FILE_SIZE = 1024;
static constexpr int64_t VECTOR_COUNT = 1000;
static constexpr int64_t INSERT_LOOP = 10;
constexpr int64_t SECONDS_EACH_HOUR = 3600;

using OStatus = oatpp::web::protocol::http::Status;
using OString = milvus::server::web::OString;
using OQueryParams = milvus::server::web::OQueryParams;
using OChunkedBuffer = oatpp::data::stream::ChunkedBuffer;
using OOutputStream = oatpp::data::stream::BufferOutputStream;
using OFloat32 = milvus::server::web::OFloat32;
using OInt64 = milvus::server::web::OInt64;
template<class T>
using OList = milvus::server::web::OList<T>;

using StatusCode = milvus::server::web::StatusCode;

namespace {

OList<OFloat32>::ObjectWrapper
RandomRowRecordDto(int64_t dim) {
    auto row_record_dto = OList<OFloat32>::createShared();

    std::default_random_engine e;
    std::uniform_real_distribution<float> u(0, 1);
    for (size_t i = 0; i < dim; i++) {
        row_record_dto->pushBack(u(e));
    }

    return row_record_dto;
}

OList<OInt64>::ObjectWrapper
RandomBinRowRecordDto(int64_t dim) {
    auto row_record_dto = OList<OInt64>::createShared();

    std::default_random_engine e;
    std::uniform_real_distribution<float> u(0, 255);
    for (size_t i = 0; i < dim / 8; i++) {
        row_record_dto->pushBack(static_cast<int64_t>(u(e)));
    }

    return row_record_dto;
}

OList<OList<OFloat32>::ObjectWrapper>::ObjectWrapper
RandomRecordsDto(int64_t dim, int64_t num) {
    auto records_dto = OList<OList<OFloat32>::ObjectWrapper>::createShared();
    for (size_t i = 0; i < num; i++) {
        records_dto->pushBack(RandomRowRecordDto(dim));
    }

    return records_dto;
}

OList<OList<OInt64>::ObjectWrapper>::ObjectWrapper
RandomBinRecordsDto(int64_t dim, int64_t num) {
    auto records_dto = OList<OList<OInt64>::ObjectWrapper>::createShared();
    for (size_t i = 0; i < num; i++) {
        records_dto->pushBack(RandomBinRowRecordDto(dim));
    }

    return records_dto;
}

nlohmann::json
RandomRawRecordJson(int64_t dim) {
    nlohmann::json json;

    std::default_random_engine e;
    std::uniform_real_distribution<float> u(0, 1);
    for (size_t i = 0; i < dim; i++) {
        json.push_back(u(e));
    }

    return json;
}

nlohmann::json
RandomRecordsJson(int64_t dim, int64_t num) {
    nlohmann::json json;
    for (size_t i = 0; i < num; i++) {
        json.push_back(RandomRawRecordJson(dim));
    }

    return json;
}

nlohmann::json
RandomRawBinRecordJson(int64_t dim) {
    nlohmann::json json;

    std::default_random_engine e;
    std::uniform_real_distribution<float> u(0, 255);
    for (size_t i = 0; i < dim / 8; i++) {
        json.push_back(static_cast<uint8_t>(u(e)));
    }

    return json;
}

nlohmann::json
RandomBinRecordsJson(int64_t dim, int64_t num) {
    nlohmann::json json;
    for (size_t i = 0; i < num; i++) {
        json.push_back(RandomRawBinRecordJson(dim));
    }

    return json;
}

std::string
RandomName() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
    std::uniform_int_distribution<unsigned> u(0, 1000000);

    size_t name_len = u(e) % 16 + 3;

    char* name = new char[name_len + 1];
    name[name_len] = '\0';

    for (size_t i = 0; i < name_len; i++) {
        unsigned random_i = u(e);
        char remainder = static_cast<char>(random_i % 26);
        name[i] = (random_i % 2 == 0) ? 'A' + remainder : 'a' + remainder;
    }

    std::string random_name(name);

    delete[] name;

    return random_name;
}

} // namespace

namespace {

class WebHandlerTest : public testing::Test {
 protected:
    static void
    SetUpTestCase() {
        auto res_mgr = milvus::scheduler::ResMgrInst::GetInstance();
        res_mgr->Clear();
        res_mgr->Add(milvus::scheduler::ResourceFactory::Create("disk", "DISK", 0, false));
        res_mgr->Add(milvus::scheduler::ResourceFactory::Create("cpu", "CPU", 0));
        res_mgr->Add(milvus::scheduler::ResourceFactory::Create("gtx1660", "GPU", 0));

        auto default_conn = milvus::scheduler::Connection("IO", 500.0);
        auto PCIE = milvus::scheduler::Connection("IO", 11000.0);
        res_mgr->Connect("disk", "cpu", default_conn);
        res_mgr->Connect("cpu", "gtx1660", PCIE);
        res_mgr->Start();
        milvus::scheduler::SchedInst::GetInstance()->Start();
        milvus::scheduler::JobMgrInst::GetInstance()->Start();

        milvus::engine::DBOptions opt;

        milvus::server::Config::GetInstance().SetDBConfigBackendUrl("sqlite://:@:/");
        boost::filesystem::remove_all("/tmp/milvus_web_handler_test");
        milvus::server::Config::GetInstance().SetStorageConfigPrimaryPath("/tmp/milvus_web_handler_test");
        milvus::server::Config::GetInstance().SetStorageConfigSecondaryPath("");
        milvus::server::Config::GetInstance().SetDBConfigArchiveDiskThreshold("");
        milvus::server::Config::GetInstance().SetDBConfigArchiveDaysThreshold("");
        milvus::server::Config::GetInstance().SetCacheConfigCacheInsertData("");
        milvus::server::Config::GetInstance().SetEngineConfigOmpThreadNum("");

        milvus::server::DBWrapper::GetInstance().StartService();
    }

    void
    SetUp() override {
        handler = std::make_shared<milvus::server::web::WebRequestHandler>();
    }

    void
    TearDown() override {
    }

    static void
    TearDownTestCase() {
        milvus::server::DBWrapper::GetInstance().StopService();
        milvus::scheduler::JobMgrInst::GetInstance()->Stop();
        milvus::scheduler::ResMgrInst::GetInstance()->Stop();
        milvus::scheduler::SchedInst::GetInstance()->Stop();
        boost::filesystem::remove_all("/tmp/milvus_web_handler_test");
    }

 protected:
    void
    GenTable(const std::string& table_name, int64_t dim, int64_t index_size, const std::string& metric) {
        auto table_dto = milvus::server::web::TableRequestDto::createShared();
        table_dto->table_name = table_name.c_str();
        table_dto->dimension = dim;
        table_dto->index_file_size = index_size;
        table_dto->metric_type = metric.c_str();

        auto status_dto = handler->CreateTable(table_dto);
    }

 protected:
    std::shared_ptr<milvus::server::web::WebRequestHandler> handler;
    std::shared_ptr<milvus::server::Context> dummy_context;
};

} // namespace

TEST_F(WebHandlerTest, TABLE) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());
    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();

    auto table_dto = milvus::server::web::TableRequestDto::createShared();
    table_dto->table_name = table_name;
    table_dto->dimension = TABLE_DIM + 100000;
    table_dto->index_file_size = INDEX_FILE_SIZE;
    table_dto->metric_type = "L2";

    // invalid dimension
    auto status_dto = handler->CreateTable(table_dto);
    ASSERT_EQ(StatusCode::ILLEGAL_DIMENSION, status_dto->code->getValue());

    // invalid index file size
    table_dto->dimension = TABLE_DIM;
    table_dto->index_file_size = -1;
    status_dto = handler->CreateTable(table_dto);
    ASSERT_EQ(StatusCode::ILLEGAL_ARGUMENT, status_dto->code->getValue());

    // invalid metric type
    table_dto->index_file_size = INDEX_FILE_SIZE;
    table_dto->metric_type = "L1";
    status_dto = handler->CreateTable(table_dto);
    ASSERT_EQ(StatusCode::ILLEGAL_METRIC_TYPE, status_dto->code->getValue());

    // create table successfully
    table_dto->metric_type = "L2";
    status_dto = handler->CreateTable(table_dto);
    ASSERT_EQ(0, status_dto->code->getValue());

    sleep(3);

    status_dto = handler->DropTable(table_name);
    ASSERT_EQ(0, status_dto->code->getValue());

    // drop table which not exists.
    status_dto = handler->DropTable(table_name + "57575yfhfdhfhdh436gdsgpppdgsgv3233");
    ASSERT_EQ(StatusCode::TABLE_NOT_EXISTS, status_dto->code->getValue());
}

TEST_F(WebHandlerTest, HAS_TABLE_TEST) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());
    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();

    GenTable(table_name->std_str(), 10, 10, "L2");

    milvus::server::web::OQueryParams query_params;
    OString response;
    auto status_dto = handler->GetTable(table_name, query_params, response);
    ASSERT_EQ(0, status_dto->code->getValue());
}

TEST_F(WebHandlerTest, GET_TABLE) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), 10, 10, "L2");

    milvus::server::web::OQueryParams query_params;
    OString result;
    auto status_dto = handler->GetTable(table_name, query_params, result);
    ASSERT_EQ(0, status_dto->code->getValue());

    auto result_json = nlohmann::json::parse(result->std_str());
    ASSERT_EQ(10, result_json["dimension"].get<int64_t>());
    ASSERT_EQ(10, result_json["index_file_size"].get<int64_t>());
    ASSERT_EQ("L2", result_json["metric_type"].get<std::string>());
}

TEST_F(WebHandlerTest, INSERT_COUNT) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), 16, 10, "L2");

    nlohmann::json body_json;
    body_json["vectors"] = RandomRecordsJson(16, 1000);
    auto ids_dto = milvus::server::web::VectorIdsDto::createShared();
    auto status_dto = handler->Insert(table_name, body_json.dump().c_str(), ids_dto);
    ASSERT_EQ(0, status_dto->code->getValue());
    ASSERT_EQ(1000, ids_dto->ids->count());

    sleep(2);

    milvus::server::web::OQueryParams query_params;
    query_params.put("fields", "num");
    OString result;
    status_dto = handler->GetTable(table_name, query_params, result);
    ASSERT_EQ(0, status_dto->code->getValue());

    auto result_json = nlohmann::json::parse(result->std_str());
    ASSERT_EQ(1000, result_json["count"].get<int64_t>());
}

TEST_F(WebHandlerTest, INDEX) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), 16, 10, "L2");

    auto index_request_dto = milvus::server::web::IndexRequestDto::createShared();
    index_request_dto->index_type = "FLAT";
    index_request_dto->nlist = 10;

    milvus::server::web::StatusDto::createShared();

    auto status_dto = handler->CreateIndex(table_name, index_request_dto);
    ASSERT_EQ(0, status_dto->code->getValue());

    status_dto = handler->DropIndex(table_name);
    ASSERT_EQ(0, status_dto->code->getValue());

    // invalid index_type
    index_request_dto->index_type = "AAA";
    status_dto = handler->CreateIndex(table_name, index_request_dto);
    ASSERT_NE(0, status_dto->code->getValue());
    ASSERT_EQ(StatusCode::ILLEGAL_INDEX_TYPE, status_dto->code->getValue());

    // invalid nlist
    index_request_dto->index_type = "FLAT";
    index_request_dto->nlist = -1;
    status_dto = handler->CreateIndex(table_name, index_request_dto);
    ASSERT_NE(0, status_dto->code->getValue());
    ASSERT_EQ(StatusCode::ILLEGAL_NLIST, status_dto->code->getValue());
}

TEST_F(WebHandlerTest, PARTITION) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), 16, 10, "L2");

    auto partition_dto = milvus::server::web::PartitionRequestDto::createShared();
    partition_dto->partition_tag = "test";

    auto status_dto = handler->CreatePartition(table_name, partition_dto);
    ASSERT_EQ(0, status_dto->code->getValue());

    auto partitions_dto = milvus::server::web::PartitionListDto::createShared();
    status_dto = handler->ShowPartitions("0", "10", table_name, partitions_dto);
    ASSERT_EQ(2, partitions_dto->partitions->count());

    status_dto = handler->DropPartition(table_name, "test");
    ASSERT_EQ(0, status_dto->code->getValue());

    // Show all partitions
    partitions_dto = milvus::server::web::PartitionListDto::createShared();
    status_dto = handler->ShowPartitions("0", "10", table_name, partitions_dto);
}

TEST_F(WebHandlerTest, SEARCH) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), TABLE_DIM, 10, "L2");

    nlohmann::json insert_json;
    insert_json["vectors"] = RandomRecordsJson(TABLE_DIM, 1000);
    auto ids_dto = milvus::server::web::VectorIdsDto::createShared();
    auto status_dto = handler->Insert(table_name, insert_json.dump().c_str(), ids_dto);
    ASSERT_EQ(milvus::server::web::SUCCESS, status_dto->code->getValue());

    nlohmann::json search_pram_json;
    search_pram_json["vectors"] = RandomRecordsJson(TABLE_DIM, 10);
    search_pram_json["topk"] = 1;
    search_pram_json["nprobe"] = 1;

    nlohmann::json search_json;
    search_json["search"] = search_pram_json;

    OString result = "";
    status_dto = handler->VectorsOp(table_name, search_json.dump().c_str(), result);
    ASSERT_EQ(0, status_dto->code->getValue()) << status_dto->message->std_str();
}

TEST_F(WebHandlerTest, SYSYEM_INFO) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    OQueryParams query_params;
    OString result;

    auto status_dto = handler->SystemInfo("status", query_params, result);
    ASSERT_EQ(0, status_dto->code->getValue());
//    ASSERT_EQ("OK", cmd_dto->reply->std_str());

    status_dto = handler->SystemInfo("version", query_params, result);
    ASSERT_EQ(0, status_dto->code->getValue());
//    ASSERT_EQ("0.7.0", cmd_dto->reply->std_str());
}

TEST_F(WebHandlerTest, FLUSH) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), 16, 10, "L2");

    nlohmann::json body_json;
    body_json["vectors"] = RandomRecordsJson(16, 1000);
    auto ids_dto = milvus::server::web::VectorIdsDto::createShared();
    auto status_dto = handler->Insert(table_name, body_json.dump().c_str(), ids_dto);
    ASSERT_EQ(0, status_dto->code->getValue()) << status_dto->message->std_str();

    nlohmann::json flush_json;
    flush_json["flush"]["table_names"] = {table_name->std_str()};
    OString result;
    status_dto = handler->SystemOp("task", flush_json.dump().c_str(), result);
    ASSERT_EQ(milvus::server::web::SUCCESS, status_dto->code->getValue());
}

TEST_F(WebHandlerTest, COMPACT) {
    handler->RegisterRequestHandler(milvus::server::RequestHandler());

    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name->std_str(), 16, 10, "L2");

    nlohmann::json body_json;
    body_json["vectors"] = RandomRecordsJson(16, 1000);
    auto ids_dto = milvus::server::web::VectorIdsDto::createShared();
    auto status_dto = handler->Insert(table_name, body_json.dump().c_str(), ids_dto);
    ASSERT_EQ(0, status_dto->code->getValue()) << status_dto->message->std_str();

    nlohmann::json compact_json;
    compact_json["compact"]["table_name"] = table_name->std_str();
    OString result;
    status_dto = handler->SystemOp("task", compact_json.dump().c_str(), result);
    ASSERT_EQ(milvus::server::web::SUCCESS, status_dto->code->getValue());
}

///////////////////////////////////////////////////////////////////////////////////////

namespace {
static const char* CONTROLLER_TEST_VALID_CONFIG_STR =
    "# Default values are used when you make no changes to the following parameters.\n"
    "\n"
    "version: 0.1"
    "\n"
    "server_config:\n"
    "  address: 0.0.0.0                  # milvus server ip address (IPv4)\n"
    "  port: 19530                       # port range: 1025 ~ 65534\n"
    "  deploy_mode: single               \n"
    "  time_zone: UTC+8\n"
    "\n"
    "db_config:\n"
    "  backend_url: sqlite://:@:/        \n"
    "\n"
    "  insert_buffer_size: 4             # GB, maximum insert buffer size allowed\n"
    "  preload_table:                    \n"
    "\n"
    "storage_config:\n"
    "  primary_path: /tmp/milvus_web_controller_test        # path used to store data and meta\n"
    "  secondary_path:                   # path used to store data only, split by semicolon\n"
    "\n"
    "metric_config:\n"
    "  enable_monitor: false             # enable monitoring or not\n"
    "  address: 127.0.0.1\n"
    "  port: 8080                        # port prometheus uses to fetch metrics\n"
    "\n"
    "cache_config:\n"
    "  cpu_cache_capacity: 4             # GB, CPU memory used for cache\n"
    "  cpu_cache_threshold: 0.85         \n"
    "  cache_insert_data: false          # whether to load inserted data into cache\n"
    "\n"
    "engine_config:\n"
    "  use_blas_threshold: 20            \n"
    "\n"
    #ifdef MILVUS_GPU_VERSION
    "gpu_resource_config:\n"
    "  enable: true                      # whether to enable GPU resources\n"
    "  cache_capacity: 4                 # GB, size of GPU memory per card used for cache, must be a positive integer\n"
    "  search_resources:                 # define the GPU devices used for search computation, must be in format gpux\n"
    "    - gpu0\n"
    "  build_index_resources:            # define the GPU devices used for index building, must be in format gpux\n"
    "    - gpu0\n"
    #endif
    "\n";

static const char* CONTROLLER_TEST_TABLE_NAME = "controller_unit_test";
static const char* CONTROLLER_TEST_CONFIG_DIR = "/tmp/milvus_web_controller_test/";
static const char* CONTROLLER_TEST_CONFIG_FILE = "config.yaml";

class TestClient : public oatpp::web::client::ApiClient {
 public:
#include OATPP_CODEGEN_BEGIN(ApiClient)
 API_CLIENT_INIT(TestClient)

    API_CALL("GET", "/", root)

    API_CALL("GET", "/state", getState)

    API_CALL("GET", "/devices", getDevices)

    API_CALL("GET", "/config/advanced", getAdvanced)

    API_CALL("OPTIONS", "/config/advanced", optionsAdvanced)

    API_CALL("PUT", "/config/advanced", setAdvanced,
             BODY_DTO(milvus::server::web::AdvancedConfigDto::ObjectWrapper, body))

#ifdef MILVUS_GPU_VERSION
    API_CALL("OPTIONS", "config/gpu_resources", optionsGpuConfig)

    API_CALL("GET", "/config/gpu_resources", getGPUConfig)

    API_CALL("PUT", "/config/gpu_resources", setGPUConfig,
             BODY_DTO(milvus::server::web::GPUConfigDto::ObjectWrapper, body))
#endif

    API_CALL("OPTIONS", "/tables", optionsTables)

    API_CALL("POST", "/tables", createTable, BODY_DTO(milvus::server::web::TableRequestDto::ObjectWrapper, body))

    API_CALL("GET", "/tables", showTables, QUERY(String, offset), QUERY(String, page_size))

    API_CALL("OPTIONS", "/tables/{table_name}", optionsTable, PATH(String, table_name, "table_name"))

    API_CALL("GET", "/tables/{table_name}", getTable, PATH(String, table_name, "table_name"), QUERY(String, info))

    API_CALL("DELETE", "/tables/{table_name}", dropTable, PATH(String, table_name, "table_name"))

    API_CALL("OPTIONS", "/tables/{table_name}/indexes", optionsIndexes, PATH(String, table_name, "table_name"))

    API_CALL("POST", "/tables/{table_name}/indexes", createIndex,
             PATH(String, table_name, "table_name"), BODY_DTO(milvus::server::web::IndexRequestDto::ObjectWrapper, body))

    API_CALL("GET", "/tables/{table_name}/indexes", getIndex, PATH(String, table_name, "table_name"))

    API_CALL("DELETE", "/tables/{table_name}/indexes", dropIndex, PATH(String, table_name, "table_name"))

    API_CALL("OPTIONS", "/tables/{table_name}/partitions", optionsPartitions, PATH(String, table_name, "table_name"))

    API_CALL("POST", "/tables/{table_name}/partitions", createPartition,
             PATH(String, table_name, "table_name"), BODY_DTO(milvus::server::web::PartitionRequestDto::ObjectWrapper, body))

    API_CALL("GET", "/tables/{table_name}/partitions", showPartitions,
             PATH(String, table_name, "table_name"), QUERY(String, offset), QUERY(String, page_size))

    API_CALL("OPTIONS", "/tables/{table_name}/partitions/{partition_tag}", optionsParTag,
             PATH(String, table_name, "table_name"), PATH(String, partition_tag, "partition_tag"))

    API_CALL("DELETE", "/tables/{table_name}/partitions/{partition_tag}", dropPartition,
             PATH(String, table_name, "table_name"), PATH(String, partition_tag))

    API_CALL("OPTIONS", "/tables/{table_name}/vectors", optionsVectors, PATH(String, table_name, "table_name"))

    API_CALL("POST", "/tables/{table_name}/vectors", insert,
             PATH(String, table_name, "table_name"), BODY_STRING(String, body))

    API_CALL("PUT", "/tables/{table_name}/vectors", vectorsOp,
             PATH(String, table_name, "table_name"), BODY_STRING(String, body))

    API_CALL("GET", "/system/{msg}", cmd, PATH(String, cmd_str, "msg"), QUERY(String, action), QUERY(String, target))

    API_CALL("PUT", "/system/{op}", op, PATH(String, cmd_str, "op"), BODY_STRING(String, body))

#include OATPP_CODEGEN_END(ApiClient)
};

class WebControllerTest : public testing::Test {
 protected:
    static void
    SetUpTestCase() {
        // Basic config
        std::string config_path = std::string(CONTROLLER_TEST_CONFIG_DIR).append(CONTROLLER_TEST_CONFIG_FILE);
        std::fstream fs(config_path.c_str(), std::ios_base::out);
        fs << CONTROLLER_TEST_VALID_CONFIG_STR;
        fs.close();

        milvus::server::Config& config = milvus::server::Config::GetInstance();
        config.LoadConfigFile(config_path);

        auto res_mgr = milvus::scheduler::ResMgrInst::GetInstance();
        res_mgr->Clear();
        res_mgr->Add(milvus::scheduler::ResourceFactory::Create("disk", "DISK", 0, false));
        res_mgr->Add(milvus::scheduler::ResourceFactory::Create("cpu", "CPU", 0));
        res_mgr->Add(milvus::scheduler::ResourceFactory::Create("gtx1660", "GPU", 0));

        auto default_conn = milvus::scheduler::Connection("IO", 500.0);
        auto PCIE = milvus::scheduler::Connection("IO", 11000.0);
        res_mgr->Connect("disk", "cpu", default_conn);
        res_mgr->Connect("cpu", "gtx1660", PCIE);
        res_mgr->Start();
        milvus::scheduler::SchedInst::GetInstance()->Start();
        milvus::scheduler::JobMgrInst::GetInstance()->Start();

        milvus::engine::DBOptions opt;

        milvus::server::Config::GetInstance().SetDBConfigBackendUrl("sqlite://:@:/");
        boost::filesystem::remove_all(CONTROLLER_TEST_CONFIG_DIR);
        milvus::server::Config::GetInstance().SetStorageConfigPrimaryPath(CONTROLLER_TEST_CONFIG_DIR);

        milvus::server::DBWrapper::GetInstance().StartService();

        milvus::server::Config::GetInstance().SetServerConfigWebPort("29999");

        milvus::server::web::WebServer::GetInstance().Start();

        sleep(5);
    }

    static void
    TearDownTestCase() {
        milvus::server::web::WebServer::GetInstance().Stop();

        milvus::server::DBWrapper::GetInstance().StopService();
        milvus::scheduler::JobMgrInst::GetInstance()->Stop();
        milvus::scheduler::ResMgrInst::GetInstance()->Stop();
        milvus::scheduler::SchedInst::GetInstance()->Stop();
        boost::filesystem::remove_all(CONTROLLER_TEST_CONFIG_DIR);
    }

    void
    GenTable(const OString& table_name, int64_t dim, int64_t index_size, const OString& metric) {
        auto response = client_ptr->getTable(table_name, "", conncetion_ptr);
        if (OStatus::CODE_200.code == response->getStatusCode()) {
            return;
        }
        auto table_dto = milvus::server::web::TableRequestDto::createShared();
        table_dto->table_name = table_name;
        table_dto->dimension = dim;
        table_dto->index_file_size = index_size;
        table_dto->metric_type = metric;
        client_ptr->createTable(table_dto, conncetion_ptr);
    }

    milvus::Status
    FlushTable(const std::string& table_name) {
        nlohmann::json flush_json;
        flush_json["flush"]["table_names"] = {table_name};
        auto response = client_ptr->op("task", flush_json.dump().c_str(), conncetion_ptr);
        if (OStatus::CODE_200.code != response->getStatusCode()) {
            return milvus::Status(milvus::SERVER_UNEXPECTED_ERROR, response->readBodyToString()->std_str());
        }

        return milvus::Status::OK();
    }

    milvus::Status
    FlushTable(const OString& table_name) {
        nlohmann::json flush_json;
        flush_json["flush"]["table_names"] = {table_name->std_str()};
        auto response = client_ptr->op("task", flush_json.dump().c_str(), conncetion_ptr);
        if (OStatus::CODE_200.code != response->getStatusCode()) {
            return milvus::Status(milvus::SERVER_UNEXPECTED_ERROR, response->readBodyToString()->std_str());
        }

        return milvus::Status::OK();
    }

    milvus::Status
    InsertData(const OString& table_name, int64_t dim, int64_t count, std::string tag = "", bool bin = false) {
        nlohmann::json insert_json;

        if (bin)
            insert_json["vectors"] = RandomBinRecordsJson(dim, count);
        else
            insert_json["vectors"] = RandomRecordsJson(dim, count);

        if (!tag.empty()) {
            insert_json["partition_tag"] = tag;
        }

        auto response = client_ptr->insert(table_name, insert_json.dump().c_str(), conncetion_ptr);
        if (OStatus::CODE_201.code != response->getStatusCode()) {
            return milvus::Status(milvus::SERVER_UNEXPECTED_ERROR, response->readBodyToString()->c_str());
        }

        return FlushTable(table_name);
    }

    milvus::Status
    InsertData(const OString& table_name, int64_t dim, int64_t count, const std::vector<int64_t>& ids, std::string tag = "", bool bin=false) {
       nlohmann::json insert_json;

       if (bin)
           insert_json["vectors"] = RandomBinRecordsJson(dim, count);
       else
           insert_json["vectors"] = RandomRecordsJson(dim, count);

       if (!ids.empty()) {
           insert_json["ids"] = ids;
       }

        if (!tag.empty()) {
            insert_json["partition_tag"] = tag;
        }

       auto response = client_ptr->insert(table_name, insert_json.dump().c_str(), conncetion_ptr);
       if (OStatus::CODE_201.code != response->getStatusCode()) {
           return milvus::Status(milvus::SERVER_UNEXPECTED_ERROR, response->readBodyToString()->c_str());
       }

       return FlushTable(table_name);
    }

    void
    SetUp() override {
        OATPP_COMPONENT(std::shared_ptr<oatpp::network::ClientConnectionProvider>, clientConnectionProvider);
        OATPP_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>, objectMapper);
        object_mapper = objectMapper;

        auto requestExecutor = oatpp::web::client::HttpRequestExecutor::createShared(clientConnectionProvider);
        client_ptr = TestClient::createShared(requestExecutor, objectMapper);

        conncetion_ptr = client_ptr->getConnection();
    }

    void
    TearDown() override {
    };

 protected:
    std::shared_ptr<oatpp::data::mapping::ObjectMapper> object_mapper;
    std::shared_ptr<oatpp::web::client::RequestExecutor::ConnectionHandle> conncetion_ptr;
    std::shared_ptr<TestClient> client_ptr;

 protected:
    void GenTable(const std::string& table_name, int64_t dim, int64_t index_file_size, int64_t metric_type) {
        auto table_dto = milvus::server::web::TableRequestDto::createShared();
        table_dto->table_name = OString(table_name.c_str());
        table_dto->dimension = dim;
        table_dto->index_file_size = index_file_size;
        table_dto->metric_type = metric_type;

        client_ptr->createTable(table_dto, conncetion_ptr);
    }
};

} // namespace
TEST_F(WebControllerTest, OPTIONS) {
    auto response = client_ptr->root(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->getState(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->optionsAdvanced(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

#ifdef MILVUS_GPU_VERSION
    response = client_ptr->optionsGpuConfig(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());
#endif

    response = client_ptr->optionsIndexes("test", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    response = client_ptr->optionsParTag("test", "tag", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    response = client_ptr->optionsPartitions("table_name", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    response = client_ptr->optionsTable("table", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    response = client_ptr->optionsTables(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    response = client_ptr->optionsVectors("table", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());
}

TEST_F(WebControllerTest, CREATE_TABLE) {
    auto table_dto = milvus::server::web::TableRequestDto::createShared();
    auto response = client_ptr->createTable(table_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    auto error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::BODY_FIELD_LOSS, error_dto->code) << error_dto->message->std_str();

    OString table_name = "web_test_create_table" + OString(RandomName().c_str());

    table_dto->table_name = table_name;
    response = client_ptr->createTable(table_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::BODY_FIELD_LOSS, error_dto->code) << error_dto->message->std_str();

    table_dto->dimension = 128;
    table_dto->index_file_size = 10;
    table_dto->metric_type = "L2";

    response = client_ptr->createTable(table_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::SUCCESS, result_dto->code->getValue()) << result_dto->message->std_str();

    // invalid table name
    table_dto->table_name = "9090&*&()";
    response = client_ptr->createTable(table_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
}

TEST_F(WebControllerTest, GET_TABLE_META) {
    OString table_name = "web_test_create_table" + OString(RandomName().c_str());
    GenTable(table_name, 10, 10, "L2");

    OQueryParams params;

    auto response = client_ptr->getTable(table_name, "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::TableFieldsDto>(object_mapper.get());
    ASSERT_EQ(table_name->std_str(), result_dto->table_name->std_str());
    ASSERT_EQ(10, result_dto->dimension);
    ASSERT_EQ("L2", result_dto->metric_type->std_str());
    ASSERT_EQ(10, result_dto->index_file_size->getValue());
    ASSERT_EQ("FLAT", result_dto->index->std_str());

    // invalid table name
    table_name = "57474dgdfhdfhdh  dgd";
    response = client_ptr->getTable(table_name, "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    auto status_sto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::ILLEGAL_TABLE_NAME, status_sto->code->getValue());

    table_name = "test_table_not_found_000000000111010101002020203020aaaaa3030435";
    response = client_ptr->getTable(table_name, "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
}

TEST_F(WebControllerTest, GET_TABLE_STAT) {
    OString table_name = "web_test_get_table_stat" + OString(RandomName().c_str());
    GenTable(table_name, 10, 10, "L2");

    OQueryParams params;

    auto response = client_ptr->getTable(table_name, "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::TableFieldsDto>(object_mapper.get());
    ASSERT_EQ(table_name->std_str(), result_dto->table_name->std_str());
    ASSERT_EQ(10, result_dto->dimension);
    ASSERT_EQ("L2", result_dto->metric_type->std_str());
    ASSERT_EQ(10, result_dto->index_file_size->getValue());
    ASSERT_EQ("FLAT", result_dto->index->std_str());

    // invalid table name
    table_name = "57474dgdfhdfhdh  dgd";
    response = client_ptr->getTable(table_name, "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    auto status_sto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::ILLEGAL_TABLE_NAME, status_sto->code->getValue());

    table_name = "test_table_not_found_000000000111010101002020203020aaaaa3030435";
    response = client_ptr->getTable(table_name, "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
}

TEST_F(WebControllerTest, SHOW_TABLES) {
    // test query table limit 1
    auto response = client_ptr->showTables("1", "1", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::TableListFieldsDto>(object_mapper.get());
    ASSERT_TRUE(result_dto->count->getValue() >= 0);

    // test query table empty
    response = client_ptr->showTables("0", "0", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->showTables("-1", "0", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());

    response = client_ptr->showTables("0", "-10", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());

    // test wrong param
    response = client_ptr->showTables("0.1", "1", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());

    response = client_ptr->showTables("1", "1.1", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());

    response = client_ptr->showTables("0", "90000000000000000000000000000000000000000000000000000000", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
}

TEST_F(WebControllerTest, DROP_TABLE) {
    auto table_name = "table_drop_test" + OString(RandomName().c_str());
    GenTable(table_name, 128, 100, "L2");
    sleep(1);

    auto response = client_ptr->dropTable(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    table_name = "table_drop_test_not_exists_" + OString(RandomName().c_str());
    response = client_ptr->dropTable(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
    auto error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::TABLE_NOT_EXISTS, error_dto->code->getValue());
}

TEST_F(WebControllerTest, INSERT) {
    auto table_name = "test_insert_table_test" + OString(RandomName().c_str());
    const int64_t dim = 64;
    GenTable(table_name, dim, 100, "L2");

    nlohmann::json insert_json;
    insert_json["vectors"] = RandomRecordsJson(dim, 20);

    auto response = client_ptr->insert(table_name, insert_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::VectorIdsDto>(object_mapper.get());
    ASSERT_EQ(20, result_dto->ids->count());

    response = client_ptr->insert(table_name + "ooowrweindexsgs", insert_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());

    response = client_ptr->dropTable(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());
}

TEST_F(WebControllerTest, INSERT_BIN) {
    auto table_name = "test_insert_bin_table_test" + OString(RandomName().c_str());
    const int64_t dim = 64;
    GenTable(table_name, dim, 100, "HAMMING");

    nlohmann::json insert_json;
    insert_json["vectors"] = RandomBinRecordsJson(dim, 20);

    auto response = client_ptr->insert(table_name, insert_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode()) << response->readBodyToString()->std_str();

    auto status = FlushTable(table_name);
    ASSERT_TRUE(status.ok()) << status.message();

    auto result_dto = response->readBodyToDto<milvus::server::web::VectorIdsDto>(object_mapper.get());
    ASSERT_EQ(20, result_dto->ids->count());

    response = client_ptr->dropTable(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());
}

TEST_F(WebControllerTest, INSERT_IDS) {
    auto table_name = "test_insert_table_test" + OString(RandomName().c_str());
    const int64_t dim = 64;
    GenTable(table_name, dim, 100, "L2");

    std::vector<int64_t> ids;
    for (size_t i = 0; i < 20; i++) {
        ids.emplace_back(i);
    }

    nlohmann::json insert_json;
    insert_json["vectors"] = RandomRecordsJson(dim, 20);
    insert_json["ids"] = ids;

    auto response = client_ptr->insert(table_name, insert_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode()) << response->readBodyToString()->std_str();
    auto result_dto = response->readBodyToDto<milvus::server::web::VectorIdsDto>(object_mapper.get());
    ASSERT_EQ(20, result_dto->ids->count());

    response = client_ptr->dropTable(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());
}

TEST_F(WebControllerTest, INDEX) {
    auto table_name = "test_insert_table_test" + OString(RandomName().c_str());
    GenTable(table_name, 64, 100, "L2");

    // test index with imcomplete param
    auto index_dto = milvus::server::web::IndexRequestDto::createShared();
    auto response = client_ptr->createIndex(table_name, index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());
    auto create_index_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::SUCCESS, create_index_dto->code);

    // drop index
    response = client_ptr->dropIndex(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    index_dto->index_type = milvus::server::web::IndexMap.at(milvus::engine::EngineType::FAISS_IDMAP).c_str();

    response = client_ptr->createIndex(table_name, index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());
    // drop index
    response = client_ptr->dropIndex(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    // create index without existing table
    response = client_ptr->createIndex(table_name + "fgafafafafafUUUUUUa124254", index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());

    index_dto->index_type = "J46";
    response = client_ptr->createIndex(table_name, index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::ILLEGAL_INDEX_TYPE, result_dto->code);

    index_dto->index_type = milvus::server::web::IndexMap.at(milvus::engine::EngineType::FAISS_IDMAP).c_str();
    index_dto->nlist = 10;

    response = client_ptr->createIndex(table_name, index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());

    // drop index
    response = client_ptr->dropIndex(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    // invalid index type
    index_dto->index_type = 100;
    response = client_ptr->createIndex(table_name, index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());

    // insert data and create index
    response = client_ptr->dropIndex(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    auto status = InsertData(table_name, 64, 200);
    ASSERT_TRUE(status.ok()) << status.message();

    index_dto->index_type = milvus::server::web::IndexMap.at(milvus::engine::EngineType::FAISS_IDMAP).c_str();
    response = client_ptr->createIndex(table_name, index_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());

    // get index
    response = client_ptr->getIndex(table_name, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
    auto result_index_dto = response->readBodyToDto<milvus::server::web::IndexDto>(object_mapper.get());
    ASSERT_EQ("FLAT", result_index_dto->index_type->std_str());
    ASSERT_EQ(10, result_index_dto->nlist->getValue());
    // get index of table which not exists
    response = client_ptr->getIndex(table_name + "dfaedXXXdfdfet4t343aa4", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
    auto error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::TABLE_NOT_EXISTS, error_dto->code->getValue());
}

TEST_F(WebControllerTest, PARTITION) {
    const OString table_name = "test_controller_partition_" + OString(RandomName().c_str());
    GenTable(table_name, 64, 100, "L2");

    auto par_param = milvus::server::web::PartitionRequestDto::createShared();
    auto response = client_ptr->createPartition(table_name, par_param);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    auto error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::BODY_FIELD_LOSS, error_dto->code);

    response = client_ptr->createPartition(table_name, par_param);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::BODY_FIELD_LOSS, error_dto->code);

    par_param->partition_tag = "tag01";
    response = client_ptr->createPartition(table_name, par_param);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode());
    auto create_result_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::SUCCESS, create_result_dto->code);

    response = client_ptr->createPartition(table_name + "afafanotgitdiexists", par_param);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::TABLE_NOT_EXISTS, error_dto->code);

    // insert 200 vectors into table with tag = 'tag01'
    auto status = InsertData(table_name, 64, 200, "tag01");
    ASSERT_TRUE(status.ok()) << status.message();

    // Show all partitins
    response = client_ptr->showPartitions(table_name, "0", "10", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
    auto result_dto = response->readBodyToDto<milvus::server::web::PartitionListDto>(object_mapper.get());
    ASSERT_EQ(2, result_dto->partitions->count());
    ASSERT_EQ("tag01", result_dto->partitions->get(1)->partition_tag->std_str());

    response = client_ptr->showPartitions(table_name, "0", "-1", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    response = client_ptr->showPartitions(table_name, "0.1", "7", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    response = client_ptr->showPartitions(table_name, "0", "1.6", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    response = client_ptr->showPartitions(table_name, "567a", "1", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());

    // show without existing tables
    response = client_ptr->showPartitions(table_name + "dfafaefaluanqibazao990099", "0", "10", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::TABLE_NOT_EXISTS, error_dto->code->getValue());

    response = client_ptr->dropPartition(table_name, "tag01", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_204.code, response->getStatusCode());

    // drop without existing tables
    response = client_ptr->dropPartition(table_name + "565755682353464aaasafdsfagagqq1223", "tag01", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
}

TEST_F(WebControllerTest, SEARCH) {
    const OString table_name = "test_search_table_test" + OString(RandomName().c_str());
    GenTable(table_name, 64, 100, "L2");

    // Insert 200 vectors into table
    auto status = InsertData(table_name, 64, 200);
    ASSERT_TRUE(status.ok()) << status.message();

    //Create partition and insert 200 vectors into it
    auto par_param = milvus::server::web::PartitionRequestDto::createShared();
    par_param->partition_tag = "tag" + OString(RandomName().c_str());
    auto response = client_ptr->createPartition(table_name, par_param);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode())
                        << "Error: " << response->readBodyToString()->std_str();

    status = InsertData(table_name, 64, 200, par_param->partition_tag->std_str());
    ASSERT_TRUE(status.ok()) << status.message();

    // Test search
    nlohmann::json search_json;
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    auto error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_NE(milvus::server::web::StatusCode::SUCCESS, error_dto->code);

    search_json["search"]["nprobe"] = 1;
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::BODY_FIELD_LOSS, error_dto->code);

    search_json["search"]["topk"] = 1;
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_NE(milvus::server::web::StatusCode::SUCCESS, error_dto->code);

    search_json["search"]["vectors"] = RandomRecordsJson(64, 10);
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    auto result_json = nlohmann::json::parse(response->readBodyToString()->std_str());
    ASSERT_TRUE(result_json.contains("num"));
    ASSERT_TRUE(result_json["num"].is_number());
    ASSERT_EQ(10, result_json["num"].get<int64_t>());

    ASSERT_TRUE(result_json.contains("result"));
    ASSERT_TRUE(result_json["result"].is_array());

    auto result0_json = result_json["result"][0];
    ASSERT_TRUE(result0_json.is_array());
    ASSERT_EQ(1, result0_json.size());

    // Test search with tags
    nlohmann::json par_json;
    par_json.push_back(par_param->partition_tag->std_str());
    search_json["search"]["partition_tags"] = par_json;

    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    // Test search without existing table
    response = client_ptr->vectorsOp(table_name + "999piyanning", search_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_404.code, response->getStatusCode());
    error_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_EQ(milvus::server::web::StatusCode::TABLE_NOT_EXISTS, error_dto->code->getValue());
}

TEST_F(WebControllerTest, SEARCH_BIN) {
    const OString table_name = "test_search_bin_table_test" + OString(RandomName().c_str());
    GenTable(table_name, 64, 100, "HAMMING");

    // Insert 200 vectors into table
    auto status = InsertData(table_name, 64, 200, "", true);
    ASSERT_TRUE(status.ok()) << status.message();

    //Create partition and insert 200 vectors into it
    auto par_param = milvus::server::web::PartitionRequestDto::createShared();
    par_param->partition_tag = "tag" + OString(RandomName().c_str());
    auto response = client_ptr->createPartition(table_name, par_param);
    ASSERT_EQ(OStatus::CODE_201.code, response->getStatusCode())
                        << "Error: " << response->readBodyToString()->std_str();

    status = InsertData(table_name, 64, 200, par_param->partition_tag->std_str(), true);
    ASSERT_TRUE(status.ok()) << status.message();

    // Test search
    nlohmann::json search_json;
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    auto result_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_NE(milvus::server::web::StatusCode::SUCCESS, result_dto->code);

    search_json["search"]["nprobe"] = 1;
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    result_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_NE(milvus::server::web::StatusCode::SUCCESS, result_dto->code);

    search_json["search"]["topk"] = 1;
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    result_dto = response->readBodyToDto<milvus::server::web::StatusDto>(object_mapper.get());
    ASSERT_NE(milvus::server::web::StatusCode::SUCCESS, result_dto->code);

    search_json["search"]["vectors"] = RandomBinRecordsJson(64, 10);
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    // validate search result
    auto result_json = nlohmann::json::parse(response->readBodyToString()->c_str());
    ASSERT_TRUE(result_json.contains("result"));
    ASSERT_TRUE(result_json["result"].is_array());
    ASSERT_EQ(10, result_json["result"].size());

    auto result0_json = result_json["result"][0];
    ASSERT_TRUE(result0_json.is_array());
    ASSERT_EQ(1, result0_json.size());

    // Test search with tags
    search_json["search"]["partition_tags"] = std::vector<std::string>();
    search_json["search"]["partition_tags"].push_back(par_param->partition_tag->std_str());
    response = client_ptr->vectorsOp(table_name, search_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
}

TEST_F(WebControllerTest, CMD) {
    auto response = client_ptr->cmd("status", "", "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->cmd("version", "", "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->cmd("mode", "", "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->cmd("tasktable", "", "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    response = client_ptr->cmd("info", "", "", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    GenTable("test_cmd", 16, 10, "L2");
    response = client_ptr->cmd("task", "load", "test_cmd", conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
}

TEST_F(WebControllerTest, ADVANCED_CONFIG) {
    auto response = client_ptr->getAdvanced(conncetion_ptr);

    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    auto config_dto = milvus::server::web::AdvancedConfigDto::createShared();
    response = client_ptr->setAdvanced(config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    config_dto->cpu_cache_capacity = 3;
    response = client_ptr->setAdvanced(config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    config_dto->cache_insert_data = true;
    response = client_ptr->setAdvanced(config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

#ifdef MILVUS_GPU_VERSION

    config_dto->gpu_search_threshold = 1000;
    response = client_ptr->setAdvanced(config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

#endif

    config_dto->use_blas_threshold = 1000;
    response = client_ptr->setAdvanced(config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    //// test fault
    // cpu cache capacity exceed total memory
    config_dto->cpu_cache_capacity = 10000000;
    response = client_ptr->setAdvanced(config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
}

#ifdef MILVUS_GPU_VERSION
TEST_F(WebControllerTest, GPU_CONFIG) {
    auto response = client_ptr->getGPUConfig(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode()) << response->readBodyToString()->c_str();

    auto gpu_config_dto = milvus::server::web::GPUConfigDto::createShared();

    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    gpu_config_dto->enable = true;
    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    gpu_config_dto->cache_capacity = 2;
    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    gpu_config_dto->build_index_resources = gpu_config_dto->build_index_resources->createShared();
    gpu_config_dto->build_index_resources->pushBack("GPU0");
    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    gpu_config_dto->search_resources = gpu_config_dto->search_resources->createShared();
    gpu_config_dto->search_resources->pushBack("GPU0");

    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());

    //// test fault config
    // cache capacity exceed GPU mem size
    gpu_config_dto->cache_capacity = 100000;
    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
    gpu_config_dto->cache_capacity = 1;

    // duplicate resources
    gpu_config_dto->search_resources->clear();
    gpu_config_dto->search_resources->pushBack("GPU0");
    gpu_config_dto->search_resources->pushBack("GPU1");
    gpu_config_dto->search_resources->pushBack("GPU0");
    response = client_ptr->setGPUConfig(gpu_config_dto, conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_400.code, response->getStatusCode());
}

#endif

TEST_F(WebControllerTest, DEVICES_CONFIG) {
    auto response = client_ptr->getDevices(conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
}
TEST_F(WebControllerTest, FLUSH) {
    auto table_name = milvus::server::web::OString(TABLE_NAME) + RandomName().c_str();
    GenTable(table_name, 16, 10, "L2");

    auto status = InsertData(table_name, 16, 1000);
    ASSERT_TRUE(status.ok()) << status.message();

    nlohmann::json flush_json;
    flush_json["flush"]["table_names"] = {table_name->std_str()};
    auto response = client_ptr->op("task", flush_json.dump().c_str(), conncetion_ptr);
    ASSERT_EQ(OStatus::CODE_200.code, response->getStatusCode());
}

