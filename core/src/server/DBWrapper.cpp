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

#include "server/DBWrapper.h"

#include <omp.h>
#include <cmath>
#include <string>
#include <vector>

#include "config/ServerConfig.h"
#include "db/Constants.h"
#include "db/DBFactory.h"
#include "db/snapshot/OperationExecutor.h"
#include "utils/CommonUtil.h"
#include "utils/Log.h"
#include "utils/StringHelpFunctions.h"

namespace milvus {
namespace server {

Status
DBWrapper::StartService() {
    Status s;

    // db config
    engine::DBOptions opt;
<<<<<<< HEAD
    opt.meta_.backend_uri_ = config.general.meta_uri();

    std::string path = config.storage.path();
    opt.meta_.path_ = path + engine::DB_FOLDER;

    opt.auto_flush_interval_ = config.storage.auto_flush_interval();
    opt.metric_enable_ = config.metric.enable();
    opt.insert_buffer_size_ = config.cache.insert_buffer_size();

    if (not config.cluster.enable()) {
=======
    s = config.GetGeneralConfigMetaURI(opt.meta_.backend_uri_);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

    std::string path;
    s = config.GetStorageConfigPath(path);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }
    opt.meta_.path_ = path + "/db";

    s = config.GetStorageConfigAutoFlushInterval(opt.auto_flush_interval_);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

    s = config.GetStorageConfigFileCleanupTimeup(opt.file_cleanup_timeout_);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

    // metric config
    s = config.GetMetricConfigEnableMonitor(opt.metric_enable_);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

    // cache config
    s = config.GetCacheConfigCacheInsertData(opt.insert_cache_immediately_);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

    int64_t insert_buffer_size = 1 * engine::GB;
    s = config.GetCacheConfigInsertBufferSize(insert_buffer_size);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }
    opt.insert_buffer_size_ = insert_buffer_size;

#if 1
    bool cluster_enable = false;
    std::string cluster_role;
    STATUS_CHECK(config.GetClusterConfigEnable(cluster_enable));
    STATUS_CHECK(config.GetClusterConfigRole(cluster_role));
    if (not cluster_enable) {
        opt.mode_ = engine::DBOptions::MODE::SINGLE;
    } else if (cluster_role == "ro") {
        opt.mode_ = engine::DBOptions::MODE::CLUSTER_READONLY;
    } else if (cluster_role == "rw") {
        opt.mode_ = engine::DBOptions::MODE::CLUSTER_WRITABLE;
    } else {
        std::cerr << "Error: cluster.role is not one of rw and ro." << std::endl;
        kill(0, SIGUSR1);
    }

#else
    std::string mode;
    s = config.GetServerConfigDeployMode(mode);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

    if (mode == "single") {
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
        opt.mode_ = engine::DBOptions::MODE::SINGLE;
    } else if (config.cluster.role() == ClusterRole::RO) {
        opt.mode_ = engine::DBOptions::MODE::CLUSTER_READONLY;
    } else if (config.cluster.role() == ClusterRole::RW) {
        opt.mode_ = engine::DBOptions::MODE::CLUSTER_WRITABLE;
    } else {
<<<<<<< HEAD
        std::cerr << "Error: cluster.role is not one of rw and ro." << std::endl;
=======
        std::cerr << "Error: server_config.deploy_mode in server_config.yaml is not one of "
                  << "single, cluster_readonly, and cluster_writable." << std::endl;
        kill(0, SIGUSR1);
    }
#endif

    // get wal configurations
    s = config.GetWalConfigEnable(opt.wal_enable_);
    if (!s.ok()) {
        std::cerr << "ERROR! Failed to get wal_enable configuration." << std::endl;
        std::cerr << s.ToString() << std::endl;
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
        kill(0, SIGUSR1);
    }

    // wal
    opt.wal_enable_ = config.wal.enable();
    if (opt.wal_enable_) {
<<<<<<< HEAD
        opt.wal_path_ = config.wal.path();
=======
        s = config.GetWalConfigRecoveryErrorIgnore(opt.recovery_error_ignore_);
        if (!s.ok()) {
            std::cerr << "ERROR! Failed to get recovery_error_ignore configuration." << std::endl;
            std::cerr << s.ToString() << std::endl;
            kill(0, SIGUSR1);
        }

        int64_t wal_buffer_size = 0;
        s = config.GetWalConfigBufferSize(wal_buffer_size);
        if (!s.ok()) {
            std::cerr << "ERROR! Failed to get buffer_size configuration." << std::endl;
            std::cerr << s.ToString() << std::endl;
            kill(0, SIGUSR1);
        }
        wal_buffer_size /= (1024 * 1024);
        opt.buffer_size_ = wal_buffer_size;

        s = config.GetWalConfigWalPath(opt.mxlog_path_);
        if (!s.ok()) {
            std::cerr << "ERROR! Failed to get mxlog_path configuration." << std::endl;
            std::cerr << s.ToString() << std::endl;
            kill(0, SIGUSR1);
        }
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
    }

    // transcript
    opt.transcript_enable_ = config.transcript.enable();
    opt.replay_script_path_ = config.transcript.replay();

    // create db root folder
    s = CommonUtil::CreateDirectory(opt.meta_.path_);
    if (!s.ok()) {
        std::cerr << "Error: Failed to create database primary path: " << path
                  << ". Possible reason: db_config.primary_path is wrong in milvus.yaml or not available." << std::endl;
        kill(0, SIGUSR1);
    }

    try {
        db_ = engine::DBFactory::BuildDB(opt);
        db_->Start();
    } catch (std::exception& ex) {
        std::cerr << "Error: failed to open database: " << ex.what()
                  << ". Possible reason: out of storage, meta schema is damaged "
                  << "or created by in-compatible Milvus version." << std::endl;
        kill(0, SIGUSR1);
    }

    // preload collection
<<<<<<< HEAD
    std::string preload_collections = config.cache.preload_collection();
=======
    std::string preload_collections;
    s = config.GetCacheConfigPreloadCollection(preload_collections);
    if (!s.ok()) {
        std::cerr << s.ToString() << std::endl;
        return s;
    }

>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
    s = PreloadCollections(preload_collections);
    if (!s.ok()) {
        std::cerr << "ERROR! Failed to preload collections: " << preload_collections << std::endl;
        std::cerr << s.ToString() << std::endl;
        kill(0, SIGUSR1);
    }

    return Status::OK();
}

Status
DBWrapper::StopService() {
    if (db_) {
        db_->Stop();
    }

    // SS TODO
    /* engine::snapshot::OperationExecutor::GetInstance().Stop(); */
    return Status::OK();
}

Status
DBWrapper::PreloadCollections(const std::string& preload_collections) {
    if (preload_collections.empty()) {
        // do nothing
    } else if (preload_collections == "*") {
        // load all collections
        std::vector<std::string> names;
        auto status = db_->ListCollections(names);
        if (!status.ok()) {
            return status;
        }

<<<<<<< HEAD
        for (auto& name : names) {
            std::vector<std::string> field_names;  // input empty field names will load all fileds
            auto status = db_->LoadCollection(nullptr, name, field_names);
=======
        for (auto& schema : table_schema_array) {
            auto status = db_->PreloadCollection(nullptr, schema.collection_id_);
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
            if (!status.ok()) {
                return status;
            }
        }
    } else {
        std::vector<std::string> collection_names;
        StringHelpFunctions::SplitStringByDelimeter(preload_collections, ",", collection_names);
        for (auto& name : collection_names) {
<<<<<<< HEAD
            std::vector<std::string> field_names;  // input empty field names will load all fileds
            auto status = db_->LoadCollection(nullptr, name, field_names);
=======
            auto status = db_->PreloadCollection(nullptr, name);
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
            if (!status.ok()) {
                return status;
            }
        }
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
