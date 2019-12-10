// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include "server/delivery/request/BaseRequest.h"

namespace milvus {
namespace server {

class CreateTableRequest : public BaseRequest {
 public:
    static BaseRequestPtr
    Create(const std::string& table_name,
           int64_t dimension,
           int32_t index_file_size,
           int32_t metric_type);

 protected:
    explicit CreateTableRequest(const std::string& table_name,
                                int64_t dimension,
                                int32_t index_file_size,
                                int32_t metric_type);

    Status
    OnExecute() override;

 private:
    const std::string table_name_;
    int64_t dimension_;
    int32_t index_file_size_;
    int32_t metric_type_;
};

}  // namespace server
}  // namespace milvus
