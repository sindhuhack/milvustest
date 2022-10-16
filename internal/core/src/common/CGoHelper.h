// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <string.h>

#include "common/type_c.h"

namespace milvus {

inline CStatus
SuccessCStatus(uint32_t cost) {
    return CStatus{Success, "", cost};
}

inline CStatus
SuccessCStatus() {
    return SuccessCStatus(-1);
}


inline CStatus
FailureCStatus(ErrorCode error_code, const std::string &str, uint32_t cost) {
    auto str_dup = strdup(str.c_str());
    return CStatus{error_code, str_dup, cost};
}
inline CStatus
FailureCStatus(ErrorCode error_code, const std::string &str) {
    return FailureCStatus(error_code, str, -1);
}

}  // namespace milvus
