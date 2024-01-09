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

#include "LogicalUnaryExpr.h"
#include "simd/hook.h"

namespace milvus {
namespace exec {

void
PhyLogicalUnaryExpr::Eval(EvalCtx& context, milvus::base::VectorPtr& result) {
    AssertInfo(inputs_.size() == 1,
               "logical unary expr must has one input, but now {}",
               inputs_.size());

    inputs_[0]->Eval(context, result);
    if (expr_->op_type_ == milvus::expr::LogicalUnaryExpr::OpType::LogicalNot) {
        auto flat_vec = GetColumnVector(result);
        bool* data = static_cast<bool*>(flat_vec->GetRawData());
#if defined(USE_DYNAMIC_SIMD)
        milvus::simd::invert_bool(data, flat_vec->size());
#else
        for (int i = 0; i < flat_vec->size(); ++i) {
            data[i] = !data[i];
        }
#endif
    }
}

}  //namespace exec
}  // namespace milvus
