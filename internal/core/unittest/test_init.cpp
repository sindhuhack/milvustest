// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>

#include "segcore/segcore_init_c.h"
#include "test_utils/SingletonConfigRestorer.h"

TEST(Init, Naive) {
    using namespace milvus;
    using namespace milvus::segcore;

    milvus::test::SingletonConfigRestorer restorer;

    SegcoreInit(nullptr);
    SegcoreSetChunkRows(32768);

    auto simd_type = SegcoreSetSimdType("auto");
    free(simd_type);

    auto& cfg = SingletonConfig::GetInstance();

    SegcoreSetEnableParallelReduce(true);
    SegcoreSetNqThresholdToEnableParallelReduce(1000);
    SegcoreSetKThresholdToEnableParallelReduce(10000);

    ASSERT_TRUE(cfg.is_enable_parallel_reduce());
    ASSERT_EQ(cfg.get_nq_threshold_to_enable_parallel_reduce(), 1000);
    ASSERT_EQ(cfg.get_k_threshold_to_enable_parallel_reduce(), 10000);
}
