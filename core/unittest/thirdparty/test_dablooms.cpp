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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "dablooms/dablooms.h"
#include "utils.h"

TEST_F(DabloomsTest, CORRECTNESS_TEST) {
    scaling_bloom_t* bloom;
    int32_t i;
    struct stats results = {0};

    bloom = new_scaling_bloom(2 * CAPACITY, ERROR_RATE);
    ASSERT_NE(bloom, nullptr);

    auto start = std::chrono::system_clock::now();

    for (i = 0; i < 2 * CAPACITY; i++) {
        if (i % 2 == 0) {
            std::string tmp = std::to_string(i);
            scaling_bloom_add(bloom, tmp.c_str(), tmp.size(), i);
        }
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time costs for add: "
              << double(duration.count()) * std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << " s" << std::endl;
    start = std::chrono::system_clock::now();

    for (i = 0; i < 2 * CAPACITY; i++) {
        if (i % 2 == 1) {
            std::string tmp = std::to_string(i);
            bloom_score(scaling_bloom_check(bloom, tmp.c_str(), tmp.size()), 0, &results);
        }
    }

    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time costs for check: "
              << double(duration.count()) * std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << " s" << std::endl;

    printf(
        "Elements added:   %6d"
        "\n"
        "Elements checked: %6d"
        "\n"
        "Total size: %d KiB"
        "\n\n",
        (i + 1) / 2, i / 2, (int)bloom->num_bytes / 1024);

    free_scaling_bloom(bloom);

    print_results(&results);
}

TEST_F(DabloomsTest, FILE_OPT_TEST) {
    scaling_bloom_t* bloom;
    int i, key_removed;
    struct stats results1 = {0};
    struct stats results2 = {0};

    bloom = new_scaling_bloom(CAPACITY, ERROR_RATE);
    ASSERT_NE(bloom, nullptr);

    auto start = std::chrono::system_clock::now();

    for (i = 0; i < CAPACITY; i++) {
        std::string tmp = std::to_string(i);
        scaling_bloom_add(bloom, tmp.c_str(), tmp.size(), i);
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << std::endl
              << "Time costs for add: "
              << double(duration.count()) * std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << " s" << std::endl;
    start = std::chrono::system_clock::now();

    for (i = 0; i < CAPACITY; i++) {
        if (i % 5 == 0) {
            std::string tmp = std::to_string(i);
            scaling_bloom_remove(bloom, tmp.c_str(), tmp.size(), i);
        }
    }

    for (i = 0; i < CAPACITY; i++) {
        std::string tmp = std::to_string(i);
        key_removed = (i % 5 == 0);
        bloom_score(scaling_bloom_check(bloom, tmp.c_str(), tmp.size()), !key_removed, &results1);
    }

    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time costs for remove: "
              << double(duration.count()) * std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << " s" << std::endl;

    bitmap_t* bitmap = bloom->bitmap;
    bloom->bitmap = nullptr;
    free_scaling_bloom(bloom);

    print_results(&results1);

    // new_scaling_bloom_from_bitmap
    scaling_bloom_t* bloom_copy = new_scaling_bloom_from_bitmap(CAPACITY, ERROR_RATE, bitmap);

    for (i = 0; i < CAPACITY; i++) {
        std::string tmp = std::to_string(i);
        key_removed = (i % 5 == 0);
        bloom_score(scaling_bloom_check(bloom_copy, tmp.c_str(), tmp.size()), !key_removed, &results2);
    }

    free_scaling_bloom(bloom_copy);

    print_results(&results2);
}

TEST_F(DabloomsTest, CNT_LARGER_THAN_CAP) {
    scaling_bloom_t* bloom;
    int i, key_removed;
    struct stats results1 = {0};
    struct stats results2 = {0};

    bloom = new_scaling_bloom(CAPACITY / 4, ERROR_RATE);
    ASSERT_NE(bloom, nullptr);

    auto start = std::chrono::system_clock::now();

    for (i = 0; i < CAPACITY; i++) {
        std::string tmp = std::to_string(i);
        scaling_bloom_add(bloom, tmp.c_str(), tmp.size(), i);
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << std::endl
              << "Time costs for add: "
              << double(duration.count()) * std::chrono::microseconds::period::num /
                 std::chrono::microseconds::period::den
              << " s" << std::endl;
    start = std::chrono::system_clock::now();

    for (i = 0; i < CAPACITY; i++) {
        if (i % 5 == 0) {
            std::string tmp = std::to_string(i);
            scaling_bloom_remove(bloom, tmp.c_str(), tmp.size(), i);
        }
    }

    for (i = 0; i < CAPACITY; i++) {
        std::string tmp = std::to_string(i);
        key_removed = (i % 5 == 0);
        bloom_score(scaling_bloom_check(bloom, tmp.c_str(), tmp.size()), !key_removed, &results1);
    }

    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time costs for remove: "
              << double(duration.count()) * std::chrono::microseconds::period::num /
                 std::chrono::microseconds::period::den
              << " s" << std::endl;

    bitmap_t* bitmap = bloom->bitmap;
    bloom->bitmap = nullptr;
    free_scaling_bloom(bloom);

    print_results(&results1);

    // new_scaling_bloom_from_bitmap
    scaling_bloom_t* bloom_copy = new_scaling_bloom_from_bitmap(CAPACITY / 4, ERROR_RATE, bitmap);

    for (i = 0; i < CAPACITY; i++) {
        std::string tmp = std::to_string(i);
        key_removed = (i % 5 == 0);
        bloom_score(scaling_bloom_check(bloom_copy, tmp.c_str(), tmp.size()), !key_removed, &results2);
    }

    free_scaling_bloom(bloom_copy);

    print_results(&results2);
}