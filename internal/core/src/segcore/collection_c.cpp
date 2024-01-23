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

#ifdef __linux__
#include <malloc.h>
#endif

#include <iostream>
#include "segcore/collection_c.h"
#include "segcore/Collection.h"

CCollection
NewCollection(const void* schema_proto_blob, const int64_t length) {
    auto collection = std::make_unique<milvus::segcore::Collection>(
        schema_proto_blob, length);
    return (void*)collection.release();
}

void
SetIndexMeta(CCollection collection,
             const void* proto_blob,
             const int64_t length) {
    auto col = (milvus::segcore::Collection*)collection;
    col->parseIndexMeta(proto_blob, length);
}

void
DeleteCollection(CCollection collection) {
    auto col = (milvus::segcore::Collection*)collection;
    delete col;
}

const char*
GetCollectionName(CCollection collection) {
    auto col = (milvus::segcore::Collection*)collection;
    return strdup(col->get_collection_name().data());
}

const char*
GetMetricTypeByFieldID(CCollection collection, const int64_t field_id) {
    auto col = (milvus::segcore::Collection*)collection;
    auto col_meta = col->GetIndexMeta();
    auto field_meta = col_meta->GetFieldIndexMeta(milvus::FieldId(field_id));
    return strdup(field_meta.GeMetricType().c_str());
}
