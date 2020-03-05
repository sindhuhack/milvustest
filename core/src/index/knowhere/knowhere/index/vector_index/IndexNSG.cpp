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

#include "knowhere/index/vector_index/IndexNSG.h"

#include "knowhere/adapter/VectorAdapter.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Timer.h"

#ifdef MILVUS_GPU_VERSION

#include "knowhere/index/vector_index/IndexGPUIDMAP.h"
#include "knowhere/index/vector_index/IndexGPUIVF.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"

#endif

#include <fiu-local.h>

#include "knowhere/index/vector_index/IndexIDMAP.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/nsg/NSG.h"
#include "knowhere/index/vector_index/nsg/NSGIO.h"

namespace knowhere {

BinarySet
NSG::Serialize() {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        fiu_do_on("NSG.Serialize.throw_exception", throw std::exception());
        algo::NsgIndex* index = index_.get();

        MemoryIOWriter writer;
        algo::write_index(index, writer);
        auto data = std::make_shared<uint8_t>();
        data.reset(writer.data_);

        BinarySet res_set;
        res_set.Append("NSG", data, writer.total);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
NSG::Load(const BinarySet& index_binary) {
    try {
        fiu_do_on("NSG.Load.throw_exception", throw std::exception());
        auto binary = index_binary.GetByName("NSG");

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        auto index = algo::read_index(reader);
        index_.reset(index);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

DatasetPtr
NSG::Search(const DatasetPtr& dataset, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GETTENSOR(dataset)

    auto elems = rows * config[meta::TOPK].get<int64_t>();
    size_t p_id_size = sizeof(int64_t) * elems;
    size_t p_dist_size = sizeof(float) * elems;
    auto p_id = (int64_t*)malloc(p_id_size);
    auto p_dist = (float*)malloc(p_dist_size);

    algo::SearchParams s_params;
    s_params.search_length = config[IndexParams::search_length];
    index_->Search((float*)p_data, rows, dim, config[meta::TOPK].get<int64_t>(), p_dist, p_id, s_params);

    auto ret_ds = std::make_shared<Dataset>();
    ret_ds->Set(meta::IDS, p_id);
    ret_ds->Set(meta::DISTANCE, p_dist);
    return ret_ds;
}

IndexModelPtr
NSG::Train(const DatasetPtr& dataset, const Config& config) {
    auto idmap = std::make_shared<IDMAP>();
    idmap->Train(config);
    idmap->AddWithoutId(dataset, config);
    Graph knng;
    const float* raw_data = idmap->GetRawVectors();
#ifdef MILVUS_GPU_VERSION
    if (config[knowhere::meta::DEVICEID] == knowhere::INVALID_VALUE) {
        auto preprocess_index = std::make_shared<IVF>();
        auto model = preprocess_index->Train(dataset, config);
        preprocess_index->set_index_model(model);
        preprocess_index->AddWithoutIds(dataset, config);
        preprocess_index->GenGraph(raw_data, config[IndexParams::knng].get<int64_t>(), knng, config);
    } else {
        // TODO(linxj): use ivf instead?
        auto gpu_idx = cloner::CopyCpuToGpu(idmap, config[knowhere::meta::DEVICEID].get<int64_t>(), config);
        auto gpu_idmap = std::dynamic_pointer_cast<GPUIDMAP>(gpu_idx);
        gpu_idmap->GenGraph(raw_data, config[IndexParams::knng].get<int64_t>(), knng, config);
    }
#else
    auto preprocess_index = std::make_shared<IVF>();
    auto model = preprocess_index->Train(dataset, config);
    preprocess_index->set_index_model(model);
    preprocess_index->AddWithoutIds(dataset, config);
    preprocess_index->GenGraph(raw_data, config[IndexParams::knng].get<int64_t>(), knng, config);
#endif

    algo::BuildParams b_params;
    b_params.candidate_pool_size = config[IndexParams::candidate];
    b_params.out_degree = config[IndexParams::out_degree];
    b_params.search_length = config[IndexParams::search_length];

    auto p_ids = dataset->Get<const int64_t*>(meta::IDS);

    GETTENSOR(dataset)
    index_ = std::make_shared<algo::NsgIndex>(dim, rows);
    index_->SetKnnGraph(knng);
    index_->Build_with_ids(rows, (float*)p_data, (int64_t*)p_ids, b_params);
    return nullptr;  // TODO(linxj): support serialize
}

void
NSG::Add(const DatasetPtr& dataset, const Config& config) {
    // do nothing
}

int64_t
NSG::Count() {
    return index_->ntotal;
}

int64_t
NSG::Dimension() {
    return index_->dimension;
}

// VectorIndexPtr
// NSG::Clone() {
//    KNOWHERE_THROW_MSG("not support");
//}

void
NSG::Seal() {
    // do nothing
}

}  // namespace knowhere
