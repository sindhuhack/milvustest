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

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <cstring>
#include <memory>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common/File.h"
#include "common/Types.h"
#include "common/EasyAssert.h"
#include "common/Exception.h"
#include "common/Utils.h"
#include "common/Slice.h"
#include "index/StringIndexMarisa.h"
#include "index/Utils.h"
#include "index/Index.h"
#include "storage/Util.h"
#include "storage/space.h"
#include "storage/LocalChunkManagerSingleton.h"

namespace milvus::index {

StringIndexMarisa::StringIndexMarisa(
    const storage::FileManagerContext& file_manager_context) {
    if (file_manager_context.Valid()) {
        file_manager_ =
            std::make_shared<storage::MemFileManagerImpl>(file_manager_context);
    }
}

StringIndexMarisa::StringIndexMarisa(
    const storage::FileManagerContext& file_manager_context,
    std::shared_ptr<milvus_storage::Space> space)
    : space_(space) {
    if (file_manager_context.Valid()) {
        file_manager_ = std::make_shared<storage::MemFileManagerImpl>(
            file_manager_context, space_);
    }
}

int64_t
StringIndexMarisa::Size() {
    return trie_.size();
}

bool
valid_str_id(size_t str_id) {
    return str_id >= 0 && str_id != MARISA_INVALID_KEY_ID;
}

void
StringIndexMarisa::BuildV2(const Config& config) {
    if (built_) {
        throw std::runtime_error("index has been built");
    }
    auto field_name = file_manager_->GetIndexMeta().field_name;
    auto reader = space_->ScanData();
    std::vector<FieldDataPtr> field_datas;
    for (auto rec = reader->Next(); rec != nullptr; rec = reader->Next()) {
        if (!rec.ok()) {
            PanicInfo(DataFormatBroken, "failed to read data");
        }
        auto data = rec.ValueUnsafe();
        auto total_num_rows = data->num_rows();
        auto col_data = data->GetColumnByName(field_name);
        auto field_data =
            storage::CreateFieldData(DataType::STRING, 0, total_num_rows);
        field_data->FillFieldData(col_data);
        field_datas.push_back(field_data);
    }
    int64_t total_num_rows = 0;

    // fill key set.
    marisa::Keyset keyset;
    for (auto data : field_datas) {
        auto slice_num = data->get_num_rows();
        for (size_t i = 0; i < slice_num; ++i) {
            keyset.push_back(
                (*static_cast<const std::string*>(data->RawValue(i))).c_str());
        }
        total_num_rows += slice_num;
    }
    trie_.build(keyset);

    // fill str_ids_
    str_ids_.resize(total_num_rows);
    int64_t offset = 0;
    for (auto data : field_datas) {
        auto slice_num = data->get_num_rows();
        for (size_t i = 0; i < slice_num; ++i) {
            auto str_id =
                lookup(*static_cast<const std::string*>(data->RawValue(i)));
            AssertInfo(valid_str_id(str_id), "invalid marisa key");
            str_ids_[offset++] = str_id;
        }
    }

    // fill str_ids_to_offsets_
    fill_offsets();

    built_ = true;
}
void
StringIndexMarisa::Build(const Config& config) {
    if (built_) {
        throw SegcoreError(IndexAlreadyBuild, "index has been built");
    }

    auto insert_files =
        GetValueFromConfig<std::vector<std::string>>(config, "insert_files");
    AssertInfo(insert_files.has_value(),
               "insert file paths is empty when build index");
    auto field_datas =
        file_manager_->CacheRawDataToMemory(insert_files.value());
    int64_t total_num_rows = 0;

    // fill key set.
    marisa::Keyset keyset;
    for (const auto& data : field_datas) {
        auto slice_num = data->get_num_rows();
        for (int64_t i = 0; i < slice_num; ++i) {
            keyset.push_back(
                (*static_cast<const std::string*>(data->RawValue(i))).c_str());
        }
        total_num_rows += slice_num;
    }
    trie_.build(keyset);

    // fill str_ids_
    str_ids_.resize(total_num_rows);
    int64_t offset = 0;
    for (const auto& data : field_datas) {
        auto slice_num = data->get_num_rows();
        for (int64_t i = 0; i < slice_num; ++i) {
            auto str_id =
                lookup(*static_cast<const std::string*>(data->RawValue(i)));
            AssertInfo(valid_str_id(str_id), "invalid marisa key");
            str_ids_[offset++] = str_id;
        }
    }

    // fill str_ids_to_offsets_
    fill_offsets();

    built_ = true;
}

void
StringIndexMarisa::Build(size_t n, const std::string* values) {
    if (built_) {
        throw SegcoreError(IndexAlreadyBuild, "index has been built");
    }

    marisa::Keyset keyset;
    {
        // fill key set.
        for (size_t i = 0; i < n; i++) {
            keyset.push_back(values[i].c_str());
        }
    }

    trie_.build(keyset);
    fill_str_ids(n, values);
    fill_offsets();

    built_ = true;
}

BinarySet
StringIndexMarisa::Serialize(const Config& config) {
    auto uuid = boost::uuids::random_generator()();
    auto uuid_string = boost::uuids::to_string(uuid);
    auto file = std::string("/tmp/") + uuid_string;

    auto fd = open(
        file.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IXUSR);
    AssertInfo(fd != -1, "open file failed");
    trie_.write(fd);

    auto size = get_file_size(fd);
    auto index_data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    ReadDataFromFD(fd, index_data.get(), size);

    close(fd);
    remove(file.c_str());

    auto str_ids_len = str_ids_.size() * sizeof(size_t);
    std::shared_ptr<uint8_t[]> str_ids(new uint8_t[str_ids_len]);
    memcpy(str_ids.get(), str_ids_.data(), str_ids_len);

    BinarySet res_set;
    res_set.Append(MARISA_TRIE_INDEX, index_data, size);
    res_set.Append(MARISA_STR_IDS, str_ids, str_ids_len);

    Disassemble(res_set);

    return res_set;
}

BinarySet
StringIndexMarisa::Upload(const Config& config) {
    auto binary_set = Serialize(config);
    file_manager_->AddFile(binary_set);

    auto remote_paths_to_size = file_manager_->GetRemotePathsToFileSize();
    BinarySet ret;
    for (auto& file : remote_paths_to_size) {
        ret.Append(file.first, nullptr, file.second);
    }

    return ret;
}

BinarySet
StringIndexMarisa::UploadV2(const Config& config) {
    auto binary_set = Serialize(config);
    file_manager_->AddFileV2(binary_set);

    auto remote_paths_to_size = file_manager_->GetRemotePathsToFileSize();
    BinarySet ret;
    for (auto& file : remote_paths_to_size) {
        ret.Append(file.first, nullptr, file.second);
    }

    return ret;
}

void
StringIndexMarisa::LoadWithoutAssemble(const BinarySet& set,
                                       const Config& config) {
    auto uuid = boost::uuids::random_generator()();
    auto uuid_string = boost::uuids::to_string(uuid);
    auto file_name = std::string("/tmp/") + uuid_string;

    auto index = set.GetByName(MARISA_TRIE_INDEX);
    auto len = index->size;

    auto file = File::Open(file_name, O_RDWR | O_CREAT | O_EXCL);
    try {
        auto written = file.Write(index->data.get(), len);
    } catch (const SegcoreError& e) {
        file.Close();
        remove(file_name.c_str());
        throw;
    }

    file.Seek(0, SEEK_SET);
    auto local_chunk_manager =
        milvus::storage::LocalChunkManagerSingleton::GetInstance()
            .GetChunkManager();
    auto index_size = local_chunk_manager->Size(file_name);
    if (config.contains(kEnableMmap)) {
        trie_.mmap(file_name.c_str());
        resource_usage_.disk_size += index_size;
    } else {
        trie_.read(file.Descriptor());
        resource_usage_.mem_size += index_size;
    }
    // make sure the file would be removed after we unmap & close it
    unlink(file_name.c_str());

    auto str_ids = set.GetByName(MARISA_STR_IDS);
    auto str_ids_len = str_ids->size;
    str_ids_.resize(str_ids_len / sizeof(size_t));
    memcpy(str_ids_.data(), str_ids->data.get(), str_ids_len);

    fill_offsets();
}

void
StringIndexMarisa::Load(const BinarySet& set, const Config& config) {
    milvus::Assemble(const_cast<BinarySet&>(set));
    LoadWithoutAssemble(set, config);
}

void
StringIndexMarisa::Load(milvus::tracer::TraceContext ctx,
                        const Config& config) {
    auto index_files =
        GetValueFromConfig<std::vector<std::string>>(config, "index_files");
    AssertInfo(index_files.has_value(),
               "index file paths is empty when load index");
    auto index_datas = file_manager_->LoadIndexToMemory(index_files.value());
    AssembleIndexDatas(index_datas);
    BinarySet binary_set;
    for (auto& [key, data] : index_datas) {
        auto size = data->Size();
        auto deleter = [&](uint8_t*) {};  // avoid repeated deconstruction
        auto buf = std::shared_ptr<uint8_t[]>(
            (uint8_t*)const_cast<void*>(data->Data()), deleter);
        binary_set.Append(key, buf, size);
    }

    LoadWithoutAssemble(binary_set, config);
}

void
StringIndexMarisa::LoadV2(const Config& config) {
    auto blobs = space_->StatisticsBlobs();
    std::vector<std::string> index_files;
    auto prefix = file_manager_->GetRemoteIndexObjectPrefixV2();
    for (auto& b : blobs) {
        if (b.name.rfind(prefix, 0) == 0) {
            index_files.push_back(b.name);
        }
    }
    std::map<std::string, FieldDataPtr> index_datas{};
    for (auto& file_name : index_files) {
        auto res = space_->GetBlobByteSize(file_name);
        if (!res.ok()) {
            PanicInfo(DataFormatBroken, "unable to read index blob");
        }
        auto index_blob_data =
            std::shared_ptr<uint8_t[]>(new uint8_t[res.value()]);
        auto status = space_->ReadBlob(file_name, index_blob_data.get());
        if (!status.ok()) {
            PanicInfo(DataFormatBroken, "unable to read index blob");
        }
        auto raw_index_blob =
            storage::DeserializeFileData(index_blob_data, res.value());
        index_datas[file_name] = raw_index_blob->GetFieldData();
    }
    AssembleIndexDatas(index_datas);
    BinarySet binary_set;
    for (auto& [key, data] : index_datas) {
        auto size = data->Size();
        auto deleter = [&](uint8_t*) {};  // avoid repeated deconstruction
        auto buf = std::shared_ptr<uint8_t[]>(
            (uint8_t*)const_cast<void*>(data->Data()), deleter);
        auto file_name = key.substr(key.find_last_of('/') + 1);
        binary_set.Append(file_name, buf, size);
    }

    LoadWithoutAssemble(binary_set, config);
}

const TargetBitmap
StringIndexMarisa::In(size_t n, const std::string* values) {
    TargetBitmap bitset(str_ids_.size());
    for (size_t i = 0; i < n; i++) {
        auto str = values[i];
        auto str_id = lookup(str);
        if (valid_str_id(str_id)) {
            auto offsets = str_ids_to_offsets_[str_id];
            for (auto offset : offsets) {
                bitset[offset] = true;
            }
        }
    }
    return bitset;
}

const TargetBitmap
StringIndexMarisa::NotIn(size_t n, const std::string* values) {
    TargetBitmap bitset(str_ids_.size(), true);
    for (size_t i = 0; i < n; i++) {
        auto str = values[i];
        auto str_id = lookup(str);
        if (valid_str_id(str_id)) {
            auto offsets = str_ids_to_offsets_[str_id];
            for (auto offset : offsets) {
                bitset[offset] = false;
            }
        }
    }
    return bitset;
}

const TargetBitmap
StringIndexMarisa::Range(std::string value, OpType op) {
    auto count = Count();
    TargetBitmap bitset(count);
    std::vector<size_t> ids;
    marisa::Agent agent;
    switch (op) {
        case OpType::GreaterThan: {
            while (trie_.predictive_search(agent)) {
                auto key = std::string(agent.key().ptr(), agent.key().length());
                if (key > value) {
                    ids.push_back(agent.key().id());
                    break;
                }
            };
            while (trie_.predictive_search(agent)) {
                ids.push_back(agent.key().id());
            }
            break;
        }
        case OpType::GreaterEqual: {
            while (trie_.predictive_search(agent)) {
                auto key = std::string(agent.key().ptr(), agent.key().length());
                if (key >= value) {
                    ids.push_back(agent.key().id());
                    break;
                }
            }
            while (trie_.predictive_search(agent)) {
                ids.push_back(agent.key().id());
            }
            break;
        }
        case OpType::LessThan: {
            while (trie_.predictive_search(agent)) {
                auto key = std::string(agent.key().ptr(), agent.key().length());
                if (key >= value) {
                    break;
                }
                ids.push_back(agent.key().id());
            }
            break;
        }
        case OpType::LessEqual: {
            while (trie_.predictive_search(agent)) {
                auto key = std::string(agent.key().ptr(), agent.key().length());
                if (key > value) {
                    break;
                }
                ids.push_back(agent.key().id());
            }
            break;
        }
        default:
            throw SegcoreError(
                OpTypeInvalid,
                fmt::format("Invalid OperatorType: {}", static_cast<int>(op)));
    }

    for (const auto str_id : ids) {
        auto offsets = str_ids_to_offsets_[str_id];
        for (auto offset : offsets) {
            bitset[offset] = true;
        }
    }
    return bitset;
}

const TargetBitmap
StringIndexMarisa::Range(std::string lower_bound_value,
                         bool lb_inclusive,
                         std::string upper_bound_value,
                         bool ub_inclusive) {
    auto count = Count();
    TargetBitmap bitset(count);
    if (lower_bound_value.compare(upper_bound_value) > 0 ||
        (lower_bound_value.compare(upper_bound_value) == 0 &&
         !(lb_inclusive && ub_inclusive))) {
        return bitset;
    }

    auto common_prefix = GetCommonPrefix(lower_bound_value, upper_bound_value);
    marisa::Agent agent;
    agent.set_query(common_prefix.c_str());
    std::vector<size_t> ids;
    while (trie_.predictive_search(agent)) {
        std::string_view val =
            std::string_view(agent.key().ptr(), agent.key().length());
        if (val > upper_bound_value ||
            (!ub_inclusive && val == upper_bound_value)) {
            break;
        }

        if (val < lower_bound_value ||
            (!lb_inclusive && val == lower_bound_value)) {
            continue;
        }

        if (((lb_inclusive && lower_bound_value <= val) ||
             (!lb_inclusive && lower_bound_value < val)) &&
            ((ub_inclusive && val <= upper_bound_value) ||
             (!ub_inclusive && val < upper_bound_value))) {
            ids.push_back(agent.key().id());
        }
    }
    for (const auto str_id : ids) {
        auto offsets = str_ids_to_offsets_[str_id];
        for (auto offset : offsets) {
            bitset[offset] = true;
        }
    }

    return bitset;
}

const TargetBitmap
StringIndexMarisa::PrefixMatch(std::string_view prefix) {
    TargetBitmap bitset(str_ids_.size());
    auto matched = prefix_match(prefix);
    for (const auto str_id : matched) {
        auto offsets = str_ids_to_offsets_[str_id];
        for (auto offset : offsets) {
            bitset[offset] = true;
        }
    }
    return bitset;
}

void
StringIndexMarisa::fill_str_ids(size_t n, const std::string* values) {
    str_ids_.resize(n);
    for (size_t i = 0; i < n; i++) {
        auto str = values[i];
        auto str_id = lookup(str);
        AssertInfo(valid_str_id(str_id), "invalid marisa key");
        str_ids_[i] = str_id;
    }
}

void
StringIndexMarisa::fill_offsets() {
    for (size_t offset = 0; offset < str_ids_.size(); offset++) {
        auto str_id = str_ids_[offset];
        if (str_ids_to_offsets_.find(str_id) == str_ids_to_offsets_.end()) {
            str_ids_to_offsets_[str_id] = std::vector<size_t>{};
        }
        str_ids_to_offsets_[str_id].push_back(offset);
    }
}

size_t
StringIndexMarisa::lookup(const std::string_view str) {
    marisa::Agent agent;
    agent.set_query(str.data());
    if (trie_.lookup(agent)) {
        return agent.key().id();
    }

    // not found the string in trie
    return MARISA_INVALID_KEY_ID;
}

std::vector<size_t>
StringIndexMarisa::prefix_match(const std::string_view prefix) {
    std::vector<size_t> ret;
    marisa::Agent agent;
    agent.set_query(prefix.data());
    while (trie_.predictive_search(agent)) {
        ret.push_back(agent.key().id());
    }
    return ret;
}

std::string
StringIndexMarisa::Reverse_Lookup(size_t offset) const {
    AssertInfo(offset < str_ids_.size(), "out of range of total count");
    marisa::Agent agent;
    agent.set_query(str_ids_[offset]);
    trie_.reverse_lookup(agent);
    return std::string(agent.key().ptr(), agent.key().length());
}

ResourceUsage
StringIndexMarisa::GetResourceUsage() const {
    return resource_usage_;
}

}  // namespace milvus::index
