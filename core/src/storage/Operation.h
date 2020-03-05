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

#include <boost/filesystem.hpp>
#include <memory>
#include <string>
#include <vector>

namespace milvus {
namespace storage {

class Operation {
 public:
    explicit Operation(const std::string& dir_path) : dir_path_(dir_path) {
    }

    virtual void
    CreateDirectory() = 0;

    virtual void
    ListDirectory(std::vector<std::string>& file_paths) const = 0;

    const std::string&
    GetDirectory() const {
        return dir_path_;
    }

    const std::string&
    GetFileName(const std::string& file_path) const {
        return boost::filesystem::path(file_path).stem().string();
    }

    const std::string&
    GetFileExt(const std::string& file_path) const {
        return boost::filesystem::path(file_path).extension().string();
    }

    virtual bool
    DeleteFile(const std::string& file_path) = 0;

    virtual void
    CopyFile(const std::string& from_name, const std::string& to_name) = 0;

    virtual void
    RenameFile(const std::string& old_name, const std::string& new_name) = 0;

    // TODO(zhiru):
    //  open(), sync(), close()
    //  function that opens a stream for reading file
    //  function that creates a new, empty file and returns an stream for appending data to this file
    //  function that creates a new, empty, temporary file and returns an stream for appending data to this file

 private:
    const std::string dir_path_;
};

using OperationPtr = std::shared_ptr<Operation>;

}  // namespace storage
}  // namespace milvus
