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

package storage

import (
	"errors"

	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type VectorChunkManager struct {
	localChunkManager  ChunkManager
	remoteChunkManager ChunkManager
}

func NewVectorChunkManager(localChunkManager ChunkManager, remoteChunkManager ChunkManager) *VectorChunkManager {
	return &VectorChunkManager{
		localChunkManager:  localChunkManager,
		remoteChunkManager: remoteChunkManager,
	}
}

func (vcm *VectorChunkManager) DownloadVectorFile(key string, schema *etcdpb.CollectionMeta) error {
	insertCodec := NewInsertCodec(schema)
	content, err := vcm.remoteChunkManager.Read(key)
	if err != nil {
		return err
	}
	blob := &Blob{
		Key:   key,
		Value: content,
	}

	_, _, data, err := insertCodec.Deserialize([]*Blob{blob})
	if err != nil {
		return err
	}

	for _, singleData := range data.Data {
		binaryVector, ok := singleData.(*BinaryVectorFieldData)
		if ok {
			vcm.localChunkManager.Write(key, binaryVector.Data)
		}
		floatVector, ok := singleData.(*FloatVectorFieldData)
		if ok {
			floatData := floatVector.Data
			result := make([]byte, 0)
			for _, singleFloat := range floatData {
				result = append(result, typeutil.Float32ToByte(singleFloat)...)
			}
			vcm.localChunkManager.Write(key, result)
		}
	}
	insertCodec.Close()
	return nil
}

func (vcm *VectorChunkManager) GetPath(key string) (string, error) {
	if vcm.localChunkManager.Exist(key) {
		return vcm.localChunkManager.GetPath(key)
	}
	return vcm.localChunkManager.GetPath(key)
}

func (vcm *VectorChunkManager) Write(key string, content []byte) error {
	return vcm.localChunkManager.Write(key, content)
}

func (vcm *VectorChunkManager) Exist(key string) bool {
	return vcm.localChunkManager.Exist(key)
}

func (vcm *VectorChunkManager) Read(key string) ([]byte, error) {
	if vcm.localChunkManager.Exist(key) {
		return vcm.localChunkManager.Read(key)
	}
	return nil, errors.New("the vector file doesn't exist, please call download first")
}

func (vcm *VectorChunkManager) ReadAt(key string, p []byte, off int64) (n int, err error) {
	return vcm.localChunkManager.ReadAt(key, p, off)
}
