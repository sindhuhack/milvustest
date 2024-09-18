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

package indexparamcheck

/*
#cgo pkg-config: milvus_core

#include <stdlib.h>	// free
#include "segcore/vector_index_c.h"
*/
import "C"

import (
	"bytes"
	"fmt"
	"sync"
	"unsafe"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/log"
)

const (
	BinaryFlag        uint64 = 1 << 0
	Float32Flag       uint64 = 1 << 1
	Float16Flag       uint64 = 1 << 2
	BFloat16Flag      uint64 = 1 << 3
	SparseFloat32Flag uint64 = 1 << 4

	// BFFlag This flag indicates that there is no need to create any index structure
	BFFlag uint64 = 1 << 16
	// KNNFlag This flag indicates that the index defaults to KNN search, meaning the recall rate is 100%
	KNNFlag uint64 = 1 << 17
	// GpuFlag This flag indicates that the index is deployed on GPU (need GPU devices)
	GpuFlag uint64 = 1 << 18
	// MmapFlag This flag indicates that the index support using mmap manage its mainly memory, which can significant improve the capacity
	MmapFlag uint64 = 1 << 19
	// MvFlag This flag indicates that the index support using materialized view to accelerate filtering search
	MvFlag uint64 = 1 << 20
	// DiskFlag This flag indicates that the index need disk
	DiskFlag uint64 = 1 << 21
)

type VecIndexMgr interface {
	init() error

	IsBinarySupport(indexType IndexType) bool
	IsFlat32Support(indexType IndexType) bool
	IsFlat16Support(indexType IndexType) bool
	IsBFlat16Support(indexType IndexType) bool
	IsSparseFloat32Support(indexType IndexType) bool
	IsDataTypeSupport(indexType IndexType, dataType schemapb.DataType) bool

	IsFlatVecIndex(indexType IndexType) bool
	IsBruteForce(indexType IndexType) bool
	IsVecIndex(indexType IndexType) bool
	IsDiskANN(indexType IndexType) bool
	IsGPUVecIndex(indexType IndexType) bool
	IsDiskVecIndex(indexType IndexType) bool
	IsMMapSupported(indexType IndexType) bool
	IsMvSupported(indexType IndexType) bool
	GetMemoryRatio(indexType IndexType) float64
	GetDiskRatio(indexType IndexType) float64
}

type VecIndexMgrImpl struct {
	features map[string]uint64
	once     sync.Once
}

func (mgr *VecIndexMgrImpl) IsBruteForce(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & BFFlag) == BFFlag
}

func (mgr *VecIndexMgrImpl) IsDiskANN(indexType IndexType) bool {
	return indexType == "DISKANN"
}

func (mgr *VecIndexMgrImpl) init() error {
	size := int(C.GetIndexListSize())
	if size == 0 {
		return fmt.Errorf("vector index list is null")
	}
	vecIndexList := make([]unsafe.Pointer, size)
	vecIndexFeatures := make([]uint64, size)

	C.GetIndexFeatures(unsafe.Pointer(&vecIndexList[0]), (*C.uint64_t)(unsafe.Pointer(&vecIndexFeatures[0])))
	mgr.features = make(map[string]uint64)
	var featureLog bytes.Buffer
	for i := 0; i < size; i++ {
		key := C.GoString((*C.char)(vecIndexList[i]))
		mgr.features[key] = vecIndexFeatures[i]
		featureLog.WriteString(key + " : " + fmt.Sprintf("%d", vecIndexFeatures[i]) + ",")
	}
	log.Info("init vecIndex features : " + featureLog.String())
	return nil
}

func (mgr *VecIndexMgrImpl) IsBinarySupport(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & BinaryFlag) == BinaryFlag
}

func (mgr *VecIndexMgrImpl) IsFlat32Support(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & Float32Flag) == Float32Flag
}

func (mgr *VecIndexMgrImpl) IsFlat16Support(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & Float16Flag) == Float16Flag
}

func (mgr *VecIndexMgrImpl) IsBFlat16Support(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & BFloat16Flag) == BFloat16Flag
}

func (mgr *VecIndexMgrImpl) IsSparseFloat32Support(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & SparseFloat32Flag) == SparseFloat32Flag
}

func (mgr *VecIndexMgrImpl) IsDataTypeSupport(indexType IndexType, dataType schemapb.DataType) bool {
	if dataType == schemapb.DataType_BinaryVector {
		return mgr.IsBinarySupport(indexType)
	} else if dataType == schemapb.DataType_FloatVector {
		return mgr.IsFlat32Support(indexType)
	} else if dataType == schemapb.DataType_BFloat16Vector {
		return mgr.IsBFlat16Support(indexType)
	} else if dataType == schemapb.DataType_Float16Vector {
		return mgr.IsFlat16Support(indexType)
	} else if dataType == schemapb.DataType_SparseFloatVector {
		return mgr.IsSparseFloat32Support(indexType)
	} else {
		return false
	}
}

func (mgr *VecIndexMgrImpl) IsFlatVecIndex(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & KNNFlag) == KNNFlag
}

func (mgr *VecIndexMgrImpl) IsMvSupported(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & MvFlag) == MvFlag
}

func (mgr *VecIndexMgrImpl) GetMemoryRatio(indexType IndexType) float64 {
	if indexType == "DISKANN" {
		return 0.25
	}
	return 1.0
}

func (mgr *VecIndexMgrImpl) GetDiskRatio(indexType IndexType) float64 {
	return 1.0
}

func (mgr *VecIndexMgrImpl) IsGPUVecIndex(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & GpuFlag) == GpuFlag
}

func (mgr *VecIndexMgrImpl) IsMMapSupported(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & MmapFlag) == MmapFlag
}

func (mgr *VecIndexMgrImpl) IsVecIndex(indexType IndexType) bool {
	_, ok := mgr.features[indexType]
	return ok
}

func (mgr *VecIndexMgrImpl) IsDiskVecIndex(indexType IndexType) bool {
	feature, ok := mgr.features[indexType]
	if !ok {
		return false
	}
	return (feature & DiskFlag) == DiskFlag
}

func newVecIndexMgr() *VecIndexMgrImpl {
	log.Info("newVecIndexMgr")
	mgr := &VecIndexMgrImpl{}
	mgr.init()
	return mgr
}

var vecIndexMgr VecIndexMgr

var getVecIndexMgrOnce sync.Once

// GetVecIndexMgrInstance gets the instance of VecIndexMgrInstance.
func GetVecIndexMgrInstance() VecIndexMgr {
	getVecIndexMgrOnce.Do(func() {
		vecIndexMgr = newVecIndexMgr()
	})
	return vecIndexMgr
}
