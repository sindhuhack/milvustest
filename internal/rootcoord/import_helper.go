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

package rootcoord

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"go.uber.org/zap"
)

type GetCollectionNameFunc func(collID, partitionID UniqueID) (string, string, error)
type IDAllocator func(count uint32) (UniqueID, UniqueID, error)
type ImportFunc func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error)
type GetSegmentStatesFunc func(ctx context.Context, req *datapb.GetSegmentStatesRequest) (*datapb.GetSegmentStatesResponse, error)
type UnsetIsImportingStateFunc func(context.Context, *datapb.UnsetIsImportingStateRequest) (*commonpb.Status, error)
type CheckDiskQuotaFunc func(ctx context.Context, collID UniqueID, files []string) error

type ImportFactory interface {
	NewGetCollectionNameFunc() GetCollectionNameFunc
	NewIDAllocator() IDAllocator
	NewImportFunc() ImportFunc
	NewGetSegmentStatesFunc() GetSegmentStatesFunc
	NewUnsetIsImportingStateFunc() UnsetIsImportingStateFunc
	NewCheckDiskQuotaFunc() CheckDiskQuotaFunc
}

type ImportFactoryImpl struct {
	c *Core
}

func (f ImportFactoryImpl) NewGetCollectionNameFunc() GetCollectionNameFunc {
	return GetCollectionNameWithCore(f.c)
}

func (f ImportFactoryImpl) NewIDAllocator() IDAllocator {
	return IDAllocatorWithCore(f.c)
}

func (f ImportFactoryImpl) NewImportFunc() ImportFunc {
	return ImportFuncWithCore(f.c)
}

func (f ImportFactoryImpl) NewGetSegmentStatesFunc() GetSegmentStatesFunc {
	return GetSegmentStatesWithCore(f.c)
}

func (f ImportFactoryImpl) NewUnsetIsImportingStateFunc() UnsetIsImportingStateFunc {
	return UnsetIsImportingStateWithCore(f.c)
}

func (f ImportFactoryImpl) NewCheckDiskQuotaFunc() CheckDiskQuotaFunc {
	return CheckDiskQuotaWithCore(f.c)
}

func NewImportFactory(c *Core) ImportFactory {
	return &ImportFactoryImpl{c: c}
}

func GetCollectionNameWithCore(c *Core) GetCollectionNameFunc {
	return func(collID, partitionID UniqueID) (string, string, error) {
		colInfo, err := c.meta.GetCollectionByID(c.ctx, collID, typeutil.MaxTimestamp, false)
		if err != nil {
			log.Error("Core failed to get collection name by id", zap.Int64("ID", collID), zap.Error(err))
			return "", "", err
		}
		partName, err := c.meta.GetPartitionNameByID(collID, partitionID, 0)
		if err != nil {
			log.Error("Core failed to get partition name by id", zap.Int64("ID", partitionID), zap.Error(err))
			return colInfo.Name, "", err
		}

		return colInfo.Name, partName, nil
	}
}

func IDAllocatorWithCore(c *Core) IDAllocator {
	return func(count uint32) (UniqueID, UniqueID, error) {
		return c.idAllocator.Alloc(count)
	}
}

func ImportFuncWithCore(c *Core) ImportFunc {
	return func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return c.broker.Import(ctx, req)
	}
}

func GetSegmentStatesWithCore(c *Core) GetSegmentStatesFunc {
	return func(ctx context.Context, req *datapb.GetSegmentStatesRequest) (*datapb.GetSegmentStatesResponse, error) {
		return c.broker.GetSegmentStates(ctx, req)
	}
}

func UnsetIsImportingStateWithCore(c *Core) UnsetIsImportingStateFunc {
	return func(ctx context.Context, req *datapb.UnsetIsImportingStateRequest) (*commonpb.Status, error) {
		return c.broker.UnsetIsImportingState(ctx, req)
	}
}

func CheckDiskQuotaWithCore(c *Core) CheckDiskQuotaFunc {
	return func(ctx context.Context, collID UniqueID, files []string) error {
		var totalSize float64
		for i := 0; i < len(files); i++ {
			filePath := files[i]
			fileSize, err := c.chunkManager.Size(ctx, filePath)
			if err != nil {
				log.Warn("Core failed to get file size", zap.String("filePath", filePath), zap.Error(err))
				return fmt.Errorf("failed to get file size of '%s', error:%w", filePath, err)
			}
			totalSize += float64(fileSize)
		}

		quotaLimit := c.quotaCenter.diskAllowance(collID)
		log.Info("Compare file size with disk quota limit", zap.Float64("totalSize", totalSize),
			zap.Float64("quotaLimit", quotaLimit))
		if totalSize > quotaLimit {
			return fmt.Errorf("file total size '%f' bytes, exceeds disk quota limit '%f' bytes", totalSize, quotaLimit)
		}

		return nil
	}
}
