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

package importv2

import (
	"context"
	"io"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/golang/protobuf/proto"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type ImportTask struct {
	*datapb.ImportTaskV2
	ctx          context.Context
	cancel       context.CancelFunc
	segmentsInfo map[int64]*datapb.ImportSegmentInfo
	req          *datapb.ImportRequest

	manager    TaskManager
	syncMgr    syncmgr.SyncManager
	cm         storage.ChunkManager
	metaCaches map[string]metacache.MetaCache
}

func NewImportTask(req *datapb.ImportRequest,
	manager TaskManager,
	syncMgr syncmgr.SyncManager,
	cm storage.ChunkManager,
) Task {
	ctx, cancel := context.WithCancel(context.Background())
	// During binlog import, even if the primary key's autoID is set to true,
	// the primary key from the binlog should be used instead of being reassigned.
	if importutilv2.IsBackup(req.GetOptions()) {
		UnsetAutoID(req.GetSchema())
	}
	task := &ImportTask{
		ImportTaskV2: &datapb.ImportTaskV2{
			JobID:        req.GetJobID(),
			TaskID:       req.GetTaskID(),
			CollectionID: req.GetCollectionID(),
			State:        datapb.ImportTaskStateV2_Pending,
		},
		ctx:          ctx,
		cancel:       cancel,
		segmentsInfo: make(map[int64]*datapb.ImportSegmentInfo),
		req:          req,
		manager:      manager,
		syncMgr:      syncMgr,
		cm:           cm,
	}
	task.initMetaCaches(req)
	return task
}

func (t *ImportTask) initMetaCaches(req *datapb.ImportRequest) {
	metaCaches := make(map[string]metacache.MetaCache)
	schema := typeutil.AppendSystemFields(req.GetSchema())
	for _, channel := range req.GetVchannels() {
		info := &datapb.ChannelWatchInfo{
			Vchan: &datapb.VchannelInfo{
				CollectionID: req.GetCollectionID(),
				ChannelName:  channel,
			},
			Schema: schema,
		}
		metaCache := metacache.NewMetaCache(info, func(segment *datapb.SegmentInfo) *metacache.BloomFilterSet {
			return metacache.NewBloomFilterSet()
		})
		metaCaches[channel] = metaCache
	}
	t.metaCaches = metaCaches
}

func (t *ImportTask) GetType() TaskType {
	return ImportTaskType
}

func (t *ImportTask) GetPartitionIDs() []int64 {
	return t.req.GetPartitionIDs()
}

func (t *ImportTask) GetVchannels() []string {
	return t.req.GetVchannels()
}

func (t *ImportTask) GetSchema() *schemapb.CollectionSchema {
	return t.req.GetSchema()
}

func (t *ImportTask) Cancel() {
	t.cancel()
}

func (t *ImportTask) GetSegmentsInfo() []*datapb.ImportSegmentInfo {
	return lo.Values(t.segmentsInfo)
}

func (t *ImportTask) Clone() Task {
	ctx, cancel := context.WithCancel(t.ctx)
	return &ImportTask{
		ImportTaskV2: proto.Clone(t.ImportTaskV2).(*datapb.ImportTaskV2),
		ctx:          ctx,
		cancel:       cancel,
		segmentsInfo: t.segmentsInfo,
		req:          t.req,
		metaCaches:   t.metaCaches,
	}
}

func (t *ImportTask) Execute() []*conc.Future[any] {
	bufferSize := paramtable.Get().DataNodeCfg.ReadBufferSizeInMB.GetAsInt() * 1024 * 1024
	log.Info("start to import", WrapLogFields(t,
		zap.Int("bufferSize", bufferSize),
		zap.Any("schema", t.GetSchema()))...)
	t.manager.Update(t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_InProgress))

	req := t.req

	fn := func(file *internalpb.ImportFile) error {
		reader, err := importutilv2.NewReader(t.ctx, t.cm, t.GetSchema(), file, t.req.GetOptions(), bufferSize)
		if err != nil {
			log.Warn("new reader failed", WrapLogFields(t, zap.String("file", file.String()), zap.Error(err))...)
			t.manager.Update(t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Failed), UpdateReason(err.Error()))
			return err
		}
		defer reader.Close()
		start := time.Now()
		err = t.importFile(reader, t)
		if err != nil {
			log.Warn("do import failed", WrapLogFields(t, zap.String("file", file.String()), zap.Error(err))...)
			t.manager.Update(t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Failed), UpdateReason(err.Error()))
			return err
		}
		log.Info("import file done", WrapLogFields(t, zap.Strings("files", file.GetPaths()),
			zap.Duration("dur", time.Since(start)))...)
		return nil
	}

	futures := make([]*conc.Future[any], 0, len(req.GetFiles()))
	for _, file := range req.GetFiles() {
		file := file
		f := GetExecPool().Submit(func() (any, error) {
			err := fn(file)
			return err, err
		})
		futures = append(futures, f)
	}
	return futures
}

func (t *ImportTask) importFile(reader importutilv2.Reader, task Task) error {
	iTask := task.(*ImportTask)
	syncFutures := make([]*conc.Future[struct{}], 0)
	syncTasks := make([]syncmgr.Task, 0)
	for {
		data, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		err = AppendSystemFieldsData(iTask, data)
		if err != nil {
			return err
		}
		hashedData, err := HashData(iTask, data)
		if err != nil {
			return err
		}
		fs, sts, err := t.sync(iTask, hashedData)
		if err != nil {
			return err
		}
		syncFutures = append(syncFutures, fs...)
		syncTasks = append(syncTasks, sts...)
	}
	err := conc.AwaitAll(syncFutures...)
	if err != nil {
		return err
	}
	for _, syncTask := range syncTasks {
		segmentInfo, err := NewImportSegmentInfo(syncTask, iTask)
		if err != nil {
			return err
		}
		t.manager.Update(task.GetTaskID(), UpdateSegmentInfo(segmentInfo))
		log.Info("sync import data done", WrapLogFields(task, zap.Any("segmentInfo", segmentInfo))...)
	}
	return nil
}

func (t *ImportTask) sync(task *ImportTask, hashedData HashedData) ([]*conc.Future[struct{}], []syncmgr.Task, error) {
	log.Info("start to sync import data", WrapLogFields(task)...)
	futures := make([]*conc.Future[struct{}], 0)
	syncTasks := make([]syncmgr.Task, 0)
	for channelIdx, datas := range hashedData {
		channel := task.GetVchannels()[channelIdx]
		for partitionIdx, data := range datas {
			if data.GetRowNum() == 0 {
				continue
			}
			partitionID := task.GetPartitionIDs()[partitionIdx]
			segmentID := PickSegment(task, channel, partitionID)
			syncTask, err := NewSyncTask(task.ctx, task, segmentID, partitionID, channel, data)
			if err != nil {
				return nil, nil, err
			}
			future := t.syncMgr.SyncData(task.ctx, syncTask)
			futures = append(futures, future)
			syncTasks = append(syncTasks, syncTask)
		}
	}
	return futures, syncTasks, nil
}
