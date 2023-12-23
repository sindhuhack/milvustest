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
	"fmt"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"github.com/samber/lo"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"
	"sync"
	"time"
)

type Executor interface {
	Start()
	Close()
}

type executor struct {
	manager TaskManager
	syncMgr syncmgr.SyncManager
	cm      storage.ChunkManager

	// TODO: dyh, add thread pool

	closeOnce sync.Once
	closeChan chan struct{}
}

func NewExecutor(manager TaskManager, syncMgr syncmgr.SyncManager, cm storage.ChunkManager) Executor {
	return &executor{
		manager:   manager,
		syncMgr:   syncMgr,
		cm:        cm,
		closeChan: make(chan struct{}),
	}
}

func (e *executor) Start() {
	log.Info("start import executor")
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-e.closeChan:
			log.Info("import executor exited")
			return
		case <-ticker.C:
			tasks := e.manager.GetBy(WithStates(datapb.ImportState_Pending))
			wg, _ := errgroup.WithContext(context.Background())
			for _, task := range tasks {
				task := task
				wg.Go(func() error {
					switch task.GetType() {
					case PreImportTaskType:
						e.RunPreImportTask(task)
					case ImportTaskType:
						e.RunImportTask(task)
					}
					return nil
				})
			}
			_ = wg.Wait()
		}
	}
}

func (e *executor) Close() {
	e.closeOnce.Do(func() {
		close(e.closeChan)
	})
}

func (e *executor) estimateReadRows(schema *schemapb.CollectionSchema) (int64, error) {
	const BufferSize = 16 * 1024 * 1024 // TODO: dyh, make it configurable
	sizePerRow, err := typeutil.EstimateSizePerRecord(schema)
	if err != nil {
		return 0, err
	}
	return int64(BufferSize / sizePerRow), nil
}

func (e *executor) handleErr(task Task, err error, msg string) {
	log.Warn(msg, zap.Int64("taskID", task.GetTaskID()),
		zap.Int64("requestID", task.GetRequestID()),
		zap.Int64("collectionID", task.GetCollectionID()),
		zap.String("state", task.GetState().String()),
		zap.String("type", task.GetType().String()),
		zap.Error(err))
	e.manager.Update(task.GetTaskID(), UpdateState(datapb.ImportState_Failed), UpdateReason(err.Error()))
}

func (e *executor) RunPreImportTask(task Task) {
	e.manager.Update(task.GetTaskID(), UpdateState(datapb.ImportState_InProgress))
	files := lo.Map(task.(*PreImportTask).GetFileStats(), func(fileStat *datapb.ImportFileStats, _ int) *datapb.ImportFile {
		return fileStat.GetImportFile()
	})

	wg, _ := errgroup.WithContext(context.TODO()) // TODO: dyh, set timeout
	for i, file := range files {
		i := i
		file := file
		wg.Go(func() error {
			reader := NewReader(e.cm, task.GetSchema(), file)
			stat, err := reader.ReadStats()
			if err != nil {
				e.handleErr(task, err, "read stats failed")
				return err
			}
			e.manager.Update(task.GetTaskID(), UpdateFileStat(i, stat))
			return nil
		})
	}
	err := wg.Wait()
	if err != nil {
		e.handleErr(task, err, "preimport failed")
		return
	}
	e.manager.Update(task.GetTaskID(), UpdateState(datapb.ImportState_Completed))
}

func (e *executor) RunImportTask(task Task) {
	e.manager.Update(task.GetTaskID(), UpdateState(datapb.ImportState_InProgress))
	count, err := e.estimateReadRows(task.GetSchema())
	if err != nil {
		e.handleErr(task, err, fmt.Sprintf("estimate rows size failed"))
		return
	}
	for _, fileInfo := range task.(*ImportTask).req.GetFilesInfo() {
		for {
			reader := NewReader(e.cm, task.GetSchema(), fileInfo.GetImportFile())
			rows, err := e.doImportOnFile(count, task, reader, fileInfo)
			if err != nil {
				e.handleErr(task, err, fmt.Sprintf("do import failed, file: %s", fileInfo.GetImportFile().String()))
				return
			}
			if rows == 0 {
				break
			}
		}
	}
	e.manager.Update(task.GetTaskID(), UpdateState(datapb.ImportState_Completed))
}

func (e *executor) doImportOnFile(count int64, task Task, reader Reader, fileInfo *datapb.ImportFileRequestInfo) (int, error) {
	insertData, err := reader.Next(count)
	if err != nil {
		e.handleErr(task, err, fmt.Sprintf(""))
		return 0, err
	}
	readRows := insertData.GetRowNum()
	if readRows == 0 {
		return 0, nil
	}
	iTask := task.(*ImportTask)
	hashedData, err := e.Hash(iTask, insertData)
	if err != nil {
		return 0, err
	}
	futures := make([]*conc.Future[error], 0)
	syncTasks := make([]*syncmgr.SyncTask, 0)
	for channelIdx, datas := range hashedData {
		channel := iTask.vchannels[channelIdx]
		for partitionID, data := range datas {
			segmentID := PickSegment(task, fileInfo, channel, partitionID)
			syncTask := NewSyncTask(iTask, segmentID, partitionID, channel, data)
			future := e.syncMgr.SyncData(context.TODO(), syncTask)
			futures = append(futures, future)
			syncTasks = append(syncTasks, syncTask)
		}
	}
	err = conc.AwaitAll(futures...) // TODO: dyh, return futures and syncTasks to increase concurrence
	if err != nil {
		return 0, err
	}
	for _, syncTask := range syncTasks {
		segmentID := syncTask.SegmentID()
		insertBinlogs, statsBinlog, _ := syncTask.Binlogs()
		metaCache := task.(*ImportTask).metaCaches[syncTask.ChannelName()]
		segment, ok := metaCache.GetSegmentByID(segmentID)
		if !ok {
			return 0, merr.WrapErrSegmentNotFound(segmentID, "import failed")
		}
		segmentInfo := &datapb.ImportSegmentInfo{
			SegmentID:    segmentID,
			ImportedRows: segment.FlushedRows(),
			Binlogs:      lo.Values(insertBinlogs),
			Statslogs:    lo.Values(statsBinlog),
		}
		e.manager.Update(task.GetTaskID(), UpdateSegmentInfo(segmentInfo))
	}
	return readRows, nil
}

func (e *executor) Hash(task *ImportTask, insertData *storage.InsertData) (map[int64]map[int64]*storage.InsertData, error) {
	var err error
	// vchannel -> (partitionID -> InsertData)
	res := make(map[int64]map[int64]*storage.InsertData)
	for i := range task.vchannels {
		res[int64(i)] = make(map[int64]*storage.InsertData)
		for _, partition := range task.partitions {
			res[int64(i)][partition], err = storage.NewInsertData(task.GetSchema())
			if err != nil {
				return nil, err
			}
		}
	}
	pkField, err := typeutil.GetPrimaryFieldSchema(task.GetSchema())
	if err != nil {
		return nil, err
	}
	vchannelNum := int64(len(task.vchannels))
	partitionNum := int64(len(task.partitions))
	fn, err := hashFunc(pkField.GetDataType())
	if err != nil {
		return nil, err
	}
	for i := 0; i < insertData.GetRowNum(); i++ {
		row := insertData.GetRow(i)
		pk := row[pkField.GetFieldID()]
		p1 := fn(pk, vchannelNum)
		p2 := fn(pk, partitionNum)
		err = res[p1][p2].Append(row)
		if err != nil {
			return nil, err
		}
	}
	return res, nil
}
