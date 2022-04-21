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

package indexcoord

import (
	"context"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/common"
	grpcindexnode "github.com/milvus-io/milvus/internal/distributed/indexnode"
	"github.com/milvus-io/milvus/internal/indexnode"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func TestIndexCoord(t *testing.T) {
	ctx := context.Background()
	indexNodeMock0 := &indexnode.Mock{}
	Params.Init()
	etcdCli := etcd.GetEtcdTestClient(t)
	defer etcdCli.Close()
	indexNodeMock0.SetEtcdClient(etcdCli)
	err := indexNodeMock0.Init()
	assert.Nil(t, err)
	err = indexNodeMock0.Register()
	assert.Nil(t, err)
	err = indexNodeMock0.Start()
	assert.Nil(t, err)
	factory := dependency.NewDefaultFactory(true)
	ic, err := NewIndexCoord(ctx, factory)
	assert.Nil(t, err)
	ic.reqTimeoutInterval = time.Second * 10
	ic.durationInterval = time.Second
	ic.assignTaskInterval = 200 * time.Millisecond
	ic.taskLimit = 20
	ic.SetEtcdClient(etcdCli)
	err = ic.Init()
	assert.Nil(t, err)
	err = ic.Register()
	assert.Nil(t, err)
	err = ic.Start()
	assert.Nil(t, err)

	err = indexNodeMock0.Stop()
	assert.Nil(t, err)

	in, err := grpcindexnode.NewServer(ctx, factory)
	assert.Nil(t, err)
	assert.NotNil(t, in)
	indexNodeMock := &indexnode.Mock{
		Build:   true,
		Failure: false,
	}
	err = in.SetClient(indexNodeMock)
	in.SetEtcdClient(etcdCli)
	assert.Nil(t, err)

	err = in.Run()
	assert.NoError(t, err)

	state, err := ic.GetComponentStates(ctx)
	assert.Nil(t, err)
	assert.Equal(t, internalpb.StateCode_Healthy, state.State.StateCode)

	indexID := int64(rand.Int())

	var indexBuildID UniqueID
	reqBuildIndex := &indexpb.BuildIndexRequest{
		IndexID:   indexID,
		DataPaths: []string{"DataPath-1", "DataPath-2"},
		NumRows:   0,
		TypeParams: []*commonpb.KeyValuePair{
			{
				Key:   "dim",
				Value: "128",
			},
		},
		FieldSchema: &schemapb.FieldSchema{
			DataType: schemapb.DataType_FloatVector,
		},
	}
	resp, err := ic.BuildIndex(ctx, reqBuildIndex)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	indexBuildID = resp.IndexBuildID
	resp2, err := ic.BuildIndex(ctx, reqBuildIndex)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, indexBuildID, resp2.IndexBuildID)
	assert.Equal(t, "already have same index", resp2.Status.Reason)

	reqGetIndex := &indexpb.GetIndexStatesRequest{
		IndexBuildIDs: []UniqueID{indexBuildID},
	}
	for {
		resp, err := ic.GetIndexStates(ctx, reqGetIndex)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		if resp.States[0].State == commonpb.IndexState_Finished ||
			resp.States[0].State == commonpb.IndexState_Failed {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	reqGetIndeFilePath := &indexpb.GetIndexFilePathsRequest{
		IndexBuildIDs: []UniqueID{indexBuildID},
	}
	respGetIndexFilePaths, err := ic.GetIndexFilePaths(ctx, reqGetIndeFilePath)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, 1, len(respGetIndexFilePaths.FilePaths))
	assert.Equal(t, 2, len(respGetIndexFilePaths.FilePaths[0].IndexFilePaths))
	assert.Equal(t, "IndexFilePath-1", respGetIndexFilePaths.FilePaths[0].IndexFilePaths[0])
	assert.Equal(t, "IndexFilePath-2", respGetIndexFilePaths.FilePaths[0].IndexFilePaths[1])

	reqDrop := &indexpb.DropIndexRequest{
		IndexID: indexID,
	}
	respDrop, err := ic.DropIndex(ctx, reqDrop)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, respDrop.ErrorCode)

	indexMeta := ic.metaTable.GetIndexMetaByIndexBuildID(indexBuildID)
	for indexMeta != nil {
		log.Info("RecycleIndexMeta", zap.Any("meta", indexMeta))
		indexMeta = ic.metaTable.GetIndexMetaByIndexBuildID(indexBuildID)
		time.Sleep(100 * time.Millisecond)
	}

	err = in.Stop()
	assert.Nil(t, err)
	err = ic.Stop()
	assert.Nil(t, err)
}

func TestIndexCoordMetrics(t *testing.T) {
	ctx := context.Background()
	indexNodeMock0 := &indexnode.Mock{}
	Params.Init()
	etcdCli := etcd.GetEtcdTestClient(t)
	defer etcdCli.Close()
	indexNodeMock0.SetEtcdClient(etcdCli)
	err := indexNodeMock0.Init()
	assert.Nil(t, err)
	err = indexNodeMock0.Register()
	assert.Nil(t, err)
	err = indexNodeMock0.Start()
	assert.Nil(t, err)
	factory := dependency.NewDefaultFactory(true)
	ic, err := NewIndexCoord(ctx, factory)
	assert.Nil(t, err)
	ic.reqTimeoutInterval = time.Second * 10
	ic.durationInterval = time.Second
	ic.assignTaskInterval = 200 * time.Millisecond
	ic.taskLimit = 20
	ic.SetEtcdClient(etcdCli)
	err = ic.Init()
	assert.Nil(t, err)
	err = ic.Register()
	assert.Nil(t, err)
	err = ic.Start()
	assert.Nil(t, err)

	err = indexNodeMock0.Stop()
	assert.Nil(t, err)

	in, err := grpcindexnode.NewServer(ctx, factory)
	assert.Nil(t, err)
	assert.NotNil(t, in)
	indexNodeMock := &indexnode.Mock{
		Build:   true,
		Failure: false,
	}
	err = in.SetClient(indexNodeMock)
	in.SetEtcdClient(etcdCli)
	assert.Nil(t, err)

	err = in.Run()
	assert.NoError(t, err)

	state, err := ic.GetComponentStates(ctx)
	assert.Nil(t, err)
	assert.Equal(t, internalpb.StateCode_Healthy, state.State.StateCode)

	t.Run("GetMetrics", func(t *testing.T) {
		req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.Nil(t, err)
		resp, err := ic.GetMetrics(ctx, req)
		assert.Nil(t, err)
		log.Info("GetMetrics, system info",
			zap.String("name", resp.ComponentName),
			zap.String("resp", resp.Response))

		respGetTimeTickChannel, err := ic.GetTimeTickChannel(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, respGetTimeTickChannel.Status.ErrorCode)

		respGetStatsChannel, err := ic.GetStatisticsChannel(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, respGetStatsChannel.Status.ErrorCode)

		ic.UpdateStateCode(internalpb.StateCode_Abnormal)
		reqSystemMetrics, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.Nil(t, err)
		respSystemMetrics, err := ic.GetMetrics(ctx, reqSystemMetrics)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, respSystemMetrics.Status.ErrorCode)
		ic.UpdateStateCode(internalpb.StateCode_Healthy)

		reqIndexNodeMetrics, err := metricsinfo.ConstructRequestByMetricType("GetIndexNodeMetrics")
		assert.Nil(t, err)
		respIndexNodeMetrics, err := ic.GetMetrics(ctx, reqIndexNodeMetrics)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, respIndexNodeMetrics.Status.ErrorCode)

		reqIndexCoordMetrics := &milvuspb.GetMetricsRequest{
			Request: "GetIndexCoordMetrics",
		}
		respIndexCoordMetrics, err := ic.GetMetrics(ctx, reqIndexCoordMetrics)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, respIndexCoordMetrics.Status.ErrorCode)
	})

	err = in.Stop()
	assert.Nil(t, err)
	err = ic.Stop()
	assert.Nil(t, err)
}

func TestIndexCoord_watchNodeLoop(t *testing.T) {
	ech := make(chan *sessionutil.SessionEvent)
	in := &IndexCoord{
		loopWg:    sync.WaitGroup{},
		loopCtx:   context.Background(),
		eventChan: ech,
		session: &sessionutil.Session{
			TriggerKill: true,
			ServerID:    0,
		},
	}
	in.loopWg.Add(1)

	flag := false
	closed := false
	sigDone := make(chan struct{}, 1)
	sigQuit := make(chan struct{}, 1)
	sc := make(chan os.Signal, 1)
	signal.Notify(sc, syscall.SIGINT)
	defer signal.Reset(syscall.SIGINT)

	go func() {
		in.watchNodeLoop()
		flag = true
		sigDone <- struct{}{}
	}()
	go func() {
		<-sc
		closed = true
		sigQuit <- struct{}{}
	}()

	close(ech)
	<-sigDone
	<-sigQuit
	assert.True(t, flag)
	assert.True(t, closed)
}

func TestIndexCoord_GetComponentStates(t *testing.T) {
	n := &IndexCoord{}
	n.stateCode.Store(internalpb.StateCode_Healthy)
	resp, err := n.GetComponentStates(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, common.NotRegisteredID, resp.State.NodeID)
	n.session = &sessionutil.Session{}
	n.session.UpdateRegistered(true)
	resp, err = n.GetComponentStates(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
}

func TestIndexCoord_NotHealthy(t *testing.T) {
	ic := &IndexCoord{}
	ic.stateCode.Store(internalpb.StateCode_Abnormal)
	req := &indexpb.BuildIndexRequest{}
	resp, err := ic.BuildIndex(context.Background(), req)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)

	req2 := &indexpb.DropIndexRequest{}
	status, err := ic.DropIndex(context.Background(), req2)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)

	req3 := &indexpb.GetIndexStatesRequest{}
	resp2, err := ic.GetIndexStates(context.Background(), req3)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp2.Status.ErrorCode)

	req4 := &indexpb.GetIndexFilePathsRequest{
		IndexBuildIDs: []UniqueID{1, 2},
	}
	resp4, err := ic.GetIndexFilePaths(context.Background(), req4)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp4.Status.ErrorCode)
}
