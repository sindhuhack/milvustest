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

package datacoord

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	grpcdatanodeclient "github.com/milvus-io/milvus/internal/distributed/datanode/client"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/retry"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

const (
	flushTimeout = 15 * time.Second
	// TODO: evaluate and update import timeout.
	importTimeout = 3 * time.Hour
)

type SessionManager interface {
	AddSession(node *NodeInfo)
	DeleteSession(node *NodeInfo)
	GetSessionIDs() []int64
	GetSessions() []*Session

	Flush(ctx context.Context, nodeID int64, req *datapb.FlushSegmentsRequest)
	FlushChannels(ctx context.Context, nodeID int64, req *datapb.FlushChannelsRequest) error
	Compaction(ctx context.Context, nodeID int64, plan *datapb.CompactionPlan) error
	SyncSegments(nodeID int64, req *datapb.SyncSegmentsRequest) error
	Import(ctx context.Context, nodeID int64, itr *datapb.ImportTaskRequest)
	GetCompactionPlansResults() map[int64]*datapb.CompactionPlanResult
	NotifyChannelOperation(ctx context.Context, nodeID int64, req *datapb.ChannelOperationsRequest) error
	CheckChannelOperationProgress(ctx context.Context, nodeID int64, info *datapb.ChannelWatchInfo) (*datapb.ChannelOperationProgressResponse, error)
	AddImportSegment(ctx context.Context, nodeID int64, req *datapb.AddImportSegmentRequest) (*datapb.AddImportSegmentResponse, error)
	CheckHealth(ctx context.Context) error
	Close()
}

var _ SessionManager = (*SessionManagerImpl)(nil)

// SessionManagerImpl provides the grpc interfaces of cluster
type SessionManagerImpl struct {
	sessions struct {
		sync.RWMutex
		data map[int64]*Session
	}
	sessionCreator dataNodeCreatorFunc
}

// SessionOpt provides a way to set params in SessionManagerImpl
type SessionOpt func(c *SessionManagerImpl)

func withSessionCreator(creator dataNodeCreatorFunc) SessionOpt {
	return func(c *SessionManagerImpl) { c.sessionCreator = creator }
}

func defaultSessionCreator() dataNodeCreatorFunc {
	return func(ctx context.Context, addr string, nodeID int64) (types.DataNodeClient, error) {
		return grpcdatanodeclient.NewClient(ctx, addr, nodeID)
	}
}

// NewSessionManagerImpl creates a new SessionManagerImpl
func NewSessionManagerImpl(options ...SessionOpt) *SessionManagerImpl {
	m := &SessionManagerImpl{
		sessions: struct {
			sync.RWMutex
			data map[int64]*Session
		}{data: make(map[int64]*Session)},
		sessionCreator: defaultSessionCreator(),
	}
	for _, opt := range options {
		opt(m)
	}
	return m
}

// AddSession creates a new session
func (c *SessionManagerImpl) AddSession(node *NodeInfo) {
	c.sessions.Lock()
	defer c.sessions.Unlock()

	session := NewSession(node, c.sessionCreator)
	c.sessions.data[node.NodeID] = session
	metrics.DataCoordNumDataNodes.WithLabelValues().Set(float64(len(c.sessions.data)))
}

// DeleteSession removes the node session
func (c *SessionManagerImpl) DeleteSession(node *NodeInfo) {
	c.sessions.Lock()
	defer c.sessions.Unlock()

	if session, ok := c.sessions.data[node.NodeID]; ok {
		session.Dispose()
		delete(c.sessions.data, node.NodeID)
	}
	metrics.DataCoordNumDataNodes.WithLabelValues().Set(float64(len(c.sessions.data)))
}

// GetSessionIDs returns IDs of all live DataNodes.
func (c *SessionManagerImpl) GetSessionIDs() []int64 {
	c.sessions.RLock()
	defer c.sessions.RUnlock()

	ret := make([]int64, 0, len(c.sessions.data))
	for id := range c.sessions.data {
		ret = append(ret, id)
	}
	return ret
}

// GetSessions gets all node sessions
func (c *SessionManagerImpl) GetSessions() []*Session {
	c.sessions.RLock()
	defer c.sessions.RUnlock()

	ret := make([]*Session, 0, len(c.sessions.data))
	for _, s := range c.sessions.data {
		ret = append(ret, s)
	}
	return ret
}

func (c *SessionManagerImpl) getClient(ctx context.Context, nodeID int64) (types.DataNodeClient, error) {
	c.sessions.RLock()
	session, ok := c.sessions.data[nodeID]
	c.sessions.RUnlock()

	if !ok {
		return nil, fmt.Errorf("can not find session of node %d", nodeID)
	}

	return session.GetOrCreateClient(ctx)
}

// Flush is a grpc interface. It will send req to nodeID asynchronously
func (c *SessionManagerImpl) Flush(ctx context.Context, nodeID int64, req *datapb.FlushSegmentsRequest) {
	go c.execFlush(ctx, nodeID, req)
}

func (c *SessionManagerImpl) execFlush(ctx context.Context, nodeID int64, req *datapb.FlushSegmentsRequest) {
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Warn("failed to get dataNode client", zap.Int64("dataNode ID", nodeID), zap.Error(err))
		return
	}
	ctx, cancel := context.WithTimeout(ctx, flushTimeout)
	defer cancel()

	resp, err := cli.FlushSegments(ctx, req)
	if err := VerifyResponse(resp, err); err != nil {
		log.Error("flush call (perhaps partially) failed", zap.Int64("dataNode ID", nodeID), zap.Error(err))
	} else {
		log.Info("flush call succeeded", zap.Int64("dataNode ID", nodeID))
	}
}

// Compaction is a grpc interface. It will send request to DataNode with provided `nodeID` synchronously.
func (c *SessionManagerImpl) Compaction(ctx context.Context, nodeID int64, plan *datapb.CompactionPlan) error {
	ctx, cancel := context.WithTimeout(ctx, Params.DataCoordCfg.CompactionRPCTimeout.GetAsDuration(time.Second))
	defer cancel()
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Warn("failed to get client", zap.Int64("nodeID", nodeID), zap.Error(err))
		return err
	}

	resp, err := cli.Compaction(ctx, plan)
	if err := VerifyResponse(resp, err); err != nil {
		log.Warn("failed to execute compaction", zap.Int64("node", nodeID), zap.Error(err), zap.Int64("planID", plan.GetPlanID()))
		return err
	}

	log.Info("success to execute compaction", zap.Int64("node", nodeID), zap.Int64("planID", plan.GetPlanID()))
	return nil
}

// SyncSegments is a grpc interface. It will send request to DataNode with provided `nodeID` synchronously.
func (c *SessionManagerImpl) SyncSegments(nodeID int64, req *datapb.SyncSegmentsRequest) error {
	log := log.With(
		zap.Int64("nodeID", nodeID),
		zap.Int64("planID", req.GetPlanID()),
	)
	ctx, cancel := context.WithTimeout(context.Background(), Params.DataCoordCfg.CompactionRPCTimeout.GetAsDuration(time.Second))
	defer cancel()
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Warn("failed to get client", zap.Error(err))
		return err
	}

	err = retry.Do(context.Background(), func() error {
		ctx, cancel := context.WithTimeout(context.Background(), Params.DataCoordCfg.CompactionRPCTimeout.GetAsDuration(time.Second))
		defer cancel()

		resp, err := cli.SyncSegments(ctx, req)
		if err := VerifyResponse(resp, err); err != nil {
			log.Warn("failed to sync segments", zap.Error(err))
			return err
		}
		return nil
	})

	if err != nil {
		log.Warn("failed to sync segments after retry", zap.Error(err))
		return err
	}

	log.Info("success to sync segments")
	return nil
}

// Import is a grpc interface. It will send request to DataNode with provided `nodeID` asynchronously.
func (c *SessionManagerImpl) Import(ctx context.Context, nodeID int64, itr *datapb.ImportTaskRequest) {
	go c.execImport(ctx, nodeID, itr)
}

// execImport gets the corresponding DataNode with its ID and calls its Import method.
func (c *SessionManagerImpl) execImport(ctx context.Context, nodeID int64, itr *datapb.ImportTaskRequest) {
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Warn("failed to get client for import", zap.Int64("nodeID", nodeID), zap.Error(err))
		return
	}
	ctx, cancel := context.WithTimeout(ctx, importTimeout)
	defer cancel()
	resp, err := cli.Import(ctx, itr)
	if err := VerifyResponse(resp, err); err != nil {
		log.Warn("failed to import", zap.Int64("node", nodeID), zap.Error(err))
		return
	}

	log.Info("success to import", zap.Int64("node", nodeID), zap.Any("import task", itr))
}

func (c *SessionManagerImpl) GetCompactionPlansResults() map[int64]*datapb.CompactionPlanResult {
	wg := sync.WaitGroup{}
	ctx := context.Background()

	plans := typeutil.NewConcurrentMap[int64, *datapb.CompactionPlanResult]()
	c.sessions.RLock()
	for nodeID, s := range c.sessions.data {
		wg.Add(1)
		go func(nodeID int64, s *Session) {
			defer wg.Done()
			cli, err := s.GetOrCreateClient(ctx)
			if err != nil {
				log.Info("Cannot Create Client", zap.Int64("NodeID", nodeID))
				return
			}
			ctx, cancel := context.WithTimeout(ctx, Params.DataCoordCfg.CompactionRPCTimeout.GetAsDuration(time.Second))
			defer cancel()
			resp, err := cli.GetCompactionState(ctx, &datapb.CompactionStateRequest{
				Base: commonpbutil.NewMsgBase(
					commonpbutil.WithMsgType(commonpb.MsgType_GetSystemConfigs),
					commonpbutil.WithSourceID(paramtable.GetNodeID()),
				),
			})

			if err := merr.CheckRPCCall(resp, err); err != nil {
				log.Info("Get State failed", zap.Error(err))
				return
			}

			for _, rst := range resp.GetResults() {
				// for compatibility issue, before 2.3.4, resp has only logpath
				// try to parse path and fill logid
				binlog.CompressCompactionBinlogs(rst.GetSegments())
				plans.Insert(rst.PlanID, rst)
			}
		}(nodeID, s)
	}
	c.sessions.RUnlock()
	wg.Wait()

	rst := make(map[int64]*datapb.CompactionPlanResult)
	plans.Range(func(planID int64, result *datapb.CompactionPlanResult) bool {
		rst[planID] = result
		return true
	})

	return rst
}

func (c *SessionManagerImpl) FlushChannels(ctx context.Context, nodeID int64, req *datapb.FlushChannelsRequest) error {
	log := log.Ctx(ctx).With(zap.Int64("nodeID", nodeID),
		zap.Time("flushTs", tsoutil.PhysicalTime(req.GetFlushTs())),
		zap.Strings("channels", req.GetChannels()))
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Warn("failed to get client", zap.Error(err))
		return err
	}

	log.Info("SessionManagerImpl.FlushChannels start")
	resp, err := cli.FlushChannels(ctx, req)
	err = VerifyResponse(resp, err)
	if err != nil {
		log.Warn("SessionManagerImpl.FlushChannels failed", zap.Error(err))
		return err
	}
	log.Info("SessionManagerImpl.FlushChannels successfully")
	return nil
}

func (c *SessionManagerImpl) NotifyChannelOperation(ctx context.Context, nodeID int64, req *datapb.ChannelOperationsRequest) error {
	log := log.Ctx(ctx).With(zap.Int64("nodeID", nodeID))
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Info("failed to get dataNode client", zap.Error(err))
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, Params.DataCoordCfg.ChannelOperationRPCTimeout.GetAsDuration(time.Second))
	defer cancel()
	resp, err := cli.NotifyChannelOperation(ctx, req)
	if err := merr.CheckRPCCall(resp, err); err != nil {
		log.Warn("Notify channel operations failed", zap.Error(err))
		return err
	}
	return nil
}

func (c *SessionManagerImpl) CheckChannelOperationProgress(ctx context.Context, nodeID int64, info *datapb.ChannelWatchInfo) (*datapb.ChannelOperationProgressResponse, error) {
	log := log.With(
		zap.Int64("nodeID", nodeID),
		zap.String("channel", info.GetVchan().GetChannelName()),
		zap.String("operation", info.GetState().String()),
	)
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Info("failed to get dataNode client", zap.Error(err))
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, Params.DataCoordCfg.ChannelOperationRPCTimeout.GetAsDuration(time.Second))
	defer cancel()
	resp, err := cli.CheckChannelOperationProgress(ctx, info)
	if err := merr.CheckRPCCall(resp, err); err != nil {
		log.Warn("Check channel operation failed", zap.Error(err))
		return nil, err
	}

	return resp, nil
}

func (c *SessionManagerImpl) AddImportSegment(ctx context.Context, nodeID int64, req *datapb.AddImportSegmentRequest) (*datapb.AddImportSegmentResponse, error) {
	// Call DataNode to add the new segment to its own flow graph.
	cli, err := c.getClient(ctx, nodeID)
	if err != nil {
		log.Error("failed to get DataNode client for SaveImportSegment",
			zap.Int64("DataNode ID", nodeID),
			zap.Error(err))
		return nil, err
	}

	resp, err := cli.AddImportSegment(ctx, req)
	if err := VerifyResponse(resp.GetStatus(), err); err != nil {
		log.Error("failed to add segment", zap.Int64("nodeID", nodeID), zap.Error(err))
		return nil, err
	}
	log.Info("succeed to add segment", zap.Int64("nodeID", nodeID), zap.Any("add segment req", req))
	return resp, err
}

func (c *SessionManagerImpl) CheckHealth(ctx context.Context) error {
	group, ctx := errgroup.WithContext(ctx)

	ids := c.GetSessionIDs()
	for _, nodeID := range ids {
		nodeID := nodeID
		group.Go(func() error {
			cli, err := c.getClient(ctx, nodeID)
			if err != nil {
				return fmt.Errorf("failed to get DataNode %d: %v", nodeID, err)
			}

			sta, err := cli.GetComponentStates(ctx, &milvuspb.GetComponentStatesRequest{})
			if err != nil {
				return err
			}
			err = merr.AnalyzeState("DataNode", nodeID, sta)
			return err
		})
	}

	return group.Wait()
}

// Close release sessions
func (c *SessionManagerImpl) Close() {
	c.sessions.Lock()
	defer c.sessions.Unlock()

	for _, s := range c.sessions.data {
		s.Dispose()
	}
	c.sessions.data = nil
}
