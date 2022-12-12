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

package dist

import (
	"context"
	"sync"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"go.uber.org/zap"
)

type Controller struct {
	mu          sync.RWMutex
	kv          kv.MetaKv
	handlers    map[int64]*distHandler
	client      session.Cluster
	nodeManager *session.NodeManager
	dist        *meta.DistributionManager
	targetMgr   *meta.TargetManager
	scheduler   task.Scheduler
}

func NewDistController(
	kv kv.MetaKv,
	client session.Cluster,
	nodeManager *session.NodeManager,
	dist *meta.DistributionManager,
	targetMgr *meta.TargetManager,
	scheduler task.Scheduler,
) *Controller {
	return &Controller{
		kv:          kv,
		handlers:    make(map[int64]*distHandler),
		client:      client,
		nodeManager: nodeManager,
		dist:        dist,
		targetMgr:   targetMgr,
		scheduler:   scheduler,
	}
}

func (dc *Controller) StartDistInstance(ctx context.Context, nodeID int64) {
	dc.mu.Lock()
	defer dc.mu.Unlock()
	if _, ok := dc.handlers[nodeID]; ok {
		log.Info("node has started", zap.Int64("nodeID", nodeID))
		return
	}
	h := newDistHandler(ctx, nodeID, dc.kv, dc.client, dc.nodeManager, dc.scheduler, dc.dist, dc.targetMgr)
	dc.handlers[nodeID] = h
}

func (dc *Controller) Remove(nodeID int64) {
	dc.mu.Lock()
	defer dc.mu.Unlock()
	if h, ok := dc.handlers[nodeID]; ok {
		h.stop()
		delete(dc.handlers, nodeID)
	}
}

func (dc *Controller) SyncAll(ctx context.Context) {
	dc.mu.RLock()
	defer dc.mu.RUnlock()

	wg := sync.WaitGroup{}
	for _, h := range dc.handlers {
		wg.Add(1)
		go func(handler *distHandler) {
			defer wg.Done()
			handler.getDistribution(ctx, nil)
		}(h)
	}
	wg.Wait()
}

func (dc *Controller) Stop() {
	dc.mu.Lock()
	defer dc.mu.Unlock()
	for _, h := range dc.handlers {
		h.stop()
	}
}
