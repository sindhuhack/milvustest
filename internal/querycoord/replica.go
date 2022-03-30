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

package querycoord

import (
	"sync"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/proto/querypb"
)

type replicaSlice = []*querypb.ReplicaInfo

type ReplicaInfos struct {
	globalGuard sync.RWMutex // We have to make sure atomically update replicas and index

	// Persistent Info
	replicas map[UniqueID]*querypb.ReplicaInfo // replicaID -> *ReplicaInfo

	// Non-persistent info
	nodeIndex map[UniqueID]map[UniqueID]*querypb.ReplicaInfo // nodeID, replicaID -> []*ReplicaInfo
}

func NewReplicaInfos() *ReplicaInfos {
	return &ReplicaInfos{
		globalGuard: sync.RWMutex{},
		replicas:    make(map[int64]*querypb.ReplicaInfo),
		nodeIndex:   make(map[int64]map[int64]*querypb.ReplicaInfo),
	}
}

func (rep *ReplicaInfos) Get(replicaID UniqueID) (*querypb.ReplicaInfo, bool) {
	rep.globalGuard.RLock()
	defer rep.globalGuard.RUnlock()

	info, ok := rep.replicas[replicaID]
	clone := proto.Clone(info).(*querypb.ReplicaInfo)
	return clone, ok
}

// Make sure atomically update replica and index
func (rep *ReplicaInfos) Insert(info *querypb.ReplicaInfo) {
	rep.globalGuard.Lock()
	defer rep.globalGuard.Unlock()

	old, ok := rep.replicas[info.ReplicaID]

	info = proto.Clone(info).(*querypb.ReplicaInfo)
	rep.replicas[info.ReplicaID] = info

	// This updates ReplicaInfo, not inserts a new one
	if ok {
		for _, nodeID := range old.NodeIds {
			nodeReplicas := rep.nodeIndex[nodeID]
			delete(nodeReplicas, old.ReplicaID)
		}
	}

	for _, nodeID := range info.NodeIds {
		replicas, ok := rep.nodeIndex[nodeID]
		if !ok {
			replicas = make(map[UniqueID]*querypb.ReplicaInfo)
		}

		replicas[info.ReplicaID] = info
		rep.nodeIndex[nodeID] = replicas
	}
}

func (rep *ReplicaInfos) GetReplicasByNodeID(nodeID UniqueID) []*querypb.ReplicaInfo {
	rep.globalGuard.RLock()
	defer rep.globalGuard.RUnlock()

	// Avoid to create entry if nodeID not found
	replicas, ok := rep.nodeIndex[nodeID]
	if !ok {
		return nil
	}

	clones := make([]*querypb.ReplicaInfo, 0, len(replicas))
	for _, replica := range replicas {
		clones = append(clones, proto.Clone(replica).(*querypb.ReplicaInfo))
	}

	return clones
}
