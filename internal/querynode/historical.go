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

package querynode

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/golang/protobuf/proto"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/types"
)

const (
	segmentMetaPrefix = "queryCoord-segmentMeta"
)

type historical struct {
	ctx context.Context

	replica      ReplicaInterface
	loader       *segmentLoader
	statsService *statsService

	mu                   sync.Mutex // guards globalSealedSegments
	globalSealedSegments map[UniqueID]*querypb.SegmentInfo

	etcdKV *etcdkv.EtcdKV
}

func newHistorical(ctx context.Context,
	rootCoord types.RootCoord,
	dataCoord types.DataCoord,
	indexCoord types.IndexCoord,
	factory msgstream.Factory,
	etcdKV *etcdkv.EtcdKV) *historical {
	replica := newCollectionReplica(etcdKV)
	loader := newSegmentLoader(ctx, rootCoord, indexCoord, dataCoord, replica, etcdKV)
	ss := newStatsService(ctx, replica, loader.indexLoader.fieldStatsChan, factory)

	return &historical{
		ctx:                  ctx,
		replica:              replica,
		loader:               loader,
		statsService:         ss,
		globalSealedSegments: make(map[UniqueID]*querypb.SegmentInfo),
		etcdKV:               etcdKV,
	}
}

func (h *historical) start() {
	go h.statsService.start()
	go h.watchGlobalSegmentMeta()
}

func (h *historical) close() {
	h.statsService.close()

	// free collectionReplica
	h.replica.freeAll()
}

func (h *historical) watchGlobalSegmentMeta() {
	log.Debug("query node watchGlobalSegmentMeta start")
	watchChan := h.etcdKV.WatchWithPrefix(segmentMetaPrefix)

	for {
		select {
		case <-h.ctx.Done():
			log.Debug("query node watchGlobalSegmentMeta close")
			return
		case resp := <-watchChan:
			for _, event := range resp.Events {
				segmentID, err := strconv.ParseInt(filepath.Base(string(event.Kv.Key)), 10, 64)
				if err != nil {
					log.Error("watchGlobalSegmentMeta failed", zap.Any("error", err.Error()))
					continue
				}
				switch event.Type {
				case mvccpb.PUT:
					log.Debug("globalSealedSegments add segment",
						zap.Any("segmentID", segmentID),
					)
					segmentInfo := &querypb.SegmentInfo{}
					err = proto.UnmarshalText(string(event.Kv.Value), segmentInfo)
					if err != nil {
						log.Error("watchGlobalSegmentMeta failed", zap.Any("error", err.Error()))
						continue
					}
					h.addGlobalSegmentInfo(segmentID, segmentInfo)
				case mvccpb.DELETE:
					log.Debug("globalSealedSegments delete segment",
						zap.Any("segmentID", segmentID),
					)
					h.removeGlobalSegmentInfo(segmentID)
				}
			}
		}
	}
}

func (h *historical) addGlobalSegmentInfo(segmentID UniqueID, segmentInfo *querypb.SegmentInfo) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.globalSealedSegments[segmentID] = segmentInfo
}

func (h *historical) removeGlobalSegmentInfo(segmentID UniqueID) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.globalSealedSegments, segmentID)
}

func (h *historical) getGlobalSegmentIDsByCollectionID(collectionID UniqueID) []UniqueID {
	h.mu.Lock()
	defer h.mu.Unlock()
	resIDs := make([]UniqueID, 0)
	for _, v := range h.globalSealedSegments {
		if v.CollectionID == collectionID {
			resIDs = append(resIDs, v.SegmentID)
		}
	}
	return resIDs
}

func (h *historical) getGlobalSegmentIDsByPartitionIds(partitionIDs []UniqueID) []UniqueID {
	h.mu.Lock()
	defer h.mu.Unlock()
	resIDs := make([]UniqueID, 0)
	for _, v := range h.globalSealedSegments {
		for _, partitionID := range partitionIDs {
			if v.PartitionID == partitionID {
				resIDs = append(resIDs, v.SegmentID)
			}
		}
	}
	return resIDs
}

func (h *historical) removeGlobalSegmentIDsByCollectionID(collectionID UniqueID) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, v := range h.globalSealedSegments {
		if v.CollectionID == collectionID {
			delete(h.globalSealedSegments, v.SegmentID)
		}
	}
}

func (h *historical) removeGlobalSegmentIDsByPartitionIds(partitionIDs []UniqueID) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, v := range h.globalSealedSegments {
		for _, partitionID := range partitionIDs {
			if v.PartitionID == partitionID {
				delete(h.globalSealedSegments, v.SegmentID)
			}
		}
	}
}

func (h *historical) search(searchReqs []*searchRequest,
	collID UniqueID,
	partIDs []UniqueID,
	plan *Plan,
	searchTs Timestamp) ([]*SearchResult, []*Segment, error) {

	searchResults := make([]*SearchResult, 0)
	segmentResults := make([]*Segment, 0)

	// get historical partition ids
	var searchPartIDs []UniqueID
	if len(partIDs) == 0 {
		hisPartIDs, err := h.replica.getPartitionIDs(collID)
		if len(hisPartIDs) == 0 {
			// no partitions in collection, do empty search
			return nil, nil, nil
		}
		if err != nil {
			return searchResults, segmentResults, err
		}
		log.Debug("no partition specified, search all partitions",
			zap.Any("collectionID", collID),
			zap.Any("all partitions", hisPartIDs),
		)
		searchPartIDs = hisPartIDs
	} else {
		for _, id := range partIDs {
			_, err := h.replica.getPartitionByID(id)
			if err == nil {
				log.Debug("append search partition id",
					zap.Any("collectionID", collID),
					zap.Any("partitionID", id),
				)
				searchPartIDs = append(searchPartIDs, id)
			}
		}
	}

	col, err := h.replica.getCollectionByID(collID)
	if err != nil {
		return nil, nil, err
	}

	// all partitions have been released
	if len(searchPartIDs) == 0 && col.getLoadType() == loadTypePartition {
		return nil, nil, errors.New("partitions have been released , collectionID = " +
			fmt.Sprintln(collID) +
			"target partitionIDs = " +
			fmt.Sprintln(partIDs))
	}

	if len(searchPartIDs) == 0 && col.getLoadType() == loadTypeCollection {
		if err = col.checkReleasedPartitions(partIDs); err != nil {
			return nil, nil, err
		}
		return nil, nil, nil
	}

	log.Debug("doing search in historical",
		zap.Any("collectionID", collID),
		zap.Any("reqPartitionIDs", partIDs),
		zap.Any("searchPartitionIDs", searchPartIDs),
	)

	for _, partID := range searchPartIDs {
		segIDs, err := h.replica.getSegmentIDs(partID)
		if err != nil {
			return searchResults, segmentResults, err
		}
		for _, segID := range segIDs {
			seg, err := h.replica.getSegmentByID(segID)
			if err != nil {
				return searchResults, segmentResults, err
			}
			if !seg.getOnService() {
				continue
			}
			searchResult, err := seg.segmentSearch(plan, searchReqs, []Timestamp{searchTs})
			if err != nil {
				return searchResults, segmentResults, err
			}
			searchResults = append(searchResults, searchResult)
			segmentResults = append(segmentResults, seg)
		}
	}

	return searchResults, segmentResults, nil
}
