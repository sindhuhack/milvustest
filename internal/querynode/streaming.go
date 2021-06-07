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
	"github.com/milvus-io/milvus/internal/msgstream"
)

type streaming struct {
	ctx context.Context

	replica      ReplicaInterface
	tSafeReplica TSafeReplicaInterface

	dataSyncService *dataSyncService
	msFactory       msgstream.Factory
}

func newStreaming(ctx context.Context, factory msgstream.Factory) *streaming {
	replica := newCollectionReplica()
	tReplica := newTSafeReplica()
	newDS := newDataSyncService(ctx, replica, tReplica, factory)

	return &streaming{
		replica:         replica,
		tSafeReplica:    tReplica,
		dataSyncService: newDS,
	}
}

func (s *streaming) start() {
	// TODO: start stats
}

func (s *streaming) close() {
	// TODO: stop stats

	if s.dataSyncService != nil {
		s.dataSyncService.close()
	}

	// free collectionReplica
	s.replica.freeAll()
}

func (s *streaming) search(searchReqs []*searchRequest, collID UniqueID, partIDs []UniqueID, plan *Plan, ts Timestamp) ([]*SearchResult, []*Segment, error) {
	searchResults := make([]*SearchResult, 0)
	segmentResults := make([]*Segment, 0)

	// get streaming partition ids
	var searchPartitionIDsInStreaming []UniqueID
	if len(partIDs) == 0 {
		partitionIDsInStreamingCol, err := s.replica.getPartitionIDs(collID)
		if err != nil {
			return searchResults, segmentResults, err
		}
		searchPartitionIDsInStreaming = partitionIDsInStreamingCol
	} else {
		for _, id := range partIDs {
			_, err2 := s.replica.getPartitionByID(id)
			if err2 == nil {
				searchPartitionIDsInStreaming = append(searchPartitionIDsInStreaming, id)
			}
		}
	}

	//TODO:: get searched channels
	for _, partitionID := range searchPartitionIDsInStreaming {
		segmentIDs, err := s.replica.getSegmentIDs(partitionID)
		if err != nil {
			return searchResults, segmentResults, err
		}
		for _, segmentID := range segmentIDs {
			segment, err := s.replica.getSegmentByID(segmentID)
			if err != nil {
				return searchResults, segmentResults, err
			}
			searchResult, err := segment.segmentSearch(plan, searchReqs, []Timestamp{ts})
			if err != nil {
				return searchResults, segmentResults, err
			}
			searchResults = append(searchResults, searchResult)
			segmentResults = append(segmentResults, segment)
		}
	}

	return searchResults, segmentResults, nil
}
