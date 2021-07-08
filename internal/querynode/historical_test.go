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
	"strconv"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/querypb"
)

func TestHistorical_GlobalSealedSegments(t *testing.T) {
	n := newQueryNodeMock()
	assert.Equal(t, 0, len(n.historical.globalSealedSegments))

	// init meta
	segmentID := UniqueID(0)
	partitionID := UniqueID(1)
	collectionID := UniqueID(2)
	segmentInfo := &querypb.SegmentInfo{
		SegmentID:    segmentID,
		CollectionID: collectionID,
		PartitionID:  partitionID,
	}

	// static test
	n.historical.addGlobalSegmentInfo(segmentID, segmentInfo)
	segmentIDs := n.historical.getGlobalSegmentIDsByCollectionID(collectionID)
	assert.Equal(t, 1, len(segmentIDs))
	assert.Equal(t, segmentIDs[0], segmentID)

	segmentIDs = n.historical.getGlobalSegmentIDsByPartitionIds([]UniqueID{partitionID})
	assert.Equal(t, 1, len(segmentIDs))
	assert.Equal(t, segmentIDs[0], segmentID)

	n.historical.removeGlobalSegmentInfo(segmentID)
	assert.Equal(t, 0, len(n.historical.globalSealedSegments))

	n.historical.addGlobalSegmentInfo(segmentID, segmentInfo)
	n.historical.removeGlobalSegmentIDsByCollectionID(collectionID)
	assert.Equal(t, 0, len(n.historical.globalSealedSegments))

	n.historical.addGlobalSegmentInfo(segmentID, segmentInfo)
	n.historical.removeGlobalSegmentIDsByPartitionIds([]UniqueID{partitionID})
	assert.Equal(t, 0, len(n.historical.globalSealedSegments))

	// watch test
	go n.historical.watchGlobalSegmentMeta()
	segmentInfoStr := proto.MarshalTextString(segmentInfo)
	assert.NotNil(t, n.etcdKV)
	segmentKey := segmentMetaPrefix + "/" + strconv.FormatInt(segmentID, 10)
	err := n.etcdKV.Save(segmentKey, segmentInfoStr)
	assert.NoError(t, err)

	time.Sleep(100 * time.Millisecond) // for etcd latency
	segmentIDs = n.historical.getGlobalSegmentIDsByCollectionID(collectionID)
	assert.Equal(t, 1, len(segmentIDs))
	assert.Equal(t, segmentIDs[0], segmentID)

	segmentIDs = n.historical.getGlobalSegmentIDsByPartitionIds([]UniqueID{partitionID})
	assert.Equal(t, 1, len(segmentIDs))
	assert.Equal(t, segmentIDs[0], segmentID)

	err = n.etcdKV.Remove(segmentKey)
	assert.NoError(t, err)
	time.Sleep(100 * time.Millisecond) // for etcd latency
	assert.Equal(t, 0, len(n.historical.globalSealedSegments))
}
