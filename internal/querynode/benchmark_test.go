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

package querynode

import (
	"context"
	"os"
	"runtime/pprof"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap/zapcore"

	"github.com/milvus-io/milvus/internal/log"
	msgstream2 "github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	maxNQ = 100
	nb    = 10000
)

func benchmarkQueryCollectionSearch(nq int, b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())

	queryCollection, err := genSimpleQueryCollection(tx, cancel, b)
	assert.NoError(b, err)

	// search only one segment
	err = queryCollection.streaming.replica.removeSegment(defaultSegmentID)
	assert.NoError(b, err)
	err = queryCollection.historical.replica.removeSegment(defaultSegmentID)
	assert.NoError(b, err)

	assert.Equal(b, 0, queryCollection.historical.replica.getSegmentNum())
	assert.Equal(b, 0, queryCollection.streaming.replica.getSegmentNum())

	segment, err := genSealedSegmentWithMsgLength(nb)
	assert.NoError(b, err)
	err = queryCollection.historical.replica.setSegment(segment)
	assert.NoError(b, err)

	sessionManager := NewSessionManager(withSessionCreator(mockProxyCreator()))
	sessionManager.AddSession(&NodeInfo{
		NodeID:  0,
		Address: "",
	})
	queryCollection.sessionManager = sessionManager

	// segment check
	assert.Equal(b, 1, queryCollection.historical.replica.getSegmentNum())
	assert.Equal(b, 0, queryCollection.streaming.replica.getSegmentNum())
	seg, err := queryCollection.historical.replica.getSegmentByID(defaultSegmentID)
	assert.NoError(b, err)
	assert.Equal(b, int64(nb), seg.getRowCount())
	sizePerRecord, err := typeutil.EstimateSizePerRecord(genSimpleSegCoreSchema())
	assert.NoError(b, err)
	expectSize := sizePerRecord * nb
	assert.Equal(b, seg.getMemSize(), int64(expectSize))

	// warming up

	msgTmp, err := genSearchMsg(10, IndexFaissIDMap)
	assert.NoError(b, err)
	for j := 0; j < 10000; j++ {
		err = queryCollection.search(msgTmp)
		assert.NoError(b, err)
	}

	msgs := make([]*msgstream2.SearchMsg, maxNQ/nq)
	for i := 0; i < maxNQ/nq; i++ {
		msg, err := genSearchMsg(nq, IndexFaissIDMap)
		assert.NoError(b, err)
		msgs[i] = msg
	}

	f, err := os.Create("nq_" + strconv.Itoa(nq) + ".perf")
	if err != nil {
		panic(err)
	}
	if err = pprof.StartCPUProfile(f); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < maxNQ/nq; j++ {
			err = queryCollection.search(msgs[j])
			assert.NoError(b, err)
		}
	}
}

func benchmarkQueryCollectionSearchIndex(nq int, indexType string, b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())

	queryCollection, err := genSimpleQueryCollection(tx, cancel, b)
	assert.NoError(b, err)

	err = queryCollection.historical.replica.removeSegment(defaultSegmentID)
	assert.NoError(b, err)
	err = queryCollection.streaming.replica.removeSegment(defaultSegmentID)
	assert.NoError(b, err)

	assert.Equal(b, 0, queryCollection.historical.replica.getSegmentNum())
	assert.Equal(b, 0, queryCollection.streaming.replica.getSegmentNum())

	node, err := genSimpleQueryNode(tx, b)
	assert.NoError(b, err)
	node.loader.historicalReplica = queryCollection.historical.replica

	err = loadIndexForSegment(tx, node, defaultSegmentID, nb, indexType, L2)
	assert.NoError(b, err)

	sessionManager := NewSessionManager(withSessionCreator(mockProxyCreator()))
	sessionManager.AddSession(&NodeInfo{
		NodeID:  0,
		Address: "",
	})
	queryCollection.sessionManager = sessionManager

	// segment check
	assert.Equal(b, 1, queryCollection.historical.replica.getSegmentNum())
	assert.Equal(b, 0, queryCollection.streaming.replica.getSegmentNum())
	seg, err := queryCollection.historical.replica.getSegmentByID(defaultSegmentID)
	assert.NoError(b, err)
	assert.Equal(b, int64(nb), seg.getRowCount())
	sizePerRecord, err := typeutil.EstimateSizePerRecord(genSimpleSegCoreSchema())
	assert.NoError(b, err)
	expectSize := sizePerRecord * nb
	assert.Equal(b, seg.getMemSize(), int64(expectSize))

	// warming up
	msgTmp, err := genSearchMsg(10, indexType)
	assert.NoError(b, err)
	for j := 0; j < 10000; j++ {
		err = queryCollection.search(msgTmp)
		assert.NoError(b, err)
	}

	msgs := make([]*msgstream2.SearchMsg, maxNQ/nq)
	for i := 0; i < maxNQ/nq; i++ {
		msg, err := genSearchMsg(nq, indexType)
		assert.NoError(b, err)
		msgs[i] = msg
	}

	f, err := os.Create(indexType + "_nq_" + strconv.Itoa(nq) + ".perf")
	if err != nil {
		panic(err)
	}
	if err = pprof.StartCPUProfile(f); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < maxNQ/nq; j++ {
			err = queryCollection.search(msgs[j])
			assert.NoError(b, err)
		}
	}
}

func BenchmarkSearch_NQ1(b *testing.B) { benchmarkQueryCollectionSearch(1, b) }

//func BenchmarkSearch_NQ10(b *testing.B)    { benchmarkQueryCollectionSearch(10, b) }
//func BenchmarkSearch_NQ100(b *testing.B)   { benchmarkQueryCollectionSearch(100, b) }
//func BenchmarkSearch_NQ1000(b *testing.B)  { benchmarkQueryCollectionSearch(1000, b) }
//func BenchmarkSearch_NQ10000(b *testing.B) { benchmarkQueryCollectionSearch(10000, b) }

func BenchmarkSearch_HNSW_NQ1(b *testing.B) {
	benchmarkQueryCollectionSearchIndex(1, IndexHNSW, b)
}

func BenchmarkSearch_IVFFLAT_NQ1(b *testing.B) {
	benchmarkQueryCollectionSearchIndex(1, IndexFaissIVFFlat, b)
}

/*
func BenchmarkSearch_IVFFLAT_NQ10(b *testing.B) {
	benchmarkQueryCollectionSearchIndex(10, IndexFaissIVFFlat, b)
}
func BenchmarkSearch_IVFFLAT_NQ100(b *testing.B) {
	benchmarkQueryCollectionSearchIndex(100, IndexFaissIVFFlat, b)
}
func BenchmarkSearch_IVFFLAT_NQ1000(b *testing.B) {
	benchmarkQueryCollectionSearchIndex(1000, IndexFaissIVFFlat, b)
}
func BenchmarkSearch_IVFFLAT_NQ10000(b *testing.B) {
	benchmarkQueryCollectionSearchIndex(10000, IndexFaissIVFFlat, b)
}
*/
