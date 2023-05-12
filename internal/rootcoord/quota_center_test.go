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

package rootcoord

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/ratelimitutil"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type queryCoordMockForQuota struct {
	mockQueryCoord
	retErr        bool
	retFailStatus bool
}

type dataCoordMockForQuota struct {
	mockDataCoord
	retErr        bool
	retFailStatus bool
}

func (q *queryCoordMockForQuota) GetMetrics(ctx context.Context, request *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	if q.retErr {
		return nil, fmt.Errorf("mock err")
	}
	if q.retFailStatus {
		return &milvuspb.GetMetricsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "mock failure status"),
		}, nil
	}
	return &milvuspb.GetMetricsResponse{
		Status: succStatus(),
	}, nil
}

func (d *dataCoordMockForQuota) GetMetrics(ctx context.Context, request *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	if d.retErr {
		return nil, fmt.Errorf("mock err")
	}
	if d.retFailStatus {
		return &milvuspb.GetMetricsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "mock failure status"),
		}, nil
	}
	return &milvuspb.GetMetricsResponse{
		Status: succStatus(),
	}, nil
}

func TestQuotaCenter(t *testing.T) {
	Params.Init()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	core, err := NewCore(ctx, nil)
	assert.Nil(t, err)
	core.tsoAllocator = newMockTsoAllocator()

	pcm := newProxyClientManager(core.proxyCreator)

	t.Run("test QuotaCenter", func(t *testing.T) {
		quotaCenter := NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{}, core.tsoAllocator)
		go quotaCenter.run()
		time.Sleep(10 * time.Millisecond)
		quotaCenter.stop()
	})

	t.Run("test syncMetrics", func(t *testing.T) {
		quotaCenter := NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{}, core.tsoAllocator)
		err = quotaCenter.syncMetrics()
		assert.Error(t, err) // for empty response

		quotaCenter = NewQuotaCenter(pcm, &queryCoordMockForQuota{retErr: true}, &dataCoordMockForQuota{}, core.tsoAllocator)
		err = quotaCenter.syncMetrics()
		assert.Error(t, err)

		quotaCenter = NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{retErr: true}, core.tsoAllocator)
		err = quotaCenter.syncMetrics()
		assert.Error(t, err)

		quotaCenter = NewQuotaCenter(pcm, &queryCoordMockForQuota{retFailStatus: true}, &dataCoordMockForQuota{}, core.tsoAllocator)
		err = quotaCenter.syncMetrics()
		assert.Error(t, err)

		quotaCenter = NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{retFailStatus: true}, core.tsoAllocator)
		err = quotaCenter.syncMetrics()
		assert.Error(t, err)
	})

	t.Run("test forceDeny", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		quotaCenter.readableCollections = []int64{1, 2, 3}
		quotaCenter.resetAllCurrentRates()
		quotaCenter.forceDenyReading(commonpb.ErrorCode_ForceDeny)
		for _, collection := range quotaCenter.readableCollections {
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DQLQuery])
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DQLQuery])
		}

		quotaCenter.writableCollections = []int64{1, 2, 3, 4}
		quotaCenter.resetAllCurrentRates()
		quotaCenter.forceDenyWriting(commonpb.ErrorCode_ForceDeny)
		for _, collection := range quotaCenter.writableCollections {
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DMLInsert])
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DMLDelete])
		}
	})

	t.Run("test calculateRates", func(t *testing.T) {
		quotaCenter := NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{}, core.tsoAllocator)
		err = quotaCenter.calculateRates()
		assert.NoError(t, err)
		alloc := newMockTsoAllocator()
		alloc.GenerateTSOF = func(count uint32) (typeutil.Timestamp, error) {
			return 0, fmt.Errorf("mock err")
		}
		quotaCenter.tsoAllocator = alloc
		err = quotaCenter.calculateRates()
		assert.Error(t, err)
	})

	t.Run("test getTimeTickDelayFactor factors", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		type ttCase struct {
			maxTtDelay     time.Duration
			curTt          time.Time
			fgTt           time.Time
			expectedFactor float64
		}
		t0 := time.Now()
		ttCases := []ttCase{
			{10 * time.Second, t0, t0.Add(1 * time.Second), 1},
			{10 * time.Second, t0, t0, 1},
			{10 * time.Second, t0.Add(1 * time.Second), t0, 0.9},
			{10 * time.Second, t0.Add(2 * time.Second), t0, 0.8},
			{10 * time.Second, t0.Add(5 * time.Second), t0, 0.5},
			{10 * time.Second, t0.Add(7 * time.Second), t0, 0.3},
			{10 * time.Second, t0.Add(9 * time.Second), t0, 0.1},
			{10 * time.Second, t0.Add(10 * time.Second), t0, 0},
			{10 * time.Second, t0.Add(100 * time.Second), t0, 0},
		}

		backup := Params.QuotaConfig.MaxTimeTickDelay

		for _, c := range ttCases {
			Params.QuotaConfig.MaxTimeTickDelay = c.maxTtDelay
			fgTs := tsoutil.ComposeTSByTime(c.fgTt, 0)
			quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{
				1: {
					Fgm: metricsinfo.FlowGraphMetric{
						NumFlowGraph:        1,
						MinFlowGraphTt:      fgTs,
						MinFlowGraphChannel: "dml",
					},
				},
			}
			curTs := tsoutil.ComposeTSByTime(c.curTt, 0)
			factors := quotaCenter.getTimeTickDelayFactor(curTs)
			for _, factor := range factors {
				assert.True(t, math.Abs(factor-c.expectedFactor) < 0.01)
			}
		}

		Params.QuotaConfig.MaxTimeTickDelay = backup
	})

	t.Run("test getTimeTickDelayFactor", func(t *testing.T) {
		// test MaxTimestamp
		quotaCenter := NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{}, core.tsoAllocator)
		factors := quotaCenter.getTimeTickDelayFactor(0)
		for factor := range factors {
			assert.Equal(t, float64(1), factor)
		}

		now := time.Now()

		Params.QuotaConfig.TtProtectionEnabled = true
		Params.QuotaConfig.MaxTimeTickDelay = 3 * time.Second

		// test force deny writing
		alloc := newMockTsoAllocator()
		alloc.GenerateTSOF = func(count uint32) (typeutil.Timestamp, error) {
			added := now.Add(Params.QuotaConfig.MaxTimeTickDelay)
			ts := tsoutil.ComposeTSByTime(added, 0)
			return ts, nil
		}
		quotaCenter.tsoAllocator = alloc
		quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{
			1: {Fgm: metricsinfo.FlowGraphMetric{
				MinFlowGraphTt:      tsoutil.ComposeTSByTime(now, 0),
				NumFlowGraph:        1,
				MinFlowGraphChannel: "dml",
			}}}
		ts, err := quotaCenter.tsoAllocator.GenerateTSO(1)
		assert.NoError(t, err)
		factors = quotaCenter.getTimeTickDelayFactor(ts)
		for _, factor := range factors {
			assert.Equal(t, float64(0), factor)
		}

		// test one-third time tick delay
		alloc.GenerateTSOF = func(count uint32) (typeutil.Timestamp, error) {
			oneThirdDelay := Params.QuotaConfig.MaxTimeTickDelay / 3
			added := now.Add(oneThirdDelay)
			oneThirdTs := tsoutil.ComposeTSByTime(added, 0)
			return oneThirdTs, nil
		}
		quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{
			1: {Fgm: metricsinfo.FlowGraphMetric{
				MinFlowGraphTt:      tsoutil.ComposeTSByTime(now, 0),
				NumFlowGraph:        1,
				MinFlowGraphChannel: "dml",
			}}}
		ts, err = quotaCenter.tsoAllocator.GenerateTSO(1)
		assert.NoError(t, err)
		factors = quotaCenter.getTimeTickDelayFactor(ts)

		for _, factor := range factors {
			ok := math.Abs(factor-2.0/3.0) < 0.0001
			assert.True(t, ok)
		}
	})

	t.Run("test calculateReadRates", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		quotaCenter.readableCollections = []int64{1, 2, 3}
		quotaCenter.proxyMetrics = map[UniqueID]*metricsinfo.ProxyQuotaMetrics{
			1: {Rms: []metricsinfo.RateMetric{
				{Label: internalpb.RateType_DQLSearch.String(), Rate: 100},
				{Label: internalpb.RateType_DQLQuery.String(), Rate: 100},
			}}}

		Params.QuotaConfig.ForceDenyReading = false
		Params.QuotaConfig.QueueProtectionEnabled = true
		Params.QuotaConfig.QueueLatencyThreshold = 100
		Params.QuotaConfig.DQLLimitEnabled = true
		Params.QuotaConfig.DQLMaxQueryRatePerCollection = 500
		Params.QuotaConfig.DQLMaxSearchRatePerCollection = 500
		quotaCenter.resetAllCurrentRates()
		quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{
			1: {SearchQueue: metricsinfo.ReadInfoInQueue{
				AvgQueueDuration: time.Duration(Params.QuotaConfig.QueueLatencyThreshold) * time.Second,
			}, Effect: metricsinfo.NodeEffect{
				NodeID:        1,
				CollectionIDs: []int64{1, 2, 3},
			}}}
		quotaCenter.calculateReadRates()
		for _, collection := range quotaCenter.readableCollections {
			assert.Equal(t, Limit(100.0*0.9), quotaCenter.currentRates[collection][internalpb.RateType_DQLSearch])
			assert.Equal(t, Limit(100.0*0.9), quotaCenter.currentRates[collection][internalpb.RateType_DQLQuery])
		}

		Params.QuotaConfig.NQInQueueThreshold = 100
		quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{
			1: {SearchQueue: metricsinfo.ReadInfoInQueue{
				UnsolvedQueue: Params.QuotaConfig.NQInQueueThreshold,
			}}}
		quotaCenter.calculateReadRates()
		for _, collection := range quotaCenter.readableCollections {
			assert.Equal(t, Limit(100.0*0.9), quotaCenter.currentRates[collection][internalpb.RateType_DQLSearch])
			assert.Equal(t, Limit(100.0*0.9), quotaCenter.currentRates[collection][internalpb.RateType_DQLQuery])
		}

		Params.QuotaConfig.ResultProtectionEnabled = true
		Params.QuotaConfig.MaxReadResultRate = 1
		quotaCenter.proxyMetrics = map[UniqueID]*metricsinfo.ProxyQuotaMetrics{
			1: {Rms: []metricsinfo.RateMetric{
				{Label: internalpb.RateType_DQLSearch.String(), Rate: 100},
				{Label: internalpb.RateType_DQLQuery.String(), Rate: 100},
				{Label: metricsinfo.ReadResultThroughput, Rate: 1.2},
			}}}
		quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{1: {SearchQueue: metricsinfo.ReadInfoInQueue{}}}
		quotaCenter.calculateReadRates()
		for _, collection := range quotaCenter.readableCollections {
			assert.Equal(t, Limit(100.0*0.9), quotaCenter.currentRates[collection][internalpb.RateType_DQLSearch])
			assert.Equal(t, Limit(100.0*0.9), quotaCenter.currentRates[collection][internalpb.RateType_DQLQuery])
		}
	})

	t.Run("test calculateWriteRates", func(t *testing.T) {
		quotaCenter := NewQuotaCenter(pcm, &queryCoordMockForQuota{}, &dataCoordMockForQuota{}, core.tsoAllocator)
		err = quotaCenter.calculateWriteRates()
		assert.NoError(t, err)

		// force deny
		forceBak := Params.QuotaConfig.ForceDenyWriting
		Params.QuotaConfig.ForceDenyWriting = true
		quotaCenter.writableCollections = []int64{1, 2, 3}
		quotaCenter.resetAllCurrentRates()
		err = quotaCenter.calculateWriteRates()
		assert.NoError(t, err)
		for _, collection := range quotaCenter.writableCollections {
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DMLInsert])
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DMLDelete])
		}
		Params.QuotaConfig.ForceDenyWriting = forceBak

		// disable tt delay protection
		disableTtBak := Params.QuotaConfig.TtProtectionEnabled
		Params.QuotaConfig.TtProtectionEnabled = false
		quotaCenter.resetAllCurrentRates()
		quotaCenter.queryNodeMetrics = make(map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics)
		quotaCenter.queryNodeMetrics[0] = &metricsinfo.QueryNodeQuotaMetrics{
			Hms: metricsinfo.HardwareMetrics{
				Memory:      100,
				MemoryUsage: 100,
			},
			Effect: metricsinfo.NodeEffect{CollectionIDs: []int64{1, 2, 3}},
		}
		err = quotaCenter.calculateWriteRates()
		Params.QuotaConfig.TtProtectionEnabled = disableTtBak
	})

	t.Run("test MemoryFactor factors", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		type memCase struct {
			lowWater       float64
			highWater      float64
			memUsage       uint64
			memTotal       uint64
			expectedFactor float64
		}
		memCases := []memCase{
			{0.8, 0.9, 10, 100, 1},
			{0.8, 0.9, 80, 100, 1},
			{0.8, 0.9, 82, 100, 0.8},
			{0.8, 0.9, 85, 100, 0.5},
			{0.8, 0.9, 88, 100, 0.2},
			{0.8, 0.9, 90, 100, 0},

			{0.85, 0.95, 25, 100, 1},
			{0.85, 0.95, 85, 100, 1},
			{0.85, 0.95, 87, 100, 0.8},
			{0.85, 0.95, 90, 100, 0.5},
			{0.85, 0.95, 93, 100, 0.2},
			{0.85, 0.95, 95, 100, 0},
		}

		lowBackup := Params.QuotaConfig.QueryNodeMemoryLowWaterLevel
		highBackup := Params.QuotaConfig.QueryNodeMemoryHighWaterLevel

		quotaCenter.writableCollections = append(quotaCenter.writableCollections, 1, 2, 3)
		for _, c := range memCases {
			Params.QuotaConfig.QueryNodeMemoryLowWaterLevel = c.lowWater
			Params.QuotaConfig.QueryNodeMemoryHighWaterLevel = c.highWater
			quotaCenter.queryNodeMetrics = map[UniqueID]*metricsinfo.QueryNodeQuotaMetrics{
				1: {
					Hms: metricsinfo.HardwareMetrics{
						MemoryUsage: c.memUsage,
						Memory:      c.memTotal,
					},
					Effect: metricsinfo.NodeEffect{
						NodeID:        1,
						CollectionIDs: []int64{1, 2, 3},
					},
				},
			}
			factors := quotaCenter.getMemoryFactor()

			for _, factor := range factors {
				assert.True(t, math.Abs(factor-c.expectedFactor) < 0.01)
			}
		}

		Params.QuotaConfig.QueryNodeMemoryLowWaterLevel = lowBackup
		Params.QuotaConfig.QueryNodeMemoryHighWaterLevel = highBackup
	})

	t.Run("test checkDiskQuota", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		quotaCenter.checkDiskQuota()

		// total DiskQuota exceeded
		quotaBackup := Params.QuotaConfig.DiskQuota
		Params.QuotaConfig.DiskQuota = 99 * 1024 * 1024
		quotaCenter.dataCoordMetrics = &metricsinfo.DataCoordQuotaMetrics{
			TotalBinlogSize:      200 * 1024 * 1024,
			CollectionBinlogSize: map[int64]int64{1: 100 * 1024 * 1024}}
		quotaCenter.writableCollections = []int64{1, 2, 3}
		quotaCenter.resetAllCurrentRates()
		quotaCenter.checkDiskQuota()
		for _, collection := range quotaCenter.writableCollections {
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DMLInsert])
			assert.Equal(t, Limit(0), quotaCenter.currentRates[collection][internalpb.RateType_DMLDelete])
		}
		Params.QuotaConfig.DiskQuota = quotaBackup

		// collection DiskQuota exceeded
		colQuotaBackup := Params.QuotaConfig.DiskQuotaPerCollection
		Params.QuotaConfig.DiskQuotaPerCollection = 30 * 1024 * 1024
		quotaCenter.dataCoordMetrics = &metricsinfo.DataCoordQuotaMetrics{CollectionBinlogSize: map[int64]int64{
			1: 20 * 1024 * 1024, 2: 30 * 1024 * 1024, 3: 60 * 1024 * 1024}}
		quotaCenter.writableCollections = []int64{1, 2, 3}
		quotaCenter.resetAllCurrentRates()
		quotaCenter.checkDiskQuota()
		assert.NotEqual(t, Limit(0), quotaCenter.currentRates[1][internalpb.RateType_DMLInsert])
		assert.NotEqual(t, Limit(0), quotaCenter.currentRates[1][internalpb.RateType_DMLDelete])
		assert.Equal(t, Limit(0), quotaCenter.currentRates[2][internalpb.RateType_DMLInsert])
		assert.Equal(t, Limit(0), quotaCenter.currentRates[2][internalpb.RateType_DMLDelete])
		assert.Equal(t, Limit(0), quotaCenter.currentRates[3][internalpb.RateType_DMLInsert])
		assert.Equal(t, Limit(0), quotaCenter.currentRates[3][internalpb.RateType_DMLDelete])
		Params.QuotaConfig.DiskQuotaPerCollection = colQuotaBackup
	})

	t.Run("test setRates", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		p1 := mocks.NewProxy(t)
		p1.EXPECT().SetRates(mock.Anything, mock.Anything).Return(nil, nil)
		pcm := &proxyClientManager{proxyClient: map[int64]types.Proxy{
			TestProxyID: p1,
		}}
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		quotaCenter.resetAllCurrentRates()
		collectionID := int64(1)
		quotaCenter.currentRates[collectionID] = make(map[internalpb.RateType]ratelimitutil.Limit)
		quotaCenter.quotaStates[collectionID] = make(map[milvuspb.QuotaState]commonpb.ErrorCode)
		quotaCenter.currentRates[collectionID][internalpb.RateType_DMLInsert] = 100
		quotaCenter.quotaStates[collectionID][milvuspb.QuotaState_DenyToWrite] = commonpb.ErrorCode_MemoryQuotaExhausted
		quotaCenter.quotaStates[collectionID][milvuspb.QuotaState_DenyToRead] = commonpb.ErrorCode_ForceDeny
		err = quotaCenter.setRates()
		assert.NoError(t, err)
	})

	t.Run("test recordMetrics", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		collectionID := int64(1)
		quotaCenter.quotaStates[collectionID] = make(map[milvuspb.QuotaState]commonpb.ErrorCode)
		quotaCenter.quotaStates[collectionID][milvuspb.QuotaState_DenyToWrite] = commonpb.ErrorCode_MemoryQuotaExhausted
		quotaCenter.quotaStates[collectionID][milvuspb.QuotaState_DenyToRead] = commonpb.ErrorCode_ForceDeny
		quotaCenter.recordMetrics()
	})

	t.Run("test guaranteeMinRate", func(t *testing.T) {
		qc := types.NewMockQueryCoord(t)
		quotaCenter := NewQuotaCenter(pcm, qc, &dataCoordMockForQuota{}, core.tsoAllocator)
		quotaCenter.resetAllCurrentRates()
		minRate := Limit(100)
		collectionID := int64(1)
		quotaCenter.currentRates[collectionID] = make(map[internalpb.RateType]ratelimitutil.Limit)
		quotaCenter.currentRates[collectionID][internalpb.RateType_DQLSearch] = Limit(50)
		quotaCenter.guaranteeMinRate(float64(minRate), internalpb.RateType_DQLSearch, 1)
		assert.Equal(t, minRate, quotaCenter.currentRates[collectionID][internalpb.RateType_DQLSearch])
	})

	t.Run("test diskAllowance", func(t *testing.T) {
		tests := []struct {
			name            string
			totalDiskQuota  float64
			collDiskQuota   float64
			totalDiskUsage  int64   // in MB
			collDiskUsage   int64   // in MB
			expectAllowance float64 // in bytes
		}{
			{"test max", math.MaxFloat64, math.MaxFloat64, 100, 100, math.MaxFloat64},
			{"test total quota exceeded", 100 * 1024 * 1024, math.MaxFloat64, 100, 100, 0},
			{"test coll quota exceeded", math.MaxFloat64, 20 * 1024 * 1024, 100, 20, 0},
			{"test not exceeded", 100 * 1024 * 1024, 20 * 1024 * 1024, 80, 10, 10 * 1024 * 1024},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				collection := UniqueID(0)
				quotaCenter := NewQuotaCenter(pcm, nil, &dataCoordMockForQuota{}, core.tsoAllocator)
				quotaCenter.resetAllCurrentRates()
				quotaBackup := Params.QuotaConfig.DiskQuota
				colQuotaBackup := Params.QuotaConfig.DiskQuotaPerCollection
				Params.QuotaConfig.DiskQuota = test.totalDiskQuota
				Params.QuotaConfig.DiskQuotaPerCollection = test.collDiskQuota
				quotaCenter.diskMu.Lock()
				quotaCenter.dataCoordMetrics = &metricsinfo.DataCoordQuotaMetrics{}
				quotaCenter.dataCoordMetrics.CollectionBinlogSize = map[int64]int64{collection: test.collDiskUsage * 1024 * 1024}
				quotaCenter.totalBinlogSize = test.totalDiskUsage * 1024 * 1024
				quotaCenter.diskMu.Unlock()
				allowance := quotaCenter.diskAllowance(collection)
				assert.Equal(t, test.expectAllowance, allowance)
				Params.QuotaConfig.DiskQuota = quotaBackup
				Params.QuotaConfig.DiskQuotaPerCollection = colQuotaBackup
			})
		}
	})
}
