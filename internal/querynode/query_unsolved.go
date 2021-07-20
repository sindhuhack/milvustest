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
	"sync"

	oplog "github.com/opentracing/opentracing-go/log"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/trace"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
)

type unsolvedStage struct {
	ctx    context.Context
	cancel context.CancelFunc

	collectionID UniqueID
	vChannel     Channel

	input  chan queryMsg
	output chan queryResult

	streaming         *streaming
	queryResultStream msgstream.MsgStream

	unsolvedMsgMu sync.Mutex // guards unsolvedMsg
	unsolvedMsg   []queryMsg

	tSafeWatcher *tSafeWatcher

	serviceableTimeMutex sync.Mutex // guards serviceableTime
	serviceableTime      Timestamp
}

func newUnsolvedStage(ctx context.Context,
	cancel context.CancelFunc,
	collectionID UniqueID,
	vChannel Channel,
	input chan queryMsg,
	output chan queryResult,
	streaming *streaming,
	queryResultStream msgstream.MsgStream) *unsolvedStage {

	return &unsolvedStage{
		ctx:               ctx,
		cancel:            cancel,
		collectionID:      collectionID,
		vChannel:          vChannel,
		input:             input,
		output:            output,
		unsolvedMsg:       make([]queryMsg, 0),
		streaming:         streaming,
		queryResultStream: queryResultStream,
	}
}

func (q *unsolvedStage) start() {
	q.register()
	go q.doUnsolvedQueryMsg()
	for {
		select {
		case <-q.ctx.Done():
			log.Debug("stop unsolvedStage", zap.Int64("collectionID", q.collectionID))
			return
		case msg := <-q.input:
			msgType := msg.Type()
			sp, ctx := trace.StartSpanFromContext(msg.TraceCtx())
			msg.SetTraceCtx(ctx)

			// tSafe check
			guaranteeTs := msg.GuaranteeTs()
			serviceTime := q.streaming.tSafeReplica.getTSafe(q.vChannel)
			if guaranteeTs > serviceTime {
				gt, _ := tsoutil.ParseTS(guaranteeTs)
				st, _ := tsoutil.ParseTS(serviceTime)
				log.Debug("query node::unsolvedStage: add to query unsolved",
					zap.Any("collectionID", q.collectionID),
					zap.Any("sm.GuaranteeTimestamp", gt),
					zap.Any("serviceTime", st),
					zap.Any("delta seconds", (guaranteeTs-serviceTime)/(1000*1000*1000)),
					zap.Any("msgID", msg.ID()),
					zap.Any("msgType", msg.Type()),
				)
				q.addToUnsolvedMsg(msg)
				sp.LogFields(
					oplog.String("send to query unsolved", "send to query unsolved"),
					oplog.Object("guarantee ts", gt),
					oplog.Object("serviceTime", st),
					oplog.Float64("delta seconds", float64(guaranteeTs-serviceTime)/(1000.0*1000.0*1000.0)),
				)
				sp.Finish()
				continue
			}

			log.Debug("doing query in vChannelStage...",
				zap.Any("collectionID", q.collectionID),
				zap.Any("msgID", msg.ID()),
				zap.Any("msgType", msgType),
			)

			switch msgType {
			case commonpb.MsgType_Retrieve:
				rm := msg.(*retrieveMsg)
				segmentRetrieved, res, err := q.retrieve(rm)
				retrieveRes := &retrieveResult{
					msg:              rm,
					err:              err,
					segmentRetrieved: segmentRetrieved,
					res:              res,
				}
				q.output <- retrieveRes
			case commonpb.MsgType_Search:
				sm := msg.(*searchMsg)
				searchResults, matchedSegments, sealedSegmentSearched, err := q.search(sm)
				searchRes := &searchResult{
					reqs:                  sm.reqs,
					msg:                   sm,
					err:                   err,
					searchResults:         searchResults,
					matchedSegments:       matchedSegments,
					sealedSegmentSearched: sealedSegmentSearched,
				}
				q.output <- searchRes
			default:
				err := fmt.Errorf("receive invalid msgType = %d", msgType)
				log.Error(err.Error())
			}

			sp.Finish()
		}
	}
}

func (q *unsolvedStage) doUnsolvedQueryMsg() {
	log.Debug("starting doUnsolvedMsg...",
		zap.Any("collectionID", q.collectionID),
		zap.Any("channel", q.vChannel),
	)
	for {
		select {
		case <-q.ctx.Done():
			log.Debug("stop Collection's doUnsolvedMsg",
				zap.Any("collectionID", q.collectionID),
				zap.Any("channel", q.vChannel),
			)
			return
		default:
			serviceTime := q.waitNewTSafe()
			st, _ := tsoutil.ParseTS(serviceTime)
			log.Debug("get tSafe from flow graph",
				zap.Int64("collectionID", q.collectionID),
				zap.Any("tSafe", st))

			q.setServiceableTime(serviceTime)

			unSolvedMsg := make([]queryMsg, 0)
			tempMsg := q.popAllUnsolvedMsg()

			for _, m := range tempMsg {
				guaranteeTs := m.GuaranteeTs()
				gt, _ := tsoutil.ParseTS(guaranteeTs)
				st, _ = tsoutil.ParseTS(serviceTime)
				log.Debug("get query message from unsolvedMsg",
					zap.Int64("collectionID", q.collectionID),
					zap.Int64("msgID", m.ID()),
					zap.Any("reqTime_p", gt),
					zap.Any("serviceTime_p", st),
					zap.Any("guaranteeTime_l", guaranteeTs),
					zap.Any("serviceTime_l", serviceTime),
				)
				if guaranteeTs <= serviceTime {
					unSolvedMsg = append(unSolvedMsg, m)
					continue
				}
				log.Debug("query node::doUnsolvedMsg: add to unsolvedMsg",
					zap.Any("collectionID", q.collectionID),
					zap.Any("sm.BeginTs", gt),
					zap.Any("serviceTime", st),
					zap.Any("delta seconds", (guaranteeTs-serviceTime)/(1000*1000*1000)),
					zap.Any("msgID", m.ID()),
				)
				q.addToUnsolvedMsg(m)
			}

			if len(unSolvedMsg) <= 0 {
				continue
			}
			for _, m := range unSolvedMsg {
				msgType := m.Type()
				var err error
				sp, ctx := trace.StartSpanFromContext(m.TraceCtx())
				m.SetTraceCtx(ctx)
				log.Debug("doing search in doUnsolvedMsg...",
					zap.Int64("collectionID", q.collectionID),
					zap.Int64("msgID", m.ID()),
				)

				switch msgType {
				case commonpb.MsgType_Retrieve:
					rm := m.(*retrieveMsg)
					segmentRetrieved, res, err := q.retrieve(rm)
					retrieveRes := &retrieveResult{
						msg:              rm,
						err:              err,
						segmentRetrieved: segmentRetrieved,
						res:              res,
					}
					q.output <- retrieveRes
				case commonpb.MsgType_Search:
					sm := m.(*searchMsg)
					searchResults, matchedSegments, sealedSegmentSearched, err := q.search(sm)
					searchRes := &searchResult{
						reqs:                  sm.reqs,
						msg:                   sm,
						err:                   err,
						searchResults:         searchResults,
						matchedSegments:       matchedSegments,
						sealedSegmentSearched: sealedSegmentSearched,
					}
					q.output <- searchRes
				default:
					err := fmt.Errorf("receive invalid msgType = %d", msgType)
					log.Error(err.Error())
					continue
				}

				if err != nil {
					log.Error(err.Error())
					publishFailedQueryResult(m, err.Error(), q.queryResultStream)
					log.Debug("do query failed in doUnsolvedMsg, publish failed query result",
						zap.Int64("collectionID", q.collectionID),
						zap.Int64("msgID", m.ID()),
					)
				}
				sp.Finish()
				log.Debug("do query done in doUnsolvedMsg",
					zap.Int64("collectionID", q.collectionID),
					zap.Int64("msgID", m.ID()),
				)
			}
			log.Debug("doUnsolvedMsg: do query done", zap.Int("num of query msg", len(unSolvedMsg)))
		}
	}
}

func (q *unsolvedStage) register() {
	channel := q.vChannel
	log.Debug("register tSafe watcher and init watcher select case",
		zap.Any("collectionID", q.collectionID),
		zap.Any("dml channel", q.vChannel),
	)
	q.tSafeWatcher = newTSafeWatcher()
	q.streaming.tSafeReplica.registerTSafeWatcher(channel, q.tSafeWatcher)
}

func (q *unsolvedStage) addToUnsolvedMsg(msg queryMsg) {
	q.unsolvedMsgMu.Lock()
	defer q.unsolvedMsgMu.Unlock()
	q.unsolvedMsg = append(q.unsolvedMsg, msg)
}

func (q *unsolvedStage) popAllUnsolvedMsg() []queryMsg {
	q.unsolvedMsgMu.Lock()
	defer q.unsolvedMsgMu.Unlock()
	tmp := q.unsolvedMsg
	q.unsolvedMsg = q.unsolvedMsg[:0]
	return tmp
}

func (q *unsolvedStage) waitNewTSafe() Timestamp {
	// block until vChannel updating tSafe
	q.tSafeWatcher.hasUpdate()
	t := q.streaming.tSafeReplica.getTSafe(q.vChannel)
	//pt, _ := tsoutil.ParseTS(t)
	//log.Debug("wait new tSafe",
	//	zap.Any("collectionID", q.collectionID),
	//	zap.Any("channel", q.vChannel),
	//	zap.Any("t", t),
	//	zap.Any("physicalTime", pt),
	//)
	return t
}

func (q *unsolvedStage) getServiceableTime() Timestamp {
	q.serviceableTimeMutex.Lock()
	defer q.serviceableTimeMutex.Unlock()
	return q.serviceableTime
}

func (q *unsolvedStage) setServiceableTime(t Timestamp) {
	q.serviceableTimeMutex.Lock()
	defer q.serviceableTimeMutex.Unlock()

	if t < q.serviceableTime {
		return
	}

	gracefulTimeInMilliSecond := Params.GracefulTime
	if gracefulTimeInMilliSecond > 0 {
		gracefulTime := tsoutil.ComposeTS(gracefulTimeInMilliSecond, 0)
		q.serviceableTime = t + gracefulTime
	} else {
		q.serviceableTime = t
	}
}

func (q *unsolvedStage) retrieve(retrieveMsg *retrieveMsg) ([]UniqueID, []*segcorepb.RetrieveResults, error) {
	collectionID := retrieveMsg.CollectionID
	tr := timerecord.NewTimeRecorder(fmt.Sprintf("retrieve %d", collectionID))

	var partitionIDsInStreaming []UniqueID
	partitionIDsInQuery := retrieveMsg.PartitionIDs
	if len(partitionIDsInQuery) == 0 {
		partitionIDsInStreamingCol, err := q.streaming.replica.getPartitionIDs(collectionID)
		if err != nil {
			return nil, nil, err
		}
		partitionIDsInStreaming = partitionIDsInStreamingCol
	} else {
		for _, id := range partitionIDsInQuery {
			_, err := q.streaming.replica.getPartitionByID(id)
			if err != nil {
				return nil, nil, err
			}
			partitionIDsInStreaming = append(partitionIDsInStreaming, id)
		}
	}
	sealedSegmentRetrieved := make([]UniqueID, 0)
	var mergeList []*segcorepb.RetrieveResults
	for _, partitionID := range partitionIDsInStreaming {
		segmentIDs, err := q.streaming.replica.getSegmentIDs(partitionID)
		if err != nil {
			return nil, nil, err
		}
		for _, segmentID := range segmentIDs {
			segment, err := q.streaming.replica.getSegmentByID(segmentID)
			if err != nil {
				return nil, nil, err
			}
			result, err := segment.getEntityByIds(retrieveMsg.plan)
			if err != nil {
				return nil, nil, err
			}
			mergeList = append(mergeList, result)
			sealedSegmentRetrieved = append(sealedSegmentRetrieved, segmentID)
		}
	}
	tr.Record("streaming retrieve done")
	tr.Elapse("all done")
	return sealedSegmentRetrieved, mergeList, nil
}

func (q *unsolvedStage) search(searchMsg *searchMsg) ([]*SearchResult, []*Segment, []UniqueID, error) {
	sp, ctx := trace.StartSpanFromContext(searchMsg.TraceCtx())
	defer sp.Finish()
	searchMsg.SetTraceCtx(ctx)

	travelTimestamp := searchMsg.TravelTimestamp
	collectionID := searchMsg.CollectionID

	// multiple search is not supported for now
	if len(searchMsg.reqs) != 1 {
		return nil, nil, nil, errors.New("illegal search requests, collectionID = " + fmt.Sprintln(collectionID))
	}
	queryNum := searchMsg.reqs[0].getNumOfQuery()
	topK := searchMsg.plan.getTopK()

	if searchMsg.GetDslType() == commonpb.DslType_BoolExprV1 {
		sp.LogFields(oplog.String("statistical time", "stats start"),
			oplog.Object("nq", queryNum),
			oplog.Object("expr", searchMsg.SerializedExprPlan))
	} else {
		sp.LogFields(oplog.String("statistical time", "stats start"),
			oplog.Object("nq", queryNum),
			oplog.Object("dsl", searchMsg.Dsl))
	}

	tr := timerecord.NewTimeRecorder(fmt.Sprintf("search %d(nq=%d, k=%d)", searchMsg.CollectionID, queryNum, topK))
	sealedSegmentSearched := make([]UniqueID, 0)

	// streaming search
	strSearchResults, strSegmentResults, err := q.streaming.search(searchMsg.reqs,
		collectionID,
		searchMsg.PartitionIDs,
		q.vChannel,
		searchMsg.plan,
		travelTimestamp)

	if err != nil {
		log.Error(err.Error())
		return nil, nil, nil, err
	}

	for _, seg := range strSegmentResults {
		sealedSegmentSearched = append(sealedSegmentSearched, seg.segmentID)
	}
	tr.Record("streaming search done")
	sp.LogFields(oplog.String("statistical time", "streaming search done"))
	tr.Elapse("all done")
	return strSearchResults, strSegmentResults, sealedSegmentSearched, nil
}
