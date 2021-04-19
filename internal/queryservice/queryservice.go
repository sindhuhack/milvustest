package queryservice

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"sync/atomic"

	nodeclient "github.com/zilliztech/milvus-distributed/internal/distributed/querynode/client"
	"github.com/zilliztech/milvus-distributed/internal/errors"
	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/datapb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
	"github.com/zilliztech/milvus-distributed/internal/proto/milvuspb"
	"github.com/zilliztech/milvus-distributed/internal/proto/querypb"
	"github.com/zilliztech/milvus-distributed/internal/querynode"
)

type MasterServiceInterface interface {
	ShowPartitions(in *milvuspb.ShowPartitionRequest) (*milvuspb.ShowPartitionResponse, error)
	ShowSegments(in *milvuspb.ShowSegmentRequest) (*milvuspb.ShowSegmentResponse, error)
}

type DataServiceInterface interface {
	GetSegmentStates(req *datapb.SegmentStatesRequest) (*datapb.SegmentStatesResponse, error)
	GetInsertChannels(req *datapb.InsertChannelRequest) ([]string, error)
}

type QueryNodeInterface interface {
	GetComponentStates() (*internalpb2.ComponentStates, error)

	AddQueryChannel(in *querypb.AddQueryChannelsRequest) (*commonpb.Status, error)
	RemoveQueryChannel(in *querypb.RemoveQueryChannelsRequest) (*commonpb.Status, error)
	WatchDmChannels(in *querypb.WatchDmChannelsRequest) (*commonpb.Status, error)
	LoadSegments(in *querypb.LoadSegmentRequest) (*commonpb.Status, error)
	ReleaseSegments(in *querypb.ReleaseSegmentRequest) (*commonpb.Status, error)
	GetSegmentInfo(req *querypb.SegmentInfoRequest) (*querypb.SegmentInfoResponse, error)
}

type QueryService struct {
	loopCtx    context.Context
	loopCancel context.CancelFunc

	queryServiceID uint64
	replica        metaReplica

	dataServiceClient   DataServiceInterface
	masterServiceClient MasterServiceInterface
	queryNodes          map[UniqueID]*queryNodeInfo
	numRegisterNode     uint64
	numQueryChannel     uint64

	stateCode  atomic.Value
	isInit     atomic.Value
	enableGrpc bool
}

func (qs *QueryService) Init() error {
	Params.Init()
	qs.isInit.Store(true)
	return nil
}

func (qs *QueryService) Start() error {
	isInit := qs.isInit.Load().(bool)
	if !isInit {
		return errors.New("call start before init")
	}
	qs.stateCode.Store(internalpb2.StateCode_HEALTHY)
	return nil
}

func (qs *QueryService) Stop() error {
	qs.loopCancel()
	qs.stateCode.Store(internalpb2.StateCode_ABNORMAL)
	return nil
}

func (qs *QueryService) GetComponentStates() (*internalpb2.ComponentStates, error) {
	serviceComponentInfo := &internalpb2.ComponentInfo{
		NodeID:    Params.QueryServiceID,
		StateCode: qs.stateCode.Load().(internalpb2.StateCode),
	}
	subComponentInfos := make([]*internalpb2.ComponentInfo, 0)
	for nodeID, node := range qs.queryNodes {
		componentStates, err := node.GetComponentStates()
		if err != nil {
			subComponentInfos = append(subComponentInfos, &internalpb2.ComponentInfo{
				NodeID:    nodeID,
				StateCode: internalpb2.StateCode_ABNORMAL,
			})
			continue
		}
		subComponentInfos = append(subComponentInfos, componentStates.State)
	}
	return &internalpb2.ComponentStates{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		State:              serviceComponentInfo,
		SubcomponentStates: subComponentInfos,
	}, nil
}

func (qs *QueryService) GetTimeTickChannel() (string, error) {
	return Params.TimeTickChannelName, nil
}

func (qs *QueryService) GetStatisticsChannel() (string, error) {
	return Params.StatsChannelName, nil
}

func (qs *QueryService) RegisterNode(req *querypb.RegisterNodeRequest) (*querypb.RegisterNodeResponse, error) {
	fmt.Println("register query node =", req.Address)
	// TODO:: add mutex
	allocatedID := len(qs.queryNodes)

	registerNodeAddress := req.Address.Ip + ":" + strconv.FormatInt(req.Address.Port, 10)
	var node *queryNodeInfo
	if qs.enableGrpc {
		client := nodeclient.NewClient(registerNodeAddress)
		node = newQueryNodeInfo(client)
	} else {
		client := querynode.NewQueryNode(qs.loopCtx, uint64(allocatedID))
		node = newQueryNodeInfo(client)
	}
	qs.queryNodes[UniqueID(allocatedID)] = node

	//TODO::return init params to queryNode
	return &querypb.RegisterNodeResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		InitParams: &internalpb2.InitParams{
			NodeID: int64(allocatedID),
		},
	}, nil
}

func (qs *QueryService) ShowCollections(req *querypb.ShowCollectionRequest) (*querypb.ShowCollectionResponse, error) {
	dbID := req.DbID
	collections, err := qs.replica.getCollections(dbID)
	collectionIDs := make([]UniqueID, 0)
	for _, collection := range collections {
		collectionIDs = append(collectionIDs, collection.id)
	}
	if err != nil {
		return &querypb.ShowCollectionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, err
	}
	return &querypb.ShowCollectionResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		CollectionIDs: collectionIDs,
	}, nil
}

func (qs *QueryService) LoadCollection(req *querypb.LoadCollectionRequest) (*commonpb.Status, error) {
	dbID := req.DbID
	collectionID := req.CollectionID
	fn := func(err error) *commonpb.Status {
		if err != nil {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			}
		}
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		}
	}
	collection, err := qs.replica.loadCollection(dbID, collectionID)
	if err != nil {
		return fn(err), err
	}

	if len(collection.dmChannelNames) != 0 {
		return fn(nil), nil
	}

	channelRequest := datapb.InsertChannelRequest{
		DbID:         req.DbID,
		CollectionID: req.CollectionID,
	}
	dmChannels, err := qs.dataServiceClient.GetInsertChannels(&channelRequest)
	if err != nil {
		return fn(err), err
	}

	// get partitionIDs
	showPartitionRequest := &milvuspb.ShowPartitionRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_kShowPartitions,
		},
		CollectionID: collectionID,
	}

	showPartitionResponse, err := qs.masterServiceClient.ShowPartitions(showPartitionRequest)
	if err != nil {
		return fn(err), err
	}
	if showPartitionResponse.Status.ErrorCode != commonpb.ErrorCode_SUCCESS {
		return showPartitionResponse.Status, err
	}
	partitionIDs := showPartitionResponse.PartitionIDs

	if len(partitionIDs) == 0 {
		loadSegmentRequest := &querypb.LoadSegmentRequest{
			CollectionID: collectionID,
		}
		for _, node := range qs.queryNodes {
			_, err := node.LoadSegments(loadSegmentRequest)
			if err != nil {
				return fn(err), err
			}
		}
		nodeIDs := qs.shuffleChannelsToQueryNode(dmChannels)
		err = qs.watchDmChannels(dmChannels, nodeIDs, collection)
		return fn(err), err
	}

	loadPartitionsRequest := &querypb.LoadPartitionRequest{
		Base:         req.Base,
		DbID:         dbID,
		CollectionID: collectionID,
		PartitionIDs: partitionIDs,
	}

	status, err := qs.LoadPartitions(loadPartitionsRequest)

	return status, err
}

func (qs *QueryService) ReleaseCollection(req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	dbID := req.DbID
	collectionID := req.CollectionID
	partitions, err := qs.replica.getPartitions(dbID, collectionID)
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, err
	}

	partitionIDs := make([]UniqueID, 0)
	for _, partition := range partitions {
		partitionIDs = append(partitionIDs, partition.id)
	}

	releasePartitionRequest := &querypb.ReleasePartitionRequest{
		Base:         req.Base,
		DbID:         dbID,
		CollectionID: collectionID,
		PartitionIDs: partitionIDs,
	}

	status, err := qs.ReleasePartitions(releasePartitionRequest)

	err = qs.replica.releaseCollection(dbID, collectionID)
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, err
	}
	//TODO:: queryNode cancel subscribe dmChannels
	return status, err
}

func (qs *QueryService) ShowPartitions(req *querypb.ShowPartitionRequest) (*querypb.ShowPartitionResponse, error) {
	dbID := req.DbID
	collectionID := req.CollectionID
	partitions, err := qs.replica.getPartitions(dbID, collectionID)
	partitionIDs := make([]UniqueID, 0)
	for _, partition := range partitions {
		partitionIDs = append(partitionIDs, partition.id)
	}
	if err != nil {
		return &querypb.ShowPartitionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, err
	}
	return &querypb.ShowPartitionResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		PartitionIDs: partitionIDs,
	}, nil
}

func (qs *QueryService) LoadPartitions(req *querypb.LoadPartitionRequest) (*commonpb.Status, error) {
	//TODO::suggest different partitions have different dm channel
	dbID := req.DbID
	collectionID := req.CollectionID
	partitionIDs := req.PartitionIDs

	fn := func(err error) *commonpb.Status {
		if err != nil {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			}
		}
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		}
	}

	if len(partitionIDs) == 0 {
		err := errors.New("partitionIDs are empty")
		return fn(err), err
	}

	var collection *collection = nil
	var err error
	if collection, err = qs.replica.loadCollection(dbID, collectionID); err != nil {
		return fn(err), err
	}

	for _, partitionID := range partitionIDs {
		partition, err := qs.replica.loadPartition(dbID, collectionID, partitionID)
		if err != nil {
			return fn(err), err
		}

		if partition == nil {
			return fn(err), nil
		}

		showSegmentRequest := &milvuspb.ShowSegmentRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_kShowSegment,
			},
			CollectionID: collectionID,
			PartitionID:  partitionID,
		}
		showSegmentResponse, err := qs.masterServiceClient.ShowSegments(showSegmentRequest)
		if err != nil {
			return fn(err), err
		}

		segmentIDs := showSegmentResponse.SegmentIDs
		segmentStates := make(map[UniqueID]*datapb.SegmentStateInfo)
		channel2segs := make(map[string][]UniqueID)

		resp, err := qs.dataServiceClient.GetSegmentStates(&datapb.SegmentStatesRequest{
			SegmentIDs: segmentIDs,
		})

		if err != nil {
			return fn(err), err
		}

		for _, state := range resp.States {
			segmentID := state.SegmentID
			segmentStates[segmentID] = state

			channelName := state.StartPosition.ChannelName

			if _, ok := channel2segs[channelName]; !ok {
				segments := make([]UniqueID, 0)
				segments = append(segments, segmentID)
				channel2segs[channelName] = segments
			} else {
				channel2segs[channelName] = append(channel2segs[channelName], segmentID)
			}
		}

		qs.replica.updatePartitionState(dbID, collectionID, partitionID, querypb.PartitionState_PartialInMemory)
		for channel, segmentIDs := range channel2segs {
			sort.Slice(segmentIDs, func(i, j int) bool {
				return segmentStates[segmentIDs[i]].StartPosition.Timestamp < segmentStates[segmentIDs[j]].StartPosition.Timestamp
			})
			var channelLoadDone = false
			for _, node := range qs.queryNodes {
				channels2node := node.dmChannelNames
				for _, ch := range channels2node {
					if channel == ch {
						channelLoadDone = true
						break
					}
				}
				if channelLoadDone {
					break
				}
			}
			if !channelLoadDone {
				states := make([]*datapb.SegmentStateInfo, 0)
				for _, id := range segmentIDs {
					states = append(states, segmentStates[id])
				}
				loadSegmentRequest := &querypb.LoadSegmentRequest{
					CollectionID:  collectionID,
					PartitionID:   partitionID,
					SegmentIDs:    segmentIDs,
					SegmentStates: states,
				}
				dmChannels := []string{channel}
				nodeIDs := qs.shuffleChannelsToQueryNode(dmChannels)
				err = qs.watchDmChannels(dmChannels, nodeIDs, collection)
				if err != nil {
					return fn(err), err
				}
				queryNode := qs.queryNodes[nodeIDs[0]]
				//TODO:: seek when loadSegment may cause more msgs consumed
				status, err := queryNode.LoadSegments(loadSegmentRequest)
				if err != nil {
					return status, err
				}
			}
		}

		qs.replica.updatePartitionState(dbID, collectionID, partitionID, querypb.PartitionState_InMemory)
	}

	if len(collection.dmChannelNames) == 0 {
		channelRequest := datapb.InsertChannelRequest{
			DbID:         dbID,
			CollectionID: collectionID,
		}

		dmChannels, err := qs.dataServiceClient.GetInsertChannels(&channelRequest)
		if err != nil {
			return fn(err), err
		}
		for _, partitionID := range partitionIDs {
			loadSegmentRequest := &querypb.LoadSegmentRequest{
				CollectionID: collectionID,
				PartitionID:  partitionID,
			}
			for _, node := range qs.queryNodes {
				_, err := node.LoadSegments(loadSegmentRequest)
				if err != nil {
					return fn(err), nil
				}
			}
			nodeIDs := qs.shuffleChannelsToQueryNode(dmChannels)
			err = qs.watchDmChannels(dmChannels, nodeIDs, collection)
		}
		if err != nil {
			return fn(err), err
		}
	}

	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_SUCCESS,
	}, nil
}

func (qs *QueryService) ReleasePartitions(req *querypb.ReleasePartitionRequest) (*commonpb.Status, error) {
	dbID := req.DbID
	collectionID := req.CollectionID
	partitionIDs := req.PartitionIDs
	segmentIDs := make([]UniqueID, 0)
	for _, partitionID := range partitionIDs {
		segments, err := qs.replica.getSegments(dbID, collectionID, partitionID)
		if err != nil {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			}, err
		}
		res := make([]UniqueID, 0)
		for _, segment := range segments {
			res = append(res, segment.id)
		}

		segmentIDs = append(segmentIDs, res...)
		err = qs.replica.releasePartition(dbID, collectionID, partitionID)
		if err != nil {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			}, err
		}
	}
	releaseSegmentRequest := &querypb.ReleaseSegmentRequest{
		Base:         req.Base,
		DbID:         dbID,
		CollectionID: collectionID,
		PartitionIDs: partitionIDs,
		SegmentIDs:   segmentIDs,
	}

	for _, node := range qs.queryNodes {
		status, err := node.client.ReleaseSegments(releaseSegmentRequest)
		if err != nil {
			return status, err
		}
	}

	//TODO:: queryNode cancel subscribe dmChannels
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_SUCCESS,
	}, nil
}

func (qs *QueryService) CreateQueryChannel() (*querypb.CreateQueryChannelResponse, error) {
	channelID := qs.numQueryChannel
	qs.numQueryChannel++
	allocatedQueryChannel := "query-" + strconv.FormatInt(int64(channelID), 10)
	allocatedQueryResultChannel := "queryResult-" + strconv.FormatInt(int64(channelID), 10)

	//TODO:: query node watch query channels
	return &querypb.CreateQueryChannelResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		RequestChannel: allocatedQueryChannel,
		ResultChannel:  allocatedQueryResultChannel,
	}, nil
}

func (qs *QueryService) GetPartitionStates(req *querypb.PartitionStatesRequest) (*querypb.PartitionStatesResponse, error) {
	states, err := qs.replica.getPartitionStates(req.DbID, req.CollectionID, req.PartitionIDs)
	if err != nil {
		return &querypb.PartitionStatesResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			PartitionDescriptions: states,
		}, err
	}
	return &querypb.PartitionStatesResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		PartitionDescriptions: states,
	}, nil
}

func (qs *QueryService) GetSegmentInfo(req *querypb.SegmentInfoRequest) (*querypb.SegmentInfoResponse, error) {
	segmentInfos := make([]*querypb.SegmentInfo, 0)
	for _, node := range qs.queryNodes {
		segmentInfo, err := node.client.GetSegmentInfo(req)
		if err != nil {
			return &querypb.SegmentInfoResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
					Reason:    err.Error(),
				},
			}, err
		}
		segmentInfos = append(segmentInfos, segmentInfo.Infos...)
	}
	return &querypb.SegmentInfoResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		Infos: segmentInfos,
	}, nil
}

func NewQueryService(ctx context.Context) (*QueryService, error) {
	nodes := make(map[UniqueID]*queryNodeInfo)
	ctx1, cancel := context.WithCancel(ctx)
	replica := newMetaReplica()
	service := &QueryService{
		loopCtx:         ctx1,
		loopCancel:      cancel,
		replica:         replica,
		queryNodes:      nodes,
		numRegisterNode: 0,
		numQueryChannel: 0,
		enableGrpc:      false,
	}
	service.stateCode.Store(internalpb2.StateCode_INITIALIZING)
	service.isInit.Store(false)
	return service, nil
}

func (qs *QueryService) SetMasterService(masterService MasterServiceInterface) {
	qs.masterServiceClient = masterService
}

func (qs *QueryService) SetDataService(dataService DataServiceInterface) {
	qs.dataServiceClient = dataService
}

func (qs *QueryService) SetEnableGrpc(en bool) {
	qs.enableGrpc = en
}

func (qs *QueryService) watchDmChannels(dmChannels []string, assignedNodeIDs []UniqueID, collection *collection) error {
	err := qs.replica.addDmChannels(collection.id, dmChannels)
	if err != nil {
		return err
	}
	node2channels := make(map[UniqueID][]string)
	for i, channel := range dmChannels {
		nodeID := assignedNodeIDs[i]
		findChannel := false
		for _, ch := range collection.dmChannelNames {
			if channel == ch {
				findChannel = true
			}
		}
		if !findChannel {
			if _, ok := node2channels[nodeID]; ok {
				node2channels[nodeID] = append(node2channels[nodeID], channel)
			} else {
				channels := make([]string, 0)
				channels = append(channels, channel)
				node2channels[nodeID] = channels
			}
		}
	}

	for nodeID, channels := range node2channels {
		node := qs.queryNodes[nodeID]
		request := &querypb.WatchDmChannelsRequest{
			ChannelIDs: channels,
		}
		_, err := node.WatchDmChannels(request)
		node.AddDmChannels(channels)
		if err != nil {
			return err
		}
	}

	return nil
}

func (qs *QueryService) shuffleChannelsToQueryNode(dmChannels []string) []UniqueID {
	maxNumDMChannel := 0
	res := make([]UniqueID, 0)
	node2lens := make(map[UniqueID]int)
	for id, node := range qs.queryNodes {
		node2lens[id] = len(node.dmChannelNames)
	}
	offset := 0
	for {
		lastOffset := offset
		for id, len := range node2lens {
			if len >= maxNumDMChannel {
				maxNumDMChannel = len
			} else {
				res = append(res, id)
				node2lens[id]++
				offset++
			}
		}
		if lastOffset == offset {
			for id := range node2lens {
				res = append(res, id)
				node2lens[id]++
				offset++
				break
			}
		}
		if offset == len(dmChannels) {
			break
		}
	}
	return res
}
