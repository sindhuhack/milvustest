package datacoord

import (
	"context"
	"fmt"
	"sync"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
)

type VshardMeta interface {
	GetVShardInfo(partitionID int64, channel string) []*datapb.VShardInfo
	SaveVShardInfos([]*datapb.VShardInfo) error
	DropVShardInfo(*datapb.VShardInfo) error

	SaveReVShardTask(*datapb.ReVShardTask) error
	DropReVShardTask(*datapb.ReVShardTask) error
	GetReVShardTasksByPartition(int64) []*datapb.ReVShardTask
	GetReVShardTaskByID(int64, int64) *datapb.ReVShardTask

	SaveVShardInfosAndReVshardTask([]*datapb.VShardInfo, *datapb.ReVShardTask) error
}

var _ VshardMeta = (*VshardMetaImpl)(nil)

type VshardMetaImpl struct {
	sync.RWMutex
	ctx                context.Context
	catalog            metastore.DataCoordCatalog
	vshardsCache       map[string]map[string]*datapb.VShardInfo // partitionID+channel -> vshardDesc -> VShardInfo
	revshardTasksCache map[int64]map[int64]*datapb.ReVShardTask // partitionID -> taskID -> task
}

func newVshardMetaImpl(ctx context.Context, catalog metastore.DataCoordCatalog) (*VshardMetaImpl, error) {
	meta := &VshardMetaImpl{
		RWMutex:            sync.RWMutex{},
		ctx:                ctx,
		catalog:            catalog,
		vshardsCache:       make(map[string]map[string]*datapb.VShardInfo),
		revshardTasksCache: make(map[int64]map[int64]*datapb.ReVShardTask),
	}
	if err := meta.reloadFromKV(); err != nil {
		return nil, err
	}
	return meta, nil
}

func vShardInfoGroupKey(vshard *datapb.VShardInfo) string {
	return fmt.Sprintf("%d-%s", vshard.PartitionId, vshard.GetVchannel())
}

func (m *VshardMetaImpl) reloadFromKV() error {
	record := timerecord.NewTimeRecorder("VshardMeta-reloadFromKV")

	vshardMetas, err := m.catalog.ListVShardInfos(m.ctx)
	if err != nil {
		return err
	}

	for _, vshard := range vshardMetas {
		key := vShardInfoGroupKey(vshard)
		_, exist := m.vshardsCache[key]
		if !exist {
			m.vshardsCache[key] = make(map[string]*datapb.VShardInfo, 0)
		}
		m.vshardsCache[key][vshard.VshardDesc.String()] = vshard
	}

	tasks, err := m.catalog.ListReVShardTasks(m.ctx)
	if err != nil {
		return err
	}

	for _, task := range tasks {
		_, exist := m.revshardTasksCache[task.GetPartitionId()]
		if !exist {
			m.revshardTasksCache[task.GetPartitionId()] = make(map[int64]*datapb.ReVShardTask, 0)
		}
		m.revshardTasksCache[task.GetPartitionId()][task.GetId()] = task
	}

	log.Info("DataCoord VshardMetaImpl reloadFromKV done", zap.Duration("duration", record.ElapseSpan()))
	return nil
}

func (m *VshardMetaImpl) SaveVShardInfos(vshards []*datapb.VShardInfo) error {
	m.Lock()
	defer m.Unlock()
	if err := m.catalog.SaveVShardInfos(m.ctx, vshards); err != nil {
		log.Error("meta update: update VshardMeta info fail", zap.Error(err))
		return err
	}
	for _, vshard := range vshards {
		key := vShardInfoGroupKey(vshard)
		_, exist := m.vshardsCache[key]
		if !exist {
			m.vshardsCache[key] = make(map[string]*datapb.VShardInfo, 0)
		}
		m.vshardsCache[key][vshard.VshardDesc.String()] = vshard
	}
	return nil
}

func (m *VshardMetaImpl) DropVShardInfo(vshard *datapb.VShardInfo) error {
	m.Lock()
	defer m.Unlock()
	if err := m.catalog.DropVShardInfo(m.ctx, vshard); err != nil {
		log.Error("meta update: drop partitionVShardInfo info fail",
			zap.Int64("collectionID", vshard.GetCollectionId()),
			zap.Int64("partitionID", vshard.GetPartitionId()),
			zap.String("vchannel", vshard.GetVchannel()),
			zap.Error(err))
		return err
	}
	delete(m.vshardsCache, vShardInfoGroupKey(vshard))
	return nil
}

func (m *VshardMetaImpl) GetVShardInfo(partitionID int64, channel string) []*datapb.VShardInfo {
	m.Lock()
	defer m.Unlock()
	key := fmt.Sprintf("%d-%s", partitionID, channel)
	partitionVshardsMap, exist := m.vshardsCache[key]
	if !exist {
		return nil
	}
	partitionVshards := make([]*datapb.VShardInfo, 0, len(partitionVshardsMap))
	for _, value := range partitionVshardsMap {
		partitionVshards = append(partitionVshards, value)
	}
	return partitionVshards
}

func (m *VshardMetaImpl) SaveReVShardTask(task *datapb.ReVShardTask) error {
	m.Lock()
	defer m.Unlock()
	if err := m.catalog.SaveReVShardTask(m.ctx, task); err != nil {
		log.Error("meta update: update VshardMeta info fail", zap.Error(err))
		return err
	}
	_, exist := m.revshardTasksCache[task.GetPartitionId()]
	if !exist {
		m.revshardTasksCache[task.GetPartitionId()] = make(map[int64]*datapb.ReVShardTask, 0)
	}
	m.revshardTasksCache[task.GetPartitionId()][task.GetId()] = task
	return nil
}

func (m *VshardMetaImpl) DropReVShardTask(vshard *datapb.ReVShardTask) error {
	m.Lock()
	defer m.Unlock()
	if err := m.catalog.DropReVShardTask(m.ctx, vshard); err != nil {
		log.Error("meta update: drop ReVShardTask info fail",
			zap.Int64("collectionID", vshard.GetCollectionId()),
			zap.Int64("partitionID", vshard.GetPartitionId()),
			zap.String("vchannel", vshard.GetVchannel()),
			zap.Int64("id", vshard.GetId()),
			zap.Error(err))
		return err
	}

	if _, exist := m.revshardTasksCache[vshard.GetPartitionId()]; exist {
		delete(m.revshardTasksCache[vshard.GetPartitionId()], vshard.GetId())
	}
	if len(m.revshardTasksCache[vshard.GetPartitionId()]) == 0 {
		delete(m.revshardTasksCache, vshard.GetPartitionId())
	}
	return nil
}

func (m *VshardMetaImpl) GetReVShardTasksByPartition(partitionID int64) []*datapb.ReVShardTask {
	m.Lock()
	defer m.Unlock()
	partitionVshardsMap, exist := m.revshardTasksCache[partitionID]
	if !exist {
		return nil
	}
	partitionVshards := make([]*datapb.ReVShardTask, 0, len(partitionVshardsMap))
	for _, value := range partitionVshardsMap {
		partitionVshards = append(partitionVshards, value)
	}
	return partitionVshards
}

func (m *VshardMetaImpl) GetReVShardTaskByID(partitionID, taskID int64) *datapb.ReVShardTask {
	m.Lock()
	defer m.Unlock()
	partitionVshardsMap, exist := m.revshardTasksCache[partitionID]
	if !exist {
		return nil
	}
	for id, value := range partitionVshardsMap {
		if id == taskID {
			return value
		}
	}
	return nil
}

func (m *VshardMetaImpl) SaveVShardInfosAndReVshardTask(vshards []*datapb.VShardInfo, task *datapb.ReVShardTask) error {
	m.Lock()
	defer m.Unlock()
	if err := m.catalog.SaveVShardInfosAndReVShardTasks(m.ctx, vshards, task); err != nil {
		log.Error("meta update: update VshardInfo and ReVshardTask fail", zap.Error(err))
		return err
	}
	// update vshard info cache
	for _, vshard := range vshards {
		key := vShardInfoGroupKey(vshard)
		_, exist := m.vshardsCache[key]
		if !exist {
			m.vshardsCache[key] = make(map[string]*datapb.VShardInfo, 0)
		}
		m.vshardsCache[key][vshard.VshardDesc.String()] = vshard
	}
	// updatte revshardTask cache
	_, exist := m.revshardTasksCache[task.GetPartitionId()]
	if !exist {
		m.revshardTasksCache[task.GetPartitionId()] = make(map[int64]*datapb.ReVShardTask, 0)
	}
	m.revshardTasksCache[task.GetPartitionId()][task.GetId()] = task
	return nil
}
