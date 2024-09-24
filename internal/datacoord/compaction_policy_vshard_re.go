package datacoord

import (
	"context"
	"fmt"
	"time"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
)

type revshardCompactionPolicy struct {
	meta          *meta
	allocator     allocator.Allocator
	handler       Handler
	vshardManager VshardManager
}

func newReVshardCompactionPolicy(
	meta *meta,
	allocator allocator.Allocator,
	handler Handler,
	vshardManager VshardManager) *revshardCompactionPolicy {
	return &revshardCompactionPolicy{
		meta:          meta,
		allocator:     allocator,
		handler:       handler,
		vshardManager: vshardManager,
	}
}

func (policy *revshardCompactionPolicy) Enable() bool {
	// todo add a new config
	return Params.DataCoordCfg.EnableAutoCompaction.GetAsBool()
}

func (policy *revshardCompactionPolicy) Trigger() (map[CompactionTriggerType][]CompactionView, error) {
	log.Info("start trigger revshard compaction...")
	ctx := context.Background()
	collections := policy.meta.GetCollections()

	events := make(map[CompactionTriggerType][]CompactionView, 0)
	views := make([]CompactionView, 0)
	for _, collection := range collections {
		collectionInfo, err := policy.handler.GetCollection(ctx, collection.ID)
		if err != nil {
			log.Warn("fail to get collection from handler")
			return nil, err
		}
		if collectionInfo == nil {
			log.Warn("collection not exist")
			return nil, nil
		}
		if !isCollectionAutoCompactionEnabled(collectionInfo) {
			log.RatedInfo(20, "collection auto compaction disabled")
			return nil, nil
		}

		for _, partitionID := range collectionInfo.Partitions {
			collectionViews, _, err := policy.triggerOnePartition(ctx, collectionInfo, partitionID)
			if err != nil {
				// not throw this error because no need to fail because of one partition
				log.Warn("fail to trigger revshard compaction", zap.Int64("collectionID", collection.ID), zap.Int64("partitionID", partitionID), zap.Error(err))
			}
			views = append(views, collectionViews...)
		}
	}
	events[TriggerTypeReVshard] = views
	return events, nil
}

func (policy *revshardCompactionPolicy) triggerOnePartition(ctx context.Context, collectionInfo *collectionInfo, partitionID int64) ([]CompactionView, int64, error) {
	log := log.With(zap.Int64("collectionID", collectionInfo.ID), zap.Int64("partitionID", partitionID))

	reVShardTasks := policy.vshardManager.GetReVShardTasks(partitionID)
	reVShardTasks = lo.Filter(reVShardTasks, func(V *datapb.ReVShardTask, _ int) bool {
		return V.GetState() == datapb.ReVShardState_ReVShard_created
	})
	if len(reVShardTasks) == 0 {
		log.Info("No revshard task in the partition, skip trigger revshard compaction")
		return nil, 0, nil
	}
	log.Info("start trigger revshard compaction")

	containsSegmentVshardFunc := func(set []*datapb.VShardDesc, target *datapb.VShardDesc) bool {
		if target == nil {
			return false
		}
		for _, desc := range set {
			if target.String() == desc.String() {
				return true
			}
		}
		return false
	}

	newTriggerID, err := policy.allocator.AllocID(ctx)
	if err != nil {
		log.Warn("fail to allocate triggerID", zap.Error(err))
		return nil, 0, err
	}

	views := make([]CompactionView, 0)
	for _, task := range reVShardTasks {
		partSegments := policy.meta.SelectSegments(SegmentFilterFunc(func(segment *SegmentInfo) bool {
			return segment.CollectionID == collectionInfo.ID &&
				segment.PartitionID == partitionID &&
				segment.InsertChannel == task.Vchannel &&
				isSegmentHealthy(segment) &&
				isFlush(segment) &&
				!segment.isCompacting && // not compacting now
				!segment.GetIsImporting() && // not importing now
				segment.GetLevel() != datapb.SegmentLevel_L0 &&
				containsSegmentVshardFunc(task.GetFrom(), segment.GetVshardDesc())
		}))

		if Params.DataCoordCfg.IndexBasedCompaction.GetAsBool() {
			partSegments = FilterInIndexedSegments(policy.handler, policy.meta, partSegments...)
		}

		collectionTTL, err := getCollectionTTL(collectionInfo.Properties)
		if err != nil {
			log.Warn("get collection ttl failed, skip to handle compaction")
			return make([]CompactionView, 0), 0, err
		}

		log.Info("generate view", zap.Int("segmentNum", len(partSegments)))
		if len(partSegments) == 0 {
			continue
		}
		view := &VshardSegmentView{
			label: &CompactionGroupLabel{
				CollectionID: collectionInfo.ID,
				PartitionID:  partitionID,
				Channel:      task.GetVchannel(),
			},
			segments:       GetViewsByInfo(partSegments...),
			collectionTTL:  collectionTTL,
			triggerID:      newTriggerID,
			fromVshards:    task.GetFrom(),
			toVshards:      task.GetTo(),
			revshardTaskId: task.GetId(),
		}
		views = append(views, view)
	}

	log.Info("finish trigger revshard compaction", zap.Int("viewNum", len(views)))
	return views, newTriggerID, nil
}

var _ CompactionView = (*VshardSegmentView)(nil)

type VshardSegmentView struct {
	label          *CompactionGroupLabel
	segments       []*SegmentView
	collectionTTL  time.Duration
	triggerID      int64
	fromVshards    []*datapb.VShardDesc
	toVshards      []*datapb.VShardDesc
	revshardTaskId int64
}

func (v *VshardSegmentView) GetGroupLabel() *CompactionGroupLabel {
	if v == nil {
		return &CompactionGroupLabel{}
	}
	return v.label
}

func (v *VshardSegmentView) GetSegmentsView() []*SegmentView {
	if v == nil {
		return nil
	}

	return v.segments
}

func (v *VshardSegmentView) Append(segments ...*SegmentView) {
	if v.segments == nil {
		v.segments = segments
		return
	}

	v.segments = append(v.segments, segments...)
}

func (v *VshardSegmentView) String() string {
	strs := lo.Map(v.segments, func(segView *SegmentView, _ int) string {
		return segView.String()
	})
	return fmt.Sprintf("label=<%s>, segmentNum=%d segments=%v", v.label.String(), len(v.segments), strs)
}

func (v *VshardSegmentView) Trigger() (CompactionView, string) {
	return v, ""
}

func (v *VshardSegmentView) ForceTrigger() (CompactionView, string) {
	panic("implement me")
}
