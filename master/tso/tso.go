// Copyright 2016 TiKV Project Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.

package tso

import (
	"path"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/czs007/suvlim/errors"
	"github.com/czs007/suvlim/master/election"
	"github.com/czs007/suvlim/util/etcdutil"
	"github.com/czs007/suvlim/util/tsoutil"
	"github.com/czs007/suvlim/util/typeutil"
	"github.com/pingcap/log"
	"go.etcd.io/etcd/clientv3"
	"go.uber.org/zap"
)

const (
	// UpdateTimestampStep is used to update timestamp.
	UpdateTimestampStep = 50 * time.Millisecond
	// updateTimestampGuard is the min timestamp interval.
	updateTimestampGuard = time.Millisecond
	// maxLogical is the max upper limit for logical time.
	// When a TSO's logical time reaches this limit,
	// the physical time will be forced to increase.
	maxLogical = int64(1 << 18)
)

// atomicObject is used to store the current TSO in memory.
type atomicObject struct {
	physical time.Time
	logical  int64
}

// timestampOracle is used to maintain the logic of tso.
type timestampOracle struct {
	client   *clientv3.Client
	rootPath string
	// TODO: remove saveInterval
	saveInterval  time.Duration
	maxResetTSGap func() time.Duration
	// For tso, set after the PD becomes a leader.
	TSO           unsafe.Pointer
	lastSavedTime atomic.Value
}

func (t *timestampOracle) getTimestampPath() string {
	return path.Join(t.rootPath, "timestamp")
}

func (t *timestampOracle) loadTimestamp() (time.Time, error) {
	data, err := etcdutil.GetValue(t.client, t.getTimestampPath())
	if err != nil {
		return typeutil.ZeroTime, err
	}
	if len(data) == 0 {
		return typeutil.ZeroTime, nil
	}
	return typeutil.ParseTimestamp(data)
}

// save timestamp, if lastTs is 0, we think the timestamp doesn't exist, so create it,
// otherwise, update it.
func (t *timestampOracle) saveTimestamp(leadership *election.Leadership, ts time.Time) error {
	key := t.getTimestampPath()
	data := typeutil.Uint64ToBytes(uint64(ts.UnixNano()))
	resp, err := leadership.LeaderTxn().
		Then(clientv3.OpPut(key, string(data))).
		Commit()
	if err != nil {
		return errors.WithStack(err)
	}
	if !resp.Succeeded {
		return errors.New("save timestamp failed, maybe we lost leader")
	}

	t.lastSavedTime.Store(ts)

	return nil
}

// SyncTimestamp is used to synchronize the timestamp.
func (t *timestampOracle) SyncTimestamp(leadership *election.Leadership) error {

	last, err := t.loadTimestamp()
	if err != nil {
		return err
	}

	next := time.Now()

	// If the current system time minus the saved etcd timestamp is less than `updateTimestampGuard`,
	// the timestamp allocation will start from the saved etcd timestamp temporarily.
	if typeutil.SubTimeByWallClock(next, last) < updateTimestampGuard {
		next = last.Add(updateTimestampGuard)
	}

	save := next.Add(t.saveInterval)
	if err = t.saveTimestamp(leadership, save); err != nil {
		return err
	}

	log.Info("sync and save timestamp", zap.Time("last", last), zap.Time("save", save), zap.Time("next", next))

	current := &atomicObject{
		physical: next,
	}
	atomic.StorePointer(&t.TSO, unsafe.Pointer(current))

	return nil
}

// ResetUserTimestamp update the physical part with specified tso.
func (t *timestampOracle) ResetUserTimestamp(leadership *election.Leadership, tso uint64) error {
	if !leadership.Check() {
		return errors.New("Setup timestamp failed, lease expired")
	}
	physical, _ := tsoutil.ParseTS(tso)
	next := physical.Add(time.Millisecond)
	prev := (*atomicObject)(atomic.LoadPointer(&t.TSO))

	// do not update
	if typeutil.SubTimeByWallClock(next, prev.physical) <= 3*updateTimestampGuard {
		return errors.New("the specified ts too small than now")
	}

	if typeutil.SubTimeByWallClock(next, prev.physical) >= t.maxResetTSGap() {
		return errors.New("the specified ts too large than now")
	}

	save := next.Add(t.saveInterval)
	if err := t.saveTimestamp(leadership, save); err != nil {
		return err
	}
	update := &atomicObject{
		physical: next,
	}
	atomic.CompareAndSwapPointer(&t.TSO, unsafe.Pointer(prev), unsafe.Pointer(update))
	return nil
}

// UpdateTimestamp is used to update the timestamp.
// This function will do two things:
// 1. When the logical time is going to be used up, increase the current physical time.
// 2. When the time window is not big enough, which means the saved etcd time minus the next physical time
//    will be less than or equal to `updateTimestampGuard`, then the time window needs to be updated and
//    we also need to save the next physical time plus `TsoSaveInterval` into etcd.
//
// Here is some constraints that this function must satisfy:
// 1. The saved time is monotonically increasing.
// 2. The physical time is monotonically increasing.
// 3. The physical time is always less than the saved timestamp.
func (t *timestampOracle) UpdateTimestamp(leadership *election.Leadership) error {
	prev := (*atomicObject)(atomic.LoadPointer(&t.TSO))
	now := time.Now()

	jetLag := typeutil.SubTimeByWallClock(now, prev.physical)
	if jetLag > 3*UpdateTimestampStep {
		log.Warn("clock offset", zap.Duration("jet-lag", jetLag), zap.Time("prev-physical", prev.physical), zap.Time("now", now))
	}

	var next time.Time
	prevLogical := atomic.LoadInt64(&prev.logical)
	// If the system time is greater, it will be synchronized with the system time.
	if jetLag > updateTimestampGuard {
		next = now
	} else if prevLogical > maxLogical/2 {
		// The reason choosing maxLogical/2 here is that it's big enough for common cases.
		// Because there is enough timestamp can be allocated before next update.
		log.Warn("the logical time may be not enough", zap.Int64("prev-logical", prevLogical))
		next = prev.physical.Add(time.Millisecond)
	} else {
		// It will still use the previous physical time to alloc the timestamp.
		return nil
	}

	// It is not safe to increase the physical time to `next`.
	// The time window needs to be updated and saved to etcd.
	if typeutil.SubTimeByWallClock(t.lastSavedTime.Load().(time.Time), next) <= updateTimestampGuard {
		save := next.Add(t.saveInterval)
		if err := t.saveTimestamp(leadership, save); err != nil {
			return err
		}
	}

	current := &atomicObject{
		physical: next,
		logical:  0,
	}

	atomic.StorePointer(&t.TSO, unsafe.Pointer(current))

	return nil
}

// ResetTimestamp is used to reset the timestamp.
func (t *timestampOracle) ResetTimestamp() {
	zero := &atomicObject{
		physical: typeutil.ZeroTime,
	}
	atomic.StorePointer(&t.TSO, unsafe.Pointer(zero))
}
