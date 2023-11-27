// Code generated by mockery v2.32.4. DO NOT EDIT.

package writebuffer

import (
	context "context"

	metacache "github.com/milvus-io/milvus/internal/datanode/metacache"
	mock "github.com/stretchr/testify/mock"

	msgpb "github.com/milvus-io/milvus-proto/go-api/v2/msgpb"

	msgstream "github.com/milvus-io/milvus/pkg/mq/msgstream"
)

// MockBufferManager is an autogenerated mock type for the BufferManager type
type MockBufferManager struct {
	mock.Mock
}

type MockBufferManager_Expecter struct {
	mock *mock.Mock
}

func (_m *MockBufferManager) EXPECT() *MockBufferManager_Expecter {
	return &MockBufferManager_Expecter{mock: &_m.Mock}
}

// BufferData provides a mock function with given fields: channel, insertMsgs, deleteMsgs, startPos, endPos
func (_m *MockBufferManager) BufferData(channel string, insertMsgs []*msgstream.InsertMsg, deleteMsgs []*msgstream.DeleteMsg, startPos *msgpb.MsgPosition, endPos *msgpb.MsgPosition) error {
	ret := _m.Called(channel, insertMsgs, deleteMsgs, startPos, endPos)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, []*msgstream.InsertMsg, []*msgstream.DeleteMsg, *msgpb.MsgPosition, *msgpb.MsgPosition) error); ok {
		r0 = rf(channel, insertMsgs, deleteMsgs, startPos, endPos)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockBufferManager_BufferData_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'BufferData'
type MockBufferManager_BufferData_Call struct {
	*mock.Call
}

// BufferData is a helper method to define mock.On call
//   - channel string
//   - insertMsgs []*msgstream.InsertMsg
//   - deleteMsgs []*msgstream.DeleteMsg
//   - startPos *msgpb.MsgPosition
//   - endPos *msgpb.MsgPosition
func (_e *MockBufferManager_Expecter) BufferData(channel interface{}, insertMsgs interface{}, deleteMsgs interface{}, startPos interface{}, endPos interface{}) *MockBufferManager_BufferData_Call {
	return &MockBufferManager_BufferData_Call{Call: _e.mock.On("BufferData", channel, insertMsgs, deleteMsgs, startPos, endPos)}
}

func (_c *MockBufferManager_BufferData_Call) Run(run func(channel string, insertMsgs []*msgstream.InsertMsg, deleteMsgs []*msgstream.DeleteMsg, startPos *msgpb.MsgPosition, endPos *msgpb.MsgPosition)) *MockBufferManager_BufferData_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].([]*msgstream.InsertMsg), args[2].([]*msgstream.DeleteMsg), args[3].(*msgpb.MsgPosition), args[4].(*msgpb.MsgPosition))
	})
	return _c
}

func (_c *MockBufferManager_BufferData_Call) Return(_a0 error) *MockBufferManager_BufferData_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBufferManager_BufferData_Call) RunAndReturn(run func(string, []*msgstream.InsertMsg, []*msgstream.DeleteMsg, *msgpb.MsgPosition, *msgpb.MsgPosition) error) *MockBufferManager_BufferData_Call {
	_c.Call.Return(run)
	return _c
}

// DropChannel provides a mock function with given fields: channel
func (_m *MockBufferManager) DropChannel(channel string) {
	_m.Called(channel)
}

// MockBufferManager_DropChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'DropChannel'
type MockBufferManager_DropChannel_Call struct {
	*mock.Call
}

// DropChannel is a helper method to define mock.On call
//   - channel string
func (_e *MockBufferManager_Expecter) DropChannel(channel interface{}) *MockBufferManager_DropChannel_Call {
	return &MockBufferManager_DropChannel_Call{Call: _e.mock.On("DropChannel", channel)}
}

func (_c *MockBufferManager_DropChannel_Call) Run(run func(channel string)) *MockBufferManager_DropChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockBufferManager_DropChannel_Call) Return() *MockBufferManager_DropChannel_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockBufferManager_DropChannel_Call) RunAndReturn(run func(string)) *MockBufferManager_DropChannel_Call {
	_c.Call.Return(run)
	return _c
}

// FlushChannel provides a mock function with given fields: ctx, channel, flushTs
func (_m *MockBufferManager) FlushChannel(ctx context.Context, channel string, flushTs uint64) error {
	ret := _m.Called(ctx, channel, flushTs)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, string, uint64) error); ok {
		r0 = rf(ctx, channel, flushTs)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockBufferManager_FlushChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'FlushChannel'
type MockBufferManager_FlushChannel_Call struct {
	*mock.Call
}

// FlushChannel is a helper method to define mock.On call
//   - ctx context.Context
//   - channel string
//   - flushTs uint64
func (_e *MockBufferManager_Expecter) FlushChannel(ctx interface{}, channel interface{}, flushTs interface{}) *MockBufferManager_FlushChannel_Call {
	return &MockBufferManager_FlushChannel_Call{Call: _e.mock.On("FlushChannel", ctx, channel, flushTs)}
}

func (_c *MockBufferManager_FlushChannel_Call) Run(run func(ctx context.Context, channel string, flushTs uint64)) *MockBufferManager_FlushChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(string), args[2].(uint64))
	})
	return _c
}

func (_c *MockBufferManager_FlushChannel_Call) Return(_a0 error) *MockBufferManager_FlushChannel_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBufferManager_FlushChannel_Call) RunAndReturn(run func(context.Context, string, uint64) error) *MockBufferManager_FlushChannel_Call {
	_c.Call.Return(run)
	return _c
}

// FlushSegments provides a mock function with given fields: ctx, channel, segmentIDs
func (_m *MockBufferManager) FlushSegments(ctx context.Context, channel string, segmentIDs []int64) error {
	ret := _m.Called(ctx, channel, segmentIDs)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, string, []int64) error); ok {
		r0 = rf(ctx, channel, segmentIDs)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockBufferManager_FlushSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'FlushSegments'
type MockBufferManager_FlushSegments_Call struct {
	*mock.Call
}

// FlushSegments is a helper method to define mock.On call
//   - ctx context.Context
//   - channel string
//   - segmentIDs []int64
func (_e *MockBufferManager_Expecter) FlushSegments(ctx interface{}, channel interface{}, segmentIDs interface{}) *MockBufferManager_FlushSegments_Call {
	return &MockBufferManager_FlushSegments_Call{Call: _e.mock.On("FlushSegments", ctx, channel, segmentIDs)}
}

func (_c *MockBufferManager_FlushSegments_Call) Run(run func(ctx context.Context, channel string, segmentIDs []int64)) *MockBufferManager_FlushSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(string), args[2].([]int64))
	})
	return _c
}

func (_c *MockBufferManager_FlushSegments_Call) Return(_a0 error) *MockBufferManager_FlushSegments_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBufferManager_FlushSegments_Call) RunAndReturn(run func(context.Context, string, []int64) error) *MockBufferManager_FlushSegments_Call {
	_c.Call.Return(run)
	return _c
}

// GetCheckpoint provides a mock function with given fields: channel
func (_m *MockBufferManager) GetCheckpoint(channel string) (*msgpb.MsgPosition, bool, error) {
	ret := _m.Called(channel)

	var r0 *msgpb.MsgPosition
	var r1 bool
	var r2 error
	if rf, ok := ret.Get(0).(func(string) (*msgpb.MsgPosition, bool, error)); ok {
		return rf(channel)
	}
	if rf, ok := ret.Get(0).(func(string) *msgpb.MsgPosition); ok {
		r0 = rf(channel)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*msgpb.MsgPosition)
		}
	}

	if rf, ok := ret.Get(1).(func(string) bool); ok {
		r1 = rf(channel)
	} else {
		r1 = ret.Get(1).(bool)
	}

	if rf, ok := ret.Get(2).(func(string) error); ok {
		r2 = rf(channel)
	} else {
		r2 = ret.Error(2)
	}

	return r0, r1, r2
}

// MockBufferManager_GetCheckpoint_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetCheckpoint'
type MockBufferManager_GetCheckpoint_Call struct {
	*mock.Call
}

// GetCheckpoint is a helper method to define mock.On call
//   - channel string
func (_e *MockBufferManager_Expecter) GetCheckpoint(channel interface{}) *MockBufferManager_GetCheckpoint_Call {
	return &MockBufferManager_GetCheckpoint_Call{Call: _e.mock.On("GetCheckpoint", channel)}
}

func (_c *MockBufferManager_GetCheckpoint_Call) Run(run func(channel string)) *MockBufferManager_GetCheckpoint_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockBufferManager_GetCheckpoint_Call) Return(_a0 *msgpb.MsgPosition, _a1 bool, _a2 error) *MockBufferManager_GetCheckpoint_Call {
	_c.Call.Return(_a0, _a1, _a2)
	return _c
}

func (_c *MockBufferManager_GetCheckpoint_Call) RunAndReturn(run func(string) (*msgpb.MsgPosition, bool, error)) *MockBufferManager_GetCheckpoint_Call {
	_c.Call.Return(run)
	return _c
}

// NotifyCheckpointUpdated provides a mock function with given fields: channel, ts
func (_m *MockBufferManager) NotifyCheckpointUpdated(channel string, ts uint64) {
	_m.Called(channel, ts)
}

// MockBufferManager_NotifyCheckpointUpdated_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'NotifyCheckpointUpdated'
type MockBufferManager_NotifyCheckpointUpdated_Call struct {
	*mock.Call
}

// NotifyCheckpointUpdated is a helper method to define mock.On call
//   - channel string
//   - ts uint64
func (_e *MockBufferManager_Expecter) NotifyCheckpointUpdated(channel interface{}, ts interface{}) *MockBufferManager_NotifyCheckpointUpdated_Call {
	return &MockBufferManager_NotifyCheckpointUpdated_Call{Call: _e.mock.On("NotifyCheckpointUpdated", channel, ts)}
}

func (_c *MockBufferManager_NotifyCheckpointUpdated_Call) Run(run func(channel string, ts uint64)) *MockBufferManager_NotifyCheckpointUpdated_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(uint64))
	})
	return _c
}

func (_c *MockBufferManager_NotifyCheckpointUpdated_Call) Return() *MockBufferManager_NotifyCheckpointUpdated_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockBufferManager_NotifyCheckpointUpdated_Call) RunAndReturn(run func(string, uint64)) *MockBufferManager_NotifyCheckpointUpdated_Call {
	_c.Call.Return(run)
	return _c
}

// Register provides a mock function with given fields: channel, _a1, storageV2Cache, opts
func (_m *MockBufferManager) Register(channel string, _a1 metacache.MetaCache, storageV2Cache *metacache.StorageV2Cache, opts ...WriteBufferOption) error {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, channel, _a1, storageV2Cache)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, metacache.MetaCache, *metacache.StorageV2Cache, ...WriteBufferOption) error); ok {
		r0 = rf(channel, _a1, storageV2Cache, opts...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockBufferManager_Register_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Register'
type MockBufferManager_Register_Call struct {
	*mock.Call
}

// Register is a helper method to define mock.On call
//   - channel string
//   - _a1 metacache.MetaCache
//   - storageV2Cache *metacache.StorageV2Cache
//   - opts ...WriteBufferOption
func (_e *MockBufferManager_Expecter) Register(channel interface{}, _a1 interface{}, storageV2Cache interface{}, opts ...interface{}) *MockBufferManager_Register_Call {
	return &MockBufferManager_Register_Call{Call: _e.mock.On("Register",
		append([]interface{}{channel, _a1, storageV2Cache}, opts...)...)}
}

func (_c *MockBufferManager_Register_Call) Run(run func(channel string, _a1 metacache.MetaCache, storageV2Cache *metacache.StorageV2Cache, opts ...WriteBufferOption)) *MockBufferManager_Register_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]WriteBufferOption, len(args)-3)
		for i, a := range args[3:] {
			if a != nil {
				variadicArgs[i] = a.(WriteBufferOption)
			}
		}
		run(args[0].(string), args[1].(metacache.MetaCache), args[2].(*metacache.StorageV2Cache), variadicArgs...)
	})
	return _c
}

func (_c *MockBufferManager_Register_Call) Return(_a0 error) *MockBufferManager_Register_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockBufferManager_Register_Call) RunAndReturn(run func(string, metacache.MetaCache, *metacache.StorageV2Cache, ...WriteBufferOption) error) *MockBufferManager_Register_Call {
	_c.Call.Return(run)
	return _c
}

// RemoveChannel provides a mock function with given fields: channel
func (_m *MockBufferManager) RemoveChannel(channel string) {
	_m.Called(channel)
}

// MockBufferManager_RemoveChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RemoveChannel'
type MockBufferManager_RemoveChannel_Call struct {
	*mock.Call
}

// RemoveChannel is a helper method to define mock.On call
//   - channel string
func (_e *MockBufferManager_Expecter) RemoveChannel(channel interface{}) *MockBufferManager_RemoveChannel_Call {
	return &MockBufferManager_RemoveChannel_Call{Call: _e.mock.On("RemoveChannel", channel)}
}

func (_c *MockBufferManager_RemoveChannel_Call) Run(run func(channel string)) *MockBufferManager_RemoveChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockBufferManager_RemoveChannel_Call) Return() *MockBufferManager_RemoveChannel_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockBufferManager_RemoveChannel_Call) RunAndReturn(run func(string)) *MockBufferManager_RemoveChannel_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockBufferManager creates a new instance of MockBufferManager. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockBufferManager(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockBufferManager {
	mock := &MockBufferManager{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
