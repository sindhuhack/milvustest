// Code generated by mockery v2.32.4. DO NOT EDIT.

package datacoord

import mock "github.com/stretchr/testify/mock"

// MockRWChannelStore is an autogenerated mock type for the RWChannelStore type
type MockRWChannelStore struct {
	mock.Mock
}

type MockRWChannelStore_Expecter struct {
	mock *mock.Mock
}

func (_m *MockRWChannelStore) EXPECT() *MockRWChannelStore_Expecter {
	return &MockRWChannelStore_Expecter{mock: &_m.Mock}
}

// AddNode provides a mock function with given fields: nodeID
func (_m *MockRWChannelStore) AddNode(nodeID int64) {
	_m.Called(nodeID)
}

// MockRWChannelStore_AddNode_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AddNode'
type MockRWChannelStore_AddNode_Call struct {
	*mock.Call
}

// AddNode is a helper method to define mock.On call
//   - nodeID int64
func (_e *MockRWChannelStore_Expecter) AddNode(nodeID interface{}) *MockRWChannelStore_AddNode_Call {
	return &MockRWChannelStore_AddNode_Call{Call: _e.mock.On("AddNode", nodeID)}
}

func (_c *MockRWChannelStore_AddNode_Call) Run(run func(nodeID int64)) *MockRWChannelStore_AddNode_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockRWChannelStore_AddNode_Call) Return() *MockRWChannelStore_AddNode_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockRWChannelStore_AddNode_Call) RunAndReturn(run func(int64)) *MockRWChannelStore_AddNode_Call {
	_c.Call.Return(run)
	return _c
}

// GetBufferChannelInfo provides a mock function with given fields:
func (_m *MockRWChannelStore) GetBufferChannelInfo() *NodeChannelInfo {
	ret := _m.Called()

	var r0 *NodeChannelInfo
	if rf, ok := ret.Get(0).(func() *NodeChannelInfo); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*NodeChannelInfo)
		}
	}

	return r0
}

// MockRWChannelStore_GetBufferChannelInfo_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetBufferChannelInfo'
type MockRWChannelStore_GetBufferChannelInfo_Call struct {
	*mock.Call
}

// GetBufferChannelInfo is a helper method to define mock.On call
func (_e *MockRWChannelStore_Expecter) GetBufferChannelInfo() *MockRWChannelStore_GetBufferChannelInfo_Call {
	return &MockRWChannelStore_GetBufferChannelInfo_Call{Call: _e.mock.On("GetBufferChannelInfo")}
}

func (_c *MockRWChannelStore_GetBufferChannelInfo_Call) Run(run func()) *MockRWChannelStore_GetBufferChannelInfo_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockRWChannelStore_GetBufferChannelInfo_Call) Return(_a0 *NodeChannelInfo) *MockRWChannelStore_GetBufferChannelInfo_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_GetBufferChannelInfo_Call) RunAndReturn(run func() *NodeChannelInfo) *MockRWChannelStore_GetBufferChannelInfo_Call {
	_c.Call.Return(run)
	return _c
}

// GetNode provides a mock function with given fields: nodeID
func (_m *MockRWChannelStore) GetNode(nodeID int64) *NodeChannelInfo {
	ret := _m.Called(nodeID)

	var r0 *NodeChannelInfo
	if rf, ok := ret.Get(0).(func(int64) *NodeChannelInfo); ok {
		r0 = rf(nodeID)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*NodeChannelInfo)
		}
	}

	return r0
}

// MockRWChannelStore_GetNode_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetNode'
type MockRWChannelStore_GetNode_Call struct {
	*mock.Call
}

// GetNode is a helper method to define mock.On call
//   - nodeID int64
func (_e *MockRWChannelStore_Expecter) GetNode(nodeID interface{}) *MockRWChannelStore_GetNode_Call {
	return &MockRWChannelStore_GetNode_Call{Call: _e.mock.On("GetNode", nodeID)}
}

func (_c *MockRWChannelStore_GetNode_Call) Run(run func(nodeID int64)) *MockRWChannelStore_GetNode_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockRWChannelStore_GetNode_Call) Return(_a0 *NodeChannelInfo) *MockRWChannelStore_GetNode_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_GetNode_Call) RunAndReturn(run func(int64) *NodeChannelInfo) *MockRWChannelStore_GetNode_Call {
	_c.Call.Return(run)
	return _c
}

// GetNodeChannelCount provides a mock function with given fields: nodeID
func (_m *MockRWChannelStore) GetNodeChannelCount(nodeID int64) int {
	ret := _m.Called(nodeID)

	var r0 int
	if rf, ok := ret.Get(0).(func(int64) int); ok {
		r0 = rf(nodeID)
	} else {
		r0 = ret.Get(0).(int)
	}

	return r0
}

// MockRWChannelStore_GetNodeChannelCount_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetNodeChannelCount'
type MockRWChannelStore_GetNodeChannelCount_Call struct {
	*mock.Call
}

// GetNodeChannelCount is a helper method to define mock.On call
//   - nodeID int64
func (_e *MockRWChannelStore_Expecter) GetNodeChannelCount(nodeID interface{}) *MockRWChannelStore_GetNodeChannelCount_Call {
	return &MockRWChannelStore_GetNodeChannelCount_Call{Call: _e.mock.On("GetNodeChannelCount", nodeID)}
}

func (_c *MockRWChannelStore_GetNodeChannelCount_Call) Run(run func(nodeID int64)) *MockRWChannelStore_GetNodeChannelCount_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockRWChannelStore_GetNodeChannelCount_Call) Return(_a0 int) *MockRWChannelStore_GetNodeChannelCount_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_GetNodeChannelCount_Call) RunAndReturn(run func(int64) int) *MockRWChannelStore_GetNodeChannelCount_Call {
	_c.Call.Return(run)
	return _c
}

// GetNodeChannelsBy provides a mock function with given fields: nodeSelector, channelSelectors
func (_m *MockRWChannelStore) GetNodeChannelsBy(nodeSelector NodeSelector, channelSelectors ...ChannelSelector) []*NodeChannelInfo {
	_va := make([]interface{}, len(channelSelectors))
	for _i := range channelSelectors {
		_va[_i] = channelSelectors[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, nodeSelector)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 []*NodeChannelInfo
	if rf, ok := ret.Get(0).(func(NodeSelector, ...ChannelSelector) []*NodeChannelInfo); ok {
		r0 = rf(nodeSelector, channelSelectors...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*NodeChannelInfo)
		}
	}

	return r0
}

// MockRWChannelStore_GetNodeChannelsBy_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetNodeChannelsBy'
type MockRWChannelStore_GetNodeChannelsBy_Call struct {
	*mock.Call
}

// GetNodeChannelsBy is a helper method to define mock.On call
//   - nodeSelector NodeSelector
//   - channelSelectors ...ChannelSelector
func (_e *MockRWChannelStore_Expecter) GetNodeChannelsBy(nodeSelector interface{}, channelSelectors ...interface{}) *MockRWChannelStore_GetNodeChannelsBy_Call {
	return &MockRWChannelStore_GetNodeChannelsBy_Call{Call: _e.mock.On("GetNodeChannelsBy",
		append([]interface{}{nodeSelector}, channelSelectors...)...)}
}

func (_c *MockRWChannelStore_GetNodeChannelsBy_Call) Run(run func(nodeSelector NodeSelector, channelSelectors ...ChannelSelector)) *MockRWChannelStore_GetNodeChannelsBy_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]ChannelSelector, len(args)-1)
		for i, a := range args[1:] {
			if a != nil {
				variadicArgs[i] = a.(ChannelSelector)
			}
		}
		run(args[0].(NodeSelector), variadicArgs...)
	})
	return _c
}

func (_c *MockRWChannelStore_GetNodeChannelsBy_Call) Return(_a0 []*NodeChannelInfo) *MockRWChannelStore_GetNodeChannelsBy_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_GetNodeChannelsBy_Call) RunAndReturn(run func(NodeSelector, ...ChannelSelector) []*NodeChannelInfo) *MockRWChannelStore_GetNodeChannelsBy_Call {
	_c.Call.Return(run)
	return _c
}

// GetNodes provides a mock function with given fields:
func (_m *MockRWChannelStore) GetNodes() []int64 {
	ret := _m.Called()

	var r0 []int64
	if rf, ok := ret.Get(0).(func() []int64); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]int64)
		}
	}

	return r0
}

// MockRWChannelStore_GetNodes_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetNodes'
type MockRWChannelStore_GetNodes_Call struct {
	*mock.Call
}

// GetNodes is a helper method to define mock.On call
func (_e *MockRWChannelStore_Expecter) GetNodes() *MockRWChannelStore_GetNodes_Call {
	return &MockRWChannelStore_GetNodes_Call{Call: _e.mock.On("GetNodes")}
}

func (_c *MockRWChannelStore_GetNodes_Call) Run(run func()) *MockRWChannelStore_GetNodes_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockRWChannelStore_GetNodes_Call) Return(_a0 []int64) *MockRWChannelStore_GetNodes_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_GetNodes_Call) RunAndReturn(run func() []int64) *MockRWChannelStore_GetNodes_Call {
	_c.Call.Return(run)
	return _c
}

// GetNodesChannels provides a mock function with given fields:
func (_m *MockRWChannelStore) GetNodesChannels() []*NodeChannelInfo {
	ret := _m.Called()

	var r0 []*NodeChannelInfo
	if rf, ok := ret.Get(0).(func() []*NodeChannelInfo); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*NodeChannelInfo)
		}
	}

	return r0
}

// MockRWChannelStore_GetNodesChannels_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetNodesChannels'
type MockRWChannelStore_GetNodesChannels_Call struct {
	*mock.Call
}

// GetNodesChannels is a helper method to define mock.On call
func (_e *MockRWChannelStore_Expecter) GetNodesChannels() *MockRWChannelStore_GetNodesChannels_Call {
	return &MockRWChannelStore_GetNodesChannels_Call{Call: _e.mock.On("GetNodesChannels")}
}

func (_c *MockRWChannelStore_GetNodesChannels_Call) Run(run func()) *MockRWChannelStore_GetNodesChannels_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockRWChannelStore_GetNodesChannels_Call) Return(_a0 []*NodeChannelInfo) *MockRWChannelStore_GetNodesChannels_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_GetNodesChannels_Call) RunAndReturn(run func() []*NodeChannelInfo) *MockRWChannelStore_GetNodesChannels_Call {
	_c.Call.Return(run)
	return _c
}

// HasChannel provides a mock function with given fields: channel
func (_m *MockRWChannelStore) HasChannel(channel string) bool {
	ret := _m.Called(channel)

	var r0 bool
	if rf, ok := ret.Get(0).(func(string) bool); ok {
		r0 = rf(channel)
	} else {
		r0 = ret.Get(0).(bool)
	}

	return r0
}

// MockRWChannelStore_HasChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'HasChannel'
type MockRWChannelStore_HasChannel_Call struct {
	*mock.Call
}

// HasChannel is a helper method to define mock.On call
//   - channel string
func (_e *MockRWChannelStore_Expecter) HasChannel(channel interface{}) *MockRWChannelStore_HasChannel_Call {
	return &MockRWChannelStore_HasChannel_Call{Call: _e.mock.On("HasChannel", channel)}
}

func (_c *MockRWChannelStore_HasChannel_Call) Run(run func(channel string)) *MockRWChannelStore_HasChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockRWChannelStore_HasChannel_Call) Return(_a0 bool) *MockRWChannelStore_HasChannel_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_HasChannel_Call) RunAndReturn(run func(string) bool) *MockRWChannelStore_HasChannel_Call {
	_c.Call.Return(run)
	return _c
}

// Reload provides a mock function with given fields:
func (_m *MockRWChannelStore) Reload() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockRWChannelStore_Reload_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Reload'
type MockRWChannelStore_Reload_Call struct {
	*mock.Call
}

// Reload is a helper method to define mock.On call
func (_e *MockRWChannelStore_Expecter) Reload() *MockRWChannelStore_Reload_Call {
	return &MockRWChannelStore_Reload_Call{Call: _e.mock.On("Reload")}
}

func (_c *MockRWChannelStore_Reload_Call) Run(run func()) *MockRWChannelStore_Reload_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockRWChannelStore_Reload_Call) Return(_a0 error) *MockRWChannelStore_Reload_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_Reload_Call) RunAndReturn(run func() error) *MockRWChannelStore_Reload_Call {
	_c.Call.Return(run)
	return _c
}

// RemoveNode provides a mock function with given fields: nodeID
func (_m *MockRWChannelStore) RemoveNode(nodeID int64) {
	_m.Called(nodeID)
}

// MockRWChannelStore_RemoveNode_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RemoveNode'
type MockRWChannelStore_RemoveNode_Call struct {
	*mock.Call
}

// RemoveNode is a helper method to define mock.On call
//   - nodeID int64
func (_e *MockRWChannelStore_Expecter) RemoveNode(nodeID interface{}) *MockRWChannelStore_RemoveNode_Call {
	return &MockRWChannelStore_RemoveNode_Call{Call: _e.mock.On("RemoveNode", nodeID)}
}

func (_c *MockRWChannelStore_RemoveNode_Call) Run(run func(nodeID int64)) *MockRWChannelStore_RemoveNode_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockRWChannelStore_RemoveNode_Call) Return() *MockRWChannelStore_RemoveNode_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockRWChannelStore_RemoveNode_Call) RunAndReturn(run func(int64)) *MockRWChannelStore_RemoveNode_Call {
	_c.Call.Return(run)
	return _c
}

// SetLegacyChannelByNode provides a mock function with given fields: nodeIDs
func (_m *MockRWChannelStore) SetLegacyChannelByNode(nodeIDs ...int64) {
	_va := make([]interface{}, len(nodeIDs))
	for _i := range nodeIDs {
		_va[_i] = nodeIDs[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, _va...)
	_m.Called(_ca...)
}

// MockRWChannelStore_SetLegacyChannelByNode_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetLegacyChannelByNode'
type MockRWChannelStore_SetLegacyChannelByNode_Call struct {
	*mock.Call
}

// SetLegacyChannelByNode is a helper method to define mock.On call
//   - nodeIDs ...int64
func (_e *MockRWChannelStore_Expecter) SetLegacyChannelByNode(nodeIDs ...interface{}) *MockRWChannelStore_SetLegacyChannelByNode_Call {
	return &MockRWChannelStore_SetLegacyChannelByNode_Call{Call: _e.mock.On("SetLegacyChannelByNode",
		append([]interface{}{}, nodeIDs...)...)}
}

func (_c *MockRWChannelStore_SetLegacyChannelByNode_Call) Run(run func(nodeIDs ...int64)) *MockRWChannelStore_SetLegacyChannelByNode_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]int64, len(args)-0)
		for i, a := range args[0:] {
			if a != nil {
				variadicArgs[i] = a.(int64)
			}
		}
		run(variadicArgs...)
	})
	return _c
}

func (_c *MockRWChannelStore_SetLegacyChannelByNode_Call) Return() *MockRWChannelStore_SetLegacyChannelByNode_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockRWChannelStore_SetLegacyChannelByNode_Call) RunAndReturn(run func(...int64)) *MockRWChannelStore_SetLegacyChannelByNode_Call {
	_c.Call.Return(run)
	return _c
}

// Update provides a mock function with given fields: op
func (_m *MockRWChannelStore) Update(op *ChannelOpSet) error {
	ret := _m.Called(op)

	var r0 error
	if rf, ok := ret.Get(0).(func(*ChannelOpSet) error); ok {
		r0 = rf(op)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockRWChannelStore_Update_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Update'
type MockRWChannelStore_Update_Call struct {
	*mock.Call
}

// Update is a helper method to define mock.On call
//   - op *ChannelOpSet
func (_e *MockRWChannelStore_Expecter) Update(op interface{}) *MockRWChannelStore_Update_Call {
	return &MockRWChannelStore_Update_Call{Call: _e.mock.On("Update", op)}
}

func (_c *MockRWChannelStore_Update_Call) Run(run func(op *ChannelOpSet)) *MockRWChannelStore_Update_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*ChannelOpSet))
	})
	return _c
}

func (_c *MockRWChannelStore_Update_Call) Return(_a0 error) *MockRWChannelStore_Update_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRWChannelStore_Update_Call) RunAndReturn(run func(*ChannelOpSet) error) *MockRWChannelStore_Update_Call {
	_c.Call.Return(run)
	return _c
}

// UpdateState provides a mock function with given fields: isSuccessful, channels
func (_m *MockRWChannelStore) UpdateState(isSuccessful bool, channels ...RWChannel) {
	_va := make([]interface{}, len(channels))
	for _i := range channels {
		_va[_i] = channels[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, isSuccessful)
	_ca = append(_ca, _va...)
	_m.Called(_ca...)
}

// MockRWChannelStore_UpdateState_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UpdateState'
type MockRWChannelStore_UpdateState_Call struct {
	*mock.Call
}

// UpdateState is a helper method to define mock.On call
//   - isSuccessful bool
//   - channels ...RWChannel
func (_e *MockRWChannelStore_Expecter) UpdateState(isSuccessful interface{}, channels ...interface{}) *MockRWChannelStore_UpdateState_Call {
	return &MockRWChannelStore_UpdateState_Call{Call: _e.mock.On("UpdateState",
		append([]interface{}{isSuccessful}, channels...)...)}
}

func (_c *MockRWChannelStore_UpdateState_Call) Run(run func(isSuccessful bool, channels ...RWChannel)) *MockRWChannelStore_UpdateState_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]RWChannel, len(args)-1)
		for i, a := range args[1:] {
			if a != nil {
				variadicArgs[i] = a.(RWChannel)
			}
		}
		run(args[0].(bool), variadicArgs...)
	})
	return _c
}

func (_c *MockRWChannelStore_UpdateState_Call) Return() *MockRWChannelStore_UpdateState_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockRWChannelStore_UpdateState_Call) RunAndReturn(run func(bool, ...RWChannel)) *MockRWChannelStore_UpdateState_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockRWChannelStore creates a new instance of MockRWChannelStore. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockRWChannelStore(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockRWChannelStore {
	mock := &MockRWChannelStore{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
