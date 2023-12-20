// Code generated by mockery v2.32.4. DO NOT EDIT.

package dependency

import (
	context "context"

	msgstream "github.com/milvus-io/milvus/pkg/mq/msgstream"
	mock "github.com/stretchr/testify/mock"

	paramtable "github.com/milvus-io/milvus/pkg/util/paramtable"

	storage "github.com/milvus-io/milvus/internal/storage"
)

// MockFactory is an autogenerated mock type for the Factory type
type MockFactory struct {
	mock.Mock
}

type MockFactory_Expecter struct {
	mock *mock.Mock
}

func (_m *MockFactory) EXPECT() *MockFactory_Expecter {
	return &MockFactory_Expecter{mock: &_m.Mock}
}

// Init provides a mock function with given fields: p
func (_m *MockFactory) Init(p *paramtable.ComponentParam) {
	_m.Called(p)
}

// MockFactory_Init_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Init'
type MockFactory_Init_Call struct {
	*mock.Call
}

// Init is a helper method to define mock.On call
//   - p *paramtable.ComponentParam
func (_e *MockFactory_Expecter) Init(p interface{}) *MockFactory_Init_Call {
	return &MockFactory_Init_Call{Call: _e.mock.On("Init", p)}
}

func (_c *MockFactory_Init_Call) Run(run func(p *paramtable.ComponentParam)) *MockFactory_Init_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*paramtable.ComponentParam))
	})
	return _c
}

func (_c *MockFactory_Init_Call) Return() *MockFactory_Init_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockFactory_Init_Call) RunAndReturn(run func(*paramtable.ComponentParam)) *MockFactory_Init_Call {
	_c.Call.Return(run)
	return _c
}

// NewMsgStream provides a mock function with given fields: ctx
func (_m *MockFactory) NewMsgStream(ctx context.Context) (msgstream.MsgStream, error) {
	ret := _m.Called(ctx)

	var r0 msgstream.MsgStream
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context) (msgstream.MsgStream, error)); ok {
		return rf(ctx)
	}
	if rf, ok := ret.Get(0).(func(context.Context) msgstream.MsgStream); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(msgstream.MsgStream)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockFactory_NewMsgStream_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'NewMsgStream'
type MockFactory_NewMsgStream_Call struct {
	*mock.Call
}

// NewMsgStream is a helper method to define mock.On call
//   - ctx context.Context
func (_e *MockFactory_Expecter) NewMsgStream(ctx interface{}) *MockFactory_NewMsgStream_Call {
	return &MockFactory_NewMsgStream_Call{Call: _e.mock.On("NewMsgStream", ctx)}
}

func (_c *MockFactory_NewMsgStream_Call) Run(run func(ctx context.Context)) *MockFactory_NewMsgStream_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockFactory_NewMsgStream_Call) Return(_a0 msgstream.MsgStream, _a1 error) *MockFactory_NewMsgStream_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockFactory_NewMsgStream_Call) RunAndReturn(run func(context.Context) (msgstream.MsgStream, error)) *MockFactory_NewMsgStream_Call {
	_c.Call.Return(run)
	return _c
}

// NewMsgStreamDisposer provides a mock function with given fields: ctx
func (_m *MockFactory) NewMsgStreamDisposer(ctx context.Context) func([]string, string) error {
	ret := _m.Called(ctx)

	var r0 func([]string, string) error
	if rf, ok := ret.Get(0).(func(context.Context) func([]string, string) error); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(func([]string, string) error)
		}
	}

	return r0
}

// MockFactory_NewMsgStreamDisposer_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'NewMsgStreamDisposer'
type MockFactory_NewMsgStreamDisposer_Call struct {
	*mock.Call
}

// NewMsgStreamDisposer is a helper method to define mock.On call
//   - ctx context.Context
func (_e *MockFactory_Expecter) NewMsgStreamDisposer(ctx interface{}) *MockFactory_NewMsgStreamDisposer_Call {
	return &MockFactory_NewMsgStreamDisposer_Call{Call: _e.mock.On("NewMsgStreamDisposer", ctx)}
}

func (_c *MockFactory_NewMsgStreamDisposer_Call) Run(run func(ctx context.Context)) *MockFactory_NewMsgStreamDisposer_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockFactory_NewMsgStreamDisposer_Call) Return(_a0 func([]string, string) error) *MockFactory_NewMsgStreamDisposer_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockFactory_NewMsgStreamDisposer_Call) RunAndReturn(run func(context.Context) func([]string, string) error) *MockFactory_NewMsgStreamDisposer_Call {
	_c.Call.Return(run)
	return _c
}

// NewPersistentStorageChunkManager provides a mock function with given fields: ctx
func (_m *MockFactory) NewPersistentStorageChunkManager(ctx context.Context) (storage.ChunkManager, error) {
	ret := _m.Called(ctx)

	var r0 storage.ChunkManager
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context) (storage.ChunkManager, error)); ok {
		return rf(ctx)
	}
	if rf, ok := ret.Get(0).(func(context.Context) storage.ChunkManager); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(storage.ChunkManager)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockFactory_NewPersistentStorageChunkManager_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'NewPersistentStorageChunkManager'
type MockFactory_NewPersistentStorageChunkManager_Call struct {
	*mock.Call
}

// NewPersistentStorageChunkManager is a helper method to define mock.On call
//   - ctx context.Context
func (_e *MockFactory_Expecter) NewPersistentStorageChunkManager(ctx interface{}) *MockFactory_NewPersistentStorageChunkManager_Call {
	return &MockFactory_NewPersistentStorageChunkManager_Call{Call: _e.mock.On("NewPersistentStorageChunkManager", ctx)}
}

func (_c *MockFactory_NewPersistentStorageChunkManager_Call) Run(run func(ctx context.Context)) *MockFactory_NewPersistentStorageChunkManager_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockFactory_NewPersistentStorageChunkManager_Call) Return(_a0 storage.ChunkManager, _a1 error) *MockFactory_NewPersistentStorageChunkManager_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockFactory_NewPersistentStorageChunkManager_Call) RunAndReturn(run func(context.Context) (storage.ChunkManager, error)) *MockFactory_NewPersistentStorageChunkManager_Call {
	_c.Call.Return(run)
	return _c
}

// NewTtMsgStream provides a mock function with given fields: ctx
func (_m *MockFactory) NewTtMsgStream(ctx context.Context) (msgstream.MsgStream, error) {
	ret := _m.Called(ctx)

	var r0 msgstream.MsgStream
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context) (msgstream.MsgStream, error)); ok {
		return rf(ctx)
	}
	if rf, ok := ret.Get(0).(func(context.Context) msgstream.MsgStream); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(msgstream.MsgStream)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockFactory_NewTtMsgStream_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'NewTtMsgStream'
type MockFactory_NewTtMsgStream_Call struct {
	*mock.Call
}

// NewTtMsgStream is a helper method to define mock.On call
//   - ctx context.Context
func (_e *MockFactory_Expecter) NewTtMsgStream(ctx interface{}) *MockFactory_NewTtMsgStream_Call {
	return &MockFactory_NewTtMsgStream_Call{Call: _e.mock.On("NewTtMsgStream", ctx)}
}

func (_c *MockFactory_NewTtMsgStream_Call) Run(run func(ctx context.Context)) *MockFactory_NewTtMsgStream_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockFactory_NewTtMsgStream_Call) Return(_a0 msgstream.MsgStream, _a1 error) *MockFactory_NewTtMsgStream_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockFactory_NewTtMsgStream_Call) RunAndReturn(run func(context.Context) (msgstream.MsgStream, error)) *MockFactory_NewTtMsgStream_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockFactory creates a new instance of MockFactory. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockFactory(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockFactory {
	mock := &MockFactory{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
