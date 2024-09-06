// Code generated by mockery v2.32.4. DO NOT EDIT.

package mock_streaming

import (
	context "context"

	message "github.com/milvus-io/milvus/pkg/streaming/util/message"
	mock "github.com/stretchr/testify/mock"

	streaming "github.com/milvus-io/milvus/internal/distributed/streaming"

	types "github.com/milvus-io/milvus/pkg/streaming/util/types"
)

// MockWALAccesser is an autogenerated mock type for the WALAccesser type
type MockWALAccesser struct {
	mock.Mock
}

type MockWALAccesser_Expecter struct {
	mock *mock.Mock
}

func (_m *MockWALAccesser) EXPECT() *MockWALAccesser_Expecter {
	return &MockWALAccesser_Expecter{mock: &_m.Mock}
}

// AppendMessages provides a mock function with given fields: ctx, msgs
func (_m *MockWALAccesser) AppendMessages(ctx context.Context, msgs ...message.MutableMessage) streaming.AppendResponses {
	_va := make([]interface{}, len(msgs))
	for _i := range msgs {
		_va[_i] = msgs[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 streaming.AppendResponses
	if rf, ok := ret.Get(0).(func(context.Context, ...message.MutableMessage) streaming.AppendResponses); ok {
		r0 = rf(ctx, msgs...)
	} else {
		r0 = ret.Get(0).(streaming.AppendResponses)
	}

	return r0
}

// MockWALAccesser_AppendMessages_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AppendMessages'
type MockWALAccesser_AppendMessages_Call struct {
	*mock.Call
}

// AppendMessages is a helper method to define mock.On call
//   - ctx context.Context
//   - msgs ...message.MutableMessage
func (_e *MockWALAccesser_Expecter) AppendMessages(ctx interface{}, msgs ...interface{}) *MockWALAccesser_AppendMessages_Call {
	return &MockWALAccesser_AppendMessages_Call{Call: _e.mock.On("AppendMessages",
		append([]interface{}{ctx}, msgs...)...)}
}

func (_c *MockWALAccesser_AppendMessages_Call) Run(run func(ctx context.Context, msgs ...message.MutableMessage)) *MockWALAccesser_AppendMessages_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]message.MutableMessage, len(args)-1)
		for i, a := range args[1:] {
			if a != nil {
				variadicArgs[i] = a.(message.MutableMessage)
			}
		}
		run(args[0].(context.Context), variadicArgs...)
	})
	return _c
}

func (_c *MockWALAccesser_AppendMessages_Call) Return(_a0 streaming.AppendResponses) *MockWALAccesser_AppendMessages_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWALAccesser_AppendMessages_Call) RunAndReturn(run func(context.Context, ...message.MutableMessage) streaming.AppendResponses) *MockWALAccesser_AppendMessages_Call {
	_c.Call.Return(run)
	return _c
}

// AppendMessagesWithOption provides a mock function with given fields: ctx, opts, msgs
func (_m *MockWALAccesser) AppendMessagesWithOption(ctx context.Context, opts streaming.AppendOption, msgs ...message.MutableMessage) streaming.AppendResponses {
	_va := make([]interface{}, len(msgs))
	for _i := range msgs {
		_va[_i] = msgs[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, opts)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 streaming.AppendResponses
	if rf, ok := ret.Get(0).(func(context.Context, streaming.AppendOption, ...message.MutableMessage) streaming.AppendResponses); ok {
		r0 = rf(ctx, opts, msgs...)
	} else {
		r0 = ret.Get(0).(streaming.AppendResponses)
	}

	return r0
}

// MockWALAccesser_AppendMessagesWithOption_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AppendMessagesWithOption'
type MockWALAccesser_AppendMessagesWithOption_Call struct {
	*mock.Call
}

// AppendMessagesWithOption is a helper method to define mock.On call
//   - ctx context.Context
//   - opts streaming.AppendOption
//   - msgs ...message.MutableMessage
func (_e *MockWALAccesser_Expecter) AppendMessagesWithOption(ctx interface{}, opts interface{}, msgs ...interface{}) *MockWALAccesser_AppendMessagesWithOption_Call {
	return &MockWALAccesser_AppendMessagesWithOption_Call{Call: _e.mock.On("AppendMessagesWithOption",
		append([]interface{}{ctx, opts}, msgs...)...)}
}

func (_c *MockWALAccesser_AppendMessagesWithOption_Call) Run(run func(ctx context.Context, opts streaming.AppendOption, msgs ...message.MutableMessage)) *MockWALAccesser_AppendMessagesWithOption_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]message.MutableMessage, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(message.MutableMessage)
			}
		}
		run(args[0].(context.Context), args[1].(streaming.AppendOption), variadicArgs...)
	})
	return _c
}

func (_c *MockWALAccesser_AppendMessagesWithOption_Call) Return(_a0 streaming.AppendResponses) *MockWALAccesser_AppendMessagesWithOption_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWALAccesser_AppendMessagesWithOption_Call) RunAndReturn(run func(context.Context, streaming.AppendOption, ...message.MutableMessage) streaming.AppendResponses) *MockWALAccesser_AppendMessagesWithOption_Call {
	_c.Call.Return(run)
	return _c
}

// RawAppend provides a mock function with given fields: ctx, msgs, opts
func (_m *MockWALAccesser) RawAppend(ctx context.Context, msgs message.MutableMessage, opts ...streaming.AppendOption) (*types.AppendResult, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, msgs)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *types.AppendResult
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, message.MutableMessage, ...streaming.AppendOption) (*types.AppendResult, error)); ok {
		return rf(ctx, msgs, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, message.MutableMessage, ...streaming.AppendOption) *types.AppendResult); ok {
		r0 = rf(ctx, msgs, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*types.AppendResult)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, message.MutableMessage, ...streaming.AppendOption) error); ok {
		r1 = rf(ctx, msgs, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockWALAccesser_RawAppend_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RawAppend'
type MockWALAccesser_RawAppend_Call struct {
	*mock.Call
}

// RawAppend is a helper method to define mock.On call
//   - ctx context.Context
//   - msgs message.MutableMessage
//   - opts ...streaming.AppendOption
func (_e *MockWALAccesser_Expecter) RawAppend(ctx interface{}, msgs interface{}, opts ...interface{}) *MockWALAccesser_RawAppend_Call {
	return &MockWALAccesser_RawAppend_Call{Call: _e.mock.On("RawAppend",
		append([]interface{}{ctx, msgs}, opts...)...)}
}

func (_c *MockWALAccesser_RawAppend_Call) Run(run func(ctx context.Context, msgs message.MutableMessage, opts ...streaming.AppendOption)) *MockWALAccesser_RawAppend_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]streaming.AppendOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(streaming.AppendOption)
			}
		}
		run(args[0].(context.Context), args[1].(message.MutableMessage), variadicArgs...)
	})
	return _c
}

func (_c *MockWALAccesser_RawAppend_Call) Return(_a0 *types.AppendResult, _a1 error) *MockWALAccesser_RawAppend_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockWALAccesser_RawAppend_Call) RunAndReturn(run func(context.Context, message.MutableMessage, ...streaming.AppendOption) (*types.AppendResult, error)) *MockWALAccesser_RawAppend_Call {
	_c.Call.Return(run)
	return _c
}

// Read provides a mock function with given fields: ctx, opts
func (_m *MockWALAccesser) Read(ctx context.Context, opts streaming.ReadOption) streaming.Scanner {
	ret := _m.Called(ctx, opts)

	var r0 streaming.Scanner
	if rf, ok := ret.Get(0).(func(context.Context, streaming.ReadOption) streaming.Scanner); ok {
		r0 = rf(ctx, opts)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(streaming.Scanner)
		}
	}

	return r0
}

// MockWALAccesser_Read_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Read'
type MockWALAccesser_Read_Call struct {
	*mock.Call
}

// Read is a helper method to define mock.On call
//   - ctx context.Context
//   - opts streaming.ReadOption
func (_e *MockWALAccesser_Expecter) Read(ctx interface{}, opts interface{}) *MockWALAccesser_Read_Call {
	return &MockWALAccesser_Read_Call{Call: _e.mock.On("Read", ctx, opts)}
}

func (_c *MockWALAccesser_Read_Call) Run(run func(ctx context.Context, opts streaming.ReadOption)) *MockWALAccesser_Read_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(streaming.ReadOption))
	})
	return _c
}

func (_c *MockWALAccesser_Read_Call) Return(_a0 streaming.Scanner) *MockWALAccesser_Read_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWALAccesser_Read_Call) RunAndReturn(run func(context.Context, streaming.ReadOption) streaming.Scanner) *MockWALAccesser_Read_Call {
	_c.Call.Return(run)
	return _c
}

// Txn provides a mock function with given fields: ctx, opts
func (_m *MockWALAccesser) Txn(ctx context.Context, opts streaming.TxnOption) (streaming.Txn, error) {
	ret := _m.Called(ctx, opts)

	var r0 streaming.Txn
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, streaming.TxnOption) (streaming.Txn, error)); ok {
		return rf(ctx, opts)
	}
	if rf, ok := ret.Get(0).(func(context.Context, streaming.TxnOption) streaming.Txn); ok {
		r0 = rf(ctx, opts)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(streaming.Txn)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, streaming.TxnOption) error); ok {
		r1 = rf(ctx, opts)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockWALAccesser_Txn_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Txn'
type MockWALAccesser_Txn_Call struct {
	*mock.Call
}

// Txn is a helper method to define mock.On call
//   - ctx context.Context
//   - opts streaming.TxnOption
func (_e *MockWALAccesser_Expecter) Txn(ctx interface{}, opts interface{}) *MockWALAccesser_Txn_Call {
	return &MockWALAccesser_Txn_Call{Call: _e.mock.On("Txn", ctx, opts)}
}

func (_c *MockWALAccesser_Txn_Call) Run(run func(ctx context.Context, opts streaming.TxnOption)) *MockWALAccesser_Txn_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(streaming.TxnOption))
	})
	return _c
}

func (_c *MockWALAccesser_Txn_Call) Return(_a0 streaming.Txn, _a1 error) *MockWALAccesser_Txn_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockWALAccesser_Txn_Call) RunAndReturn(run func(context.Context, streaming.TxnOption) (streaming.Txn, error)) *MockWALAccesser_Txn_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockWALAccesser creates a new instance of MockWALAccesser. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockWALAccesser(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockWALAccesser {
	mock := &MockWALAccesser{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
