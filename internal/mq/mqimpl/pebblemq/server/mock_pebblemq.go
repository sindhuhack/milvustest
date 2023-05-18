// Code generated by mockery v2.16.0. DO NOT EDIT.

package server

import mock "github.com/stretchr/testify/mock"

// MockPebbleMQ is an autogenerated mock type for the RocksMQ type
type MockPebbleMQ struct {
	mock.Mock
}

type MockPebbleMQ_Expecter struct {
	mock *mock.Mock
}

func (_m *MockPebbleMQ) EXPECT() *MockPebbleMQ_Expecter {
	return &MockPebbleMQ_Expecter{mock: &_m.Mock}
}

// CheckTopicValid provides a mock function with given fields: topicName
func (_m *MockPebbleMQ) CheckTopicValid(topicName string) error {
	ret := _m.Called(topicName)

	var r0 error
	if rf, ok := ret.Get(0).(func(string) error); ok {
		r0 = rf(topicName)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_CheckTopicValid_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CheckTopicValid'
type MockPebbleMQ_CheckTopicValid_Call struct {
	*mock.Call
}

// CheckTopicValid is a helper method to define mock.On call
//   - topicName string
func (_e *MockPebbleMQ_Expecter) CheckTopicValid(topicName interface{}) *MockPebbleMQ_CheckTopicValid_Call {
	return &MockPebbleMQ_CheckTopicValid_Call{Call: _e.mock.On("CheckTopicValid", topicName)}
}

func (_c *MockPebbleMQ_CheckTopicValid_Call) Run(run func(topicName string)) *MockPebbleMQ_CheckTopicValid_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_CheckTopicValid_Call) Return(_a0 error) *MockPebbleMQ_CheckTopicValid_Call {
	_c.Call.Return(_a0)
	return _c
}

// Close provides a mock function with given fields:
func (_m *MockPebbleMQ) Close() {
	_m.Called()
}

// MockPebbleMQ_Close_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Close'
type MockPebbleMQ_Close_Call struct {
	*mock.Call
}

// Close is a helper method to define mock.On call
func (_e *MockPebbleMQ_Expecter) Close() *MockPebbleMQ_Close_Call {
	return &MockPebbleMQ_Close_Call{Call: _e.mock.On("Close")}
}

func (_c *MockPebbleMQ_Close_Call) Run(run func()) *MockPebbleMQ_Close_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockPebbleMQ_Close_Call) Return() *MockPebbleMQ_Close_Call {
	_c.Call.Return()
	return _c
}

// Consume provides a mock function with given fields: topicName, groupName, n
func (_m *MockPebbleMQ) Consume(topicName string, groupName string, n int) ([]ConsumerMessage, error) {
	ret := _m.Called(topicName, groupName, n)

	var r0 []ConsumerMessage
	if rf, ok := ret.Get(0).(func(string, string, int) []ConsumerMessage); ok {
		r0 = rf(topicName, groupName, n)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]ConsumerMessage)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(string, string, int) error); ok {
		r1 = rf(topicName, groupName, n)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockPebbleMQ_Consume_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Consume'
type MockPebbleMQ_Consume_Call struct {
	*mock.Call
}

// Consume is a helper method to define mock.On call
//   - topicName string
//   - groupName string
//   - n int
func (_e *MockPebbleMQ_Expecter) Consume(topicName interface{}, groupName interface{}, n interface{}) *MockPebbleMQ_Consume_Call {
	return &MockPebbleMQ_Consume_Call{Call: _e.mock.On("Consume", topicName, groupName, n)}
}

func (_c *MockPebbleMQ_Consume_Call) Run(run func(topicName string, groupName string, n int)) *MockPebbleMQ_Consume_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string), args[2].(int))
	})
	return _c
}

func (_c *MockPebbleMQ_Consume_Call) Return(_a0 []ConsumerMessage, _a1 error) *MockPebbleMQ_Consume_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// CreateConsumerGroup provides a mock function with given fields: topicName, groupName
func (_m *MockPebbleMQ) CreateConsumerGroup(topicName string, groupName string) error {
	ret := _m.Called(topicName, groupName)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, string) error); ok {
		r0 = rf(topicName, groupName)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_CreateConsumerGroup_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CreateConsumerGroup'
type MockPebbleMQ_CreateConsumerGroup_Call struct {
	*mock.Call
}

// CreateConsumerGroup is a helper method to define mock.On call
//   - topicName string
//   - groupName string
func (_e *MockPebbleMQ_Expecter) CreateConsumerGroup(topicName interface{}, groupName interface{}) *MockPebbleMQ_CreateConsumerGroup_Call {
	return &MockPebbleMQ_CreateConsumerGroup_Call{Call: _e.mock.On("CreateConsumerGroup", topicName, groupName)}
}

func (_c *MockPebbleMQ_CreateConsumerGroup_Call) Run(run func(topicName string, groupName string)) *MockPebbleMQ_CreateConsumerGroup_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_CreateConsumerGroup_Call) Return(_a0 error) *MockPebbleMQ_CreateConsumerGroup_Call {
	_c.Call.Return(_a0)
	return _c
}

// CreateTopic provides a mock function with given fields: topicName
func (_m *MockPebbleMQ) CreateTopic(topicName string) error {
	ret := _m.Called(topicName)

	var r0 error
	if rf, ok := ret.Get(0).(func(string) error); ok {
		r0 = rf(topicName)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_CreateTopic_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CreateTopic'
type MockPebbleMQ_CreateTopic_Call struct {
	*mock.Call
}

// CreateTopic is a helper method to define mock.On call
//   - topicName string
func (_e *MockPebbleMQ_Expecter) CreateTopic(topicName interface{}) *MockPebbleMQ_CreateTopic_Call {
	return &MockPebbleMQ_CreateTopic_Call{Call: _e.mock.On("CreateTopic", topicName)}
}

func (_c *MockPebbleMQ_CreateTopic_Call) Run(run func(topicName string)) *MockPebbleMQ_CreateTopic_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_CreateTopic_Call) Return(_a0 error) *MockPebbleMQ_CreateTopic_Call {
	_c.Call.Return(_a0)
	return _c
}

// DestroyConsumerGroup provides a mock function with given fields: topicName, groupName
func (_m *MockPebbleMQ) DestroyConsumerGroup(topicName string, groupName string) error {
	ret := _m.Called(topicName, groupName)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, string) error); ok {
		r0 = rf(topicName, groupName)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_DestroyConsumerGroup_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'DestroyConsumerGroup'
type MockPebbleMQ_DestroyConsumerGroup_Call struct {
	*mock.Call
}

// DestroyConsumerGroup is a helper method to define mock.On call
//   - topicName string
//   - groupName string
func (_e *MockPebbleMQ_Expecter) DestroyConsumerGroup(topicName interface{}, groupName interface{}) *MockPebbleMQ_DestroyConsumerGroup_Call {
	return &MockPebbleMQ_DestroyConsumerGroup_Call{Call: _e.mock.On("DestroyConsumerGroup", topicName, groupName)}
}

func (_c *MockPebbleMQ_DestroyConsumerGroup_Call) Run(run func(topicName string, groupName string)) *MockPebbleMQ_DestroyConsumerGroup_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_DestroyConsumerGroup_Call) Return(_a0 error) *MockPebbleMQ_DestroyConsumerGroup_Call {
	_c.Call.Return(_a0)
	return _c
}

// DestroyTopic provides a mock function with given fields: topicName
func (_m *MockPebbleMQ) DestroyTopic(topicName string) error {
	ret := _m.Called(topicName)

	var r0 error
	if rf, ok := ret.Get(0).(func(string) error); ok {
		r0 = rf(topicName)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_DestroyTopic_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'DestroyTopic'
type MockPebbleMQ_DestroyTopic_Call struct {
	*mock.Call
}

// DestroyTopic is a helper method to define mock.On call
//   - topicName string
func (_e *MockPebbleMQ_Expecter) DestroyTopic(topicName interface{}) *MockPebbleMQ_DestroyTopic_Call {
	return &MockPebbleMQ_DestroyTopic_Call{Call: _e.mock.On("DestroyTopic", topicName)}
}

func (_c *MockPebbleMQ_DestroyTopic_Call) Run(run func(topicName string)) *MockPebbleMQ_DestroyTopic_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_DestroyTopic_Call) Return(_a0 error) *MockPebbleMQ_DestroyTopic_Call {
	_c.Call.Return(_a0)
	return _c
}

// ExistConsumerGroup provides a mock function with given fields: topicName, groupName
func (_m *MockPebbleMQ) ExistConsumerGroup(topicName string, groupName string) (bool, *Consumer, error) {
	ret := _m.Called(topicName, groupName)

	var r0 bool
	if rf, ok := ret.Get(0).(func(string, string) bool); ok {
		r0 = rf(topicName, groupName)
	} else {
		r0 = ret.Get(0).(bool)
	}

	var r1 *Consumer
	if rf, ok := ret.Get(1).(func(string, string) *Consumer); ok {
		r1 = rf(topicName, groupName)
	} else {
		if ret.Get(1) != nil {
			r1 = ret.Get(1).(*Consumer)
		}
	}

	var r2 error
	if rf, ok := ret.Get(2).(func(string, string) error); ok {
		r2 = rf(topicName, groupName)
	} else {
		r2 = ret.Error(2)
	}

	return r0, r1, r2
}

// MockPebbleMQ_ExistConsumerGroup_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ExistConsumerGroup'
type MockPebbleMQ_ExistConsumerGroup_Call struct {
	*mock.Call
}

// ExistConsumerGroup is a helper method to define mock.On call
//   - topicName string
//   - groupName string
func (_e *MockPebbleMQ_Expecter) ExistConsumerGroup(topicName interface{}, groupName interface{}) *MockPebbleMQ_ExistConsumerGroup_Call {
	return &MockPebbleMQ_ExistConsumerGroup_Call{Call: _e.mock.On("ExistConsumerGroup", topicName, groupName)}
}

func (_c *MockPebbleMQ_ExistConsumerGroup_Call) Run(run func(topicName string, groupName string)) *MockPebbleMQ_ExistConsumerGroup_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_ExistConsumerGroup_Call) Return(_a0 bool, _a1 *Consumer, _a2 error) *MockPebbleMQ_ExistConsumerGroup_Call {
	_c.Call.Return(_a0, _a1, _a2)
	return _c
}

// GetLatestMsg provides a mock function with given fields: topicName
func (_m *MockPebbleMQ) GetLatestMsg(topicName string) (int64, error) {
	ret := _m.Called(topicName)

	var r0 int64
	if rf, ok := ret.Get(0).(func(string) int64); ok {
		r0 = rf(topicName)
	} else {
		r0 = ret.Get(0).(int64)
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(string) error); ok {
		r1 = rf(topicName)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockPebbleMQ_GetLatestMsg_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetLatestMsg'
type MockPebbleMQ_GetLatestMsg_Call struct {
	*mock.Call
}

// GetLatestMsg is a helper method to define mock.On call
//   - topicName string
func (_e *MockPebbleMQ_Expecter) GetLatestMsg(topicName interface{}) *MockPebbleMQ_GetLatestMsg_Call {
	return &MockPebbleMQ_GetLatestMsg_Call{Call: _e.mock.On("GetLatestMsg", topicName)}
}

func (_c *MockPebbleMQ_GetLatestMsg_Call) Run(run func(topicName string)) *MockPebbleMQ_GetLatestMsg_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_GetLatestMsg_Call) Return(_a0 int64, _a1 error) *MockPebbleMQ_GetLatestMsg_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Notify provides a mock function with given fields: topicName, groupName
func (_m *MockPebbleMQ) Notify(topicName string, groupName string) {
	_m.Called(topicName, groupName)
}

// MockPebbleMQ_Notify_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Notify'
type MockPebbleMQ_Notify_Call struct {
	*mock.Call
}

// Notify is a helper method to define mock.On call
//   - topicName string
//   - groupName string
func (_e *MockPebbleMQ_Expecter) Notify(topicName interface{}, groupName interface{}) *MockPebbleMQ_Notify_Call {
	return &MockPebbleMQ_Notify_Call{Call: _e.mock.On("Notify", topicName, groupName)}
}

func (_c *MockPebbleMQ_Notify_Call) Run(run func(topicName string, groupName string)) *MockPebbleMQ_Notify_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_Notify_Call) Return() *MockPebbleMQ_Notify_Call {
	_c.Call.Return()
	return _c
}

// Produce provides a mock function with given fields: topicName, messages
func (_m *MockPebbleMQ) Produce(topicName string, messages []ProducerMessage) ([]int64, error) {
	ret := _m.Called(topicName, messages)

	var r0 []int64
	if rf, ok := ret.Get(0).(func(string, []ProducerMessage) []int64); ok {
		r0 = rf(topicName, messages)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]int64)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(string, []ProducerMessage) error); ok {
		r1 = rf(topicName, messages)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockPebbleMQ_Produce_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Produce'
type MockPebbleMQ_Produce_Call struct {
	*mock.Call
}

// Produce is a helper method to define mock.On call
//   - topicName string
//   - messages []ProducerMessage
func (_e *MockPebbleMQ_Expecter) Produce(topicName interface{}, messages interface{}) *MockPebbleMQ_Produce_Call {
	return &MockPebbleMQ_Produce_Call{Call: _e.mock.On("Produce", topicName, messages)}
}

func (_c *MockPebbleMQ_Produce_Call) Run(run func(topicName string, messages []ProducerMessage)) *MockPebbleMQ_Produce_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].([]ProducerMessage))
	})
	return _c
}

func (_c *MockPebbleMQ_Produce_Call) Return(_a0 []int64, _a1 error) *MockPebbleMQ_Produce_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// RegisterConsumer provides a mock function with given fields: consumer
func (_m *MockPebbleMQ) RegisterConsumer(consumer *Consumer) error {
	ret := _m.Called(consumer)

	var r0 error
	if rf, ok := ret.Get(0).(func(*Consumer) error); ok {
		r0 = rf(consumer)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_RegisterConsumer_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RegisterConsumer'
type MockPebbleMQ_RegisterConsumer_Call struct {
	*mock.Call
}

// RegisterConsumer is a helper method to define mock.On call
//   - consumer *Consumer
func (_e *MockPebbleMQ_Expecter) RegisterConsumer(consumer interface{}) *MockPebbleMQ_RegisterConsumer_Call {
	return &MockPebbleMQ_RegisterConsumer_Call{Call: _e.mock.On("RegisterConsumer", consumer)}
}

func (_c *MockPebbleMQ_RegisterConsumer_Call) Run(run func(consumer *Consumer)) *MockPebbleMQ_RegisterConsumer_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*Consumer))
	})
	return _c
}

func (_c *MockPebbleMQ_RegisterConsumer_Call) Return(_a0 error) *MockPebbleMQ_RegisterConsumer_Call {
	_c.Call.Return(_a0)
	return _c
}

// Seek provides a mock function with given fields: topicName, groupName, msgID
func (_m *MockPebbleMQ) Seek(topicName string, groupName string, msgID int64) error {
	ret := _m.Called(topicName, groupName, msgID)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, string, int64) error); ok {
		r0 = rf(topicName, groupName, msgID)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_Seek_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Seek'
type MockPebbleMQ_Seek_Call struct {
	*mock.Call
}

// Seek is a helper method to define mock.On call
//   - topicName string
//   - groupName string
//   - msgID int64
func (_e *MockPebbleMQ_Expecter) Seek(topicName interface{}, groupName interface{}, msgID interface{}) *MockPebbleMQ_Seek_Call {
	return &MockPebbleMQ_Seek_Call{Call: _e.mock.On("Seek", topicName, groupName, msgID)}
}

func (_c *MockPebbleMQ_Seek_Call) Run(run func(topicName string, groupName string, msgID int64)) *MockPebbleMQ_Seek_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string), args[2].(int64))
	})
	return _c
}

func (_c *MockPebbleMQ_Seek_Call) Return(_a0 error) *MockPebbleMQ_Seek_Call {
	_c.Call.Return(_a0)
	return _c
}

// SeekToLatest provides a mock function with given fields: topicName, groupName
func (_m *MockPebbleMQ) SeekToLatest(topicName string, groupName string) error {
	ret := _m.Called(topicName, groupName)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, string) error); ok {
		r0 = rf(topicName, groupName)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockPebbleMQ_SeekToLatest_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SeekToLatest'
type MockPebbleMQ_SeekToLatest_Call struct {
	*mock.Call
}

// SeekToLatest is a helper method to define mock.On call
//   - topicName string
//   - groupName string
func (_e *MockPebbleMQ_Expecter) SeekToLatest(topicName interface{}, groupName interface{}) *MockPebbleMQ_SeekToLatest_Call {
	return &MockPebbleMQ_SeekToLatest_Call{Call: _e.mock.On("SeekToLatest", topicName, groupName)}
}

func (_c *MockPebbleMQ_SeekToLatest_Call) Run(run func(topicName string, groupName string)) *MockPebbleMQ_SeekToLatest_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(string))
	})
	return _c
}

func (_c *MockPebbleMQ_SeekToLatest_Call) Return(_a0 error) *MockPebbleMQ_SeekToLatest_Call {
	_c.Call.Return(_a0)
	return _c
}

type mockConstructorTestingTNewMockPebbleMQ interface {
	mock.TestingT
	Cleanup(func())
}

// NewMockPebbleMQ creates a new instance of MockPebbleMQ. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
func NewMockPebbleMQ(t mockConstructorTestingTNewMockPebbleMQ) *MockPebbleMQ {
	mock := &MockPebbleMQ{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
