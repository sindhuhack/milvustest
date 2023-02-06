// Code generated by mockery v2.16.0. DO NOT EDIT.

package cluster

import (
	context "context"

	commonpb "github.com/milvus-io/milvus-proto/go-api/commonpb"

	internalpb "github.com/milvus-io/milvus/internal/proto/internalpb"

	milvuspb "github.com/milvus-io/milvus-proto/go-api/milvuspb"

	mock "github.com/stretchr/testify/mock"

	querypb "github.com/milvus-io/milvus/internal/proto/querypb"
)

// MockQueryNode is an autogenerated mock type for the QueryNode type
type MockQueryNode struct {
	mock.Mock
}

type MockQueryNode_Expecter struct {
	mock *mock.Mock
}

func (_m *MockQueryNode) EXPECT() *MockQueryNode_Expecter {
	return &MockQueryNode_Expecter{mock: &_m.Mock}
}

// Delete provides a mock function with given fields: _a0, _a1
func (_m *MockQueryNode) Delete(_a0 context.Context, _a1 *querypb.DeleteRequest) (*commonpb.Status, error) {
	ret := _m.Called(_a0, _a1)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.DeleteRequest) *commonpb.Status); ok {
		r0 = rf(_a0, _a1)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.DeleteRequest) error); ok {
		r1 = rf(_a0, _a1)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_Delete_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Delete'
type MockQueryNode_Delete_Call struct {
	*mock.Call
}

// Delete is a helper method to define mock.On call
//  - _a0 context.Context
//  - _a1 *querypb.DeleteRequest
func (_e *MockQueryNode_Expecter) Delete(_a0 interface{}, _a1 interface{}) *MockQueryNode_Delete_Call {
	return &MockQueryNode_Delete_Call{Call: _e.mock.On("Delete", _a0, _a1)}
}

func (_c *MockQueryNode_Delete_Call) Run(run func(_a0 context.Context, _a1 *querypb.DeleteRequest)) *MockQueryNode_Delete_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.DeleteRequest))
	})
	return _c
}

func (_c *MockQueryNode_Delete_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_Delete_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetComponentStates provides a mock function with given fields: ctx
func (_m *MockQueryNode) GetComponentStates(ctx context.Context) (*milvuspb.ComponentStates, error) {
	ret := _m.Called(ctx)

	var r0 *milvuspb.ComponentStates
	if rf, ok := ret.Get(0).(func(context.Context) *milvuspb.ComponentStates); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.ComponentStates)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetComponentStates_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetComponentStates'
type MockQueryNode_GetComponentStates_Call struct {
	*mock.Call
}

// GetComponentStates is a helper method to define mock.On call
//  - ctx context.Context
func (_e *MockQueryNode_Expecter) GetComponentStates(ctx interface{}) *MockQueryNode_GetComponentStates_Call {
	return &MockQueryNode_GetComponentStates_Call{Call: _e.mock.On("GetComponentStates", ctx)}
}

func (_c *MockQueryNode_GetComponentStates_Call) Run(run func(ctx context.Context)) *MockQueryNode_GetComponentStates_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockQueryNode_GetComponentStates_Call) Return(_a0 *milvuspb.ComponentStates, _a1 error) *MockQueryNode_GetComponentStates_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetDataDistribution provides a mock function with given fields: _a0, _a1
func (_m *MockQueryNode) GetDataDistribution(_a0 context.Context, _a1 *querypb.GetDataDistributionRequest) (*querypb.GetDataDistributionResponse, error) {
	ret := _m.Called(_a0, _a1)

	var r0 *querypb.GetDataDistributionResponse
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.GetDataDistributionRequest) *querypb.GetDataDistributionResponse); ok {
		r0 = rf(_a0, _a1)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*querypb.GetDataDistributionResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.GetDataDistributionRequest) error); ok {
		r1 = rf(_a0, _a1)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetDataDistribution_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetDataDistribution'
type MockQueryNode_GetDataDistribution_Call struct {
	*mock.Call
}

// GetDataDistribution is a helper method to define mock.On call
//  - _a0 context.Context
//  - _a1 *querypb.GetDataDistributionRequest
func (_e *MockQueryNode_Expecter) GetDataDistribution(_a0 interface{}, _a1 interface{}) *MockQueryNode_GetDataDistribution_Call {
	return &MockQueryNode_GetDataDistribution_Call{Call: _e.mock.On("GetDataDistribution", _a0, _a1)}
}

func (_c *MockQueryNode_GetDataDistribution_Call) Run(run func(_a0 context.Context, _a1 *querypb.GetDataDistributionRequest)) *MockQueryNode_GetDataDistribution_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.GetDataDistributionRequest))
	})
	return _c
}

func (_c *MockQueryNode_GetDataDistribution_Call) Return(_a0 *querypb.GetDataDistributionResponse, _a1 error) *MockQueryNode_GetDataDistribution_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetMetrics provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *milvuspb.GetMetricsResponse
	if rf, ok := ret.Get(0).(func(context.Context, *milvuspb.GetMetricsRequest) *milvuspb.GetMetricsResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.GetMetricsResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *milvuspb.GetMetricsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetMetrics_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetMetrics'
type MockQueryNode_GetMetrics_Call struct {
	*mock.Call
}

// GetMetrics is a helper method to define mock.On call
//  - ctx context.Context
//  - req *milvuspb.GetMetricsRequest
func (_e *MockQueryNode_Expecter) GetMetrics(ctx interface{}, req interface{}) *MockQueryNode_GetMetrics_Call {
	return &MockQueryNode_GetMetrics_Call{Call: _e.mock.On("GetMetrics", ctx, req)}
}

func (_c *MockQueryNode_GetMetrics_Call) Run(run func(ctx context.Context, req *milvuspb.GetMetricsRequest)) *MockQueryNode_GetMetrics_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*milvuspb.GetMetricsRequest))
	})
	return _c
}

func (_c *MockQueryNode_GetMetrics_Call) Return(_a0 *milvuspb.GetMetricsResponse, _a1 error) *MockQueryNode_GetMetrics_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetSegmentInfo provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) GetSegmentInfo(ctx context.Context, req *querypb.GetSegmentInfoRequest) (*querypb.GetSegmentInfoResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *querypb.GetSegmentInfoResponse
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.GetSegmentInfoRequest) *querypb.GetSegmentInfoResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*querypb.GetSegmentInfoResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.GetSegmentInfoRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetSegmentInfo_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetSegmentInfo'
type MockQueryNode_GetSegmentInfo_Call struct {
	*mock.Call
}

// GetSegmentInfo is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.GetSegmentInfoRequest
func (_e *MockQueryNode_Expecter) GetSegmentInfo(ctx interface{}, req interface{}) *MockQueryNode_GetSegmentInfo_Call {
	return &MockQueryNode_GetSegmentInfo_Call{Call: _e.mock.On("GetSegmentInfo", ctx, req)}
}

func (_c *MockQueryNode_GetSegmentInfo_Call) Run(run func(ctx context.Context, req *querypb.GetSegmentInfoRequest)) *MockQueryNode_GetSegmentInfo_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.GetSegmentInfoRequest))
	})
	return _c
}

func (_c *MockQueryNode_GetSegmentInfo_Call) Return(_a0 *querypb.GetSegmentInfoResponse, _a1 error) *MockQueryNode_GetSegmentInfo_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetStatistics provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) GetStatistics(ctx context.Context, req *querypb.GetStatisticsRequest) (*internalpb.GetStatisticsResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *internalpb.GetStatisticsResponse
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.GetStatisticsRequest) *internalpb.GetStatisticsResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*internalpb.GetStatisticsResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.GetStatisticsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetStatistics_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetStatistics'
type MockQueryNode_GetStatistics_Call struct {
	*mock.Call
}

// GetStatistics is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.GetStatisticsRequest
func (_e *MockQueryNode_Expecter) GetStatistics(ctx interface{}, req interface{}) *MockQueryNode_GetStatistics_Call {
	return &MockQueryNode_GetStatistics_Call{Call: _e.mock.On("GetStatistics", ctx, req)}
}

func (_c *MockQueryNode_GetStatistics_Call) Run(run func(ctx context.Context, req *querypb.GetStatisticsRequest)) *MockQueryNode_GetStatistics_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.GetStatisticsRequest))
	})
	return _c
}

func (_c *MockQueryNode_GetStatistics_Call) Return(_a0 *internalpb.GetStatisticsResponse, _a1 error) *MockQueryNode_GetStatistics_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetStatisticsChannel provides a mock function with given fields: ctx
func (_m *MockQueryNode) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret := _m.Called(ctx)

	var r0 *milvuspb.StringResponse
	if rf, ok := ret.Get(0).(func(context.Context) *milvuspb.StringResponse); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.StringResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetStatisticsChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetStatisticsChannel'
type MockQueryNode_GetStatisticsChannel_Call struct {
	*mock.Call
}

// GetStatisticsChannel is a helper method to define mock.On call
//  - ctx context.Context
func (_e *MockQueryNode_Expecter) GetStatisticsChannel(ctx interface{}) *MockQueryNode_GetStatisticsChannel_Call {
	return &MockQueryNode_GetStatisticsChannel_Call{Call: _e.mock.On("GetStatisticsChannel", ctx)}
}

func (_c *MockQueryNode_GetStatisticsChannel_Call) Run(run func(ctx context.Context)) *MockQueryNode_GetStatisticsChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockQueryNode_GetStatisticsChannel_Call) Return(_a0 *milvuspb.StringResponse, _a1 error) *MockQueryNode_GetStatisticsChannel_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetTimeTickChannel provides a mock function with given fields: ctx
func (_m *MockQueryNode) GetTimeTickChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret := _m.Called(ctx)

	var r0 *milvuspb.StringResponse
	if rf, ok := ret.Get(0).(func(context.Context) *milvuspb.StringResponse); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.StringResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_GetTimeTickChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetTimeTickChannel'
type MockQueryNode_GetTimeTickChannel_Call struct {
	*mock.Call
}

// GetTimeTickChannel is a helper method to define mock.On call
//  - ctx context.Context
func (_e *MockQueryNode_Expecter) GetTimeTickChannel(ctx interface{}) *MockQueryNode_GetTimeTickChannel_Call {
	return &MockQueryNode_GetTimeTickChannel_Call{Call: _e.mock.On("GetTimeTickChannel", ctx)}
}

func (_c *MockQueryNode_GetTimeTickChannel_Call) Run(run func(ctx context.Context)) *MockQueryNode_GetTimeTickChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockQueryNode_GetTimeTickChannel_Call) Return(_a0 *milvuspb.StringResponse, _a1 error) *MockQueryNode_GetTimeTickChannel_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Init provides a mock function with given fields:
func (_m *MockQueryNode) Init() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockQueryNode_Init_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Init'
type MockQueryNode_Init_Call struct {
	*mock.Call
}

// Init is a helper method to define mock.On call
func (_e *MockQueryNode_Expecter) Init() *MockQueryNode_Init_Call {
	return &MockQueryNode_Init_Call{Call: _e.mock.On("Init")}
}

func (_c *MockQueryNode_Init_Call) Run(run func()) *MockQueryNode_Init_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockQueryNode_Init_Call) Return(_a0 error) *MockQueryNode_Init_Call {
	_c.Call.Return(_a0)
	return _c
}

// LoadSegments provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) LoadSegments(ctx context.Context, req *querypb.LoadSegmentsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.LoadSegmentsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.LoadSegmentsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_LoadSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'LoadSegments'
type MockQueryNode_LoadSegments_Call struct {
	*mock.Call
}

// LoadSegments is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.LoadSegmentsRequest
func (_e *MockQueryNode_Expecter) LoadSegments(ctx interface{}, req interface{}) *MockQueryNode_LoadSegments_Call {
	return &MockQueryNode_LoadSegments_Call{Call: _e.mock.On("LoadSegments", ctx, req)}
}

func (_c *MockQueryNode_LoadSegments_Call) Run(run func(ctx context.Context, req *querypb.LoadSegmentsRequest)) *MockQueryNode_LoadSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.LoadSegmentsRequest))
	})
	return _c
}

func (_c *MockQueryNode_LoadSegments_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_LoadSegments_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Query provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) Query(ctx context.Context, req *querypb.QueryRequest) (*internalpb.RetrieveResults, error) {
	ret := _m.Called(ctx, req)

	var r0 *internalpb.RetrieveResults
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.QueryRequest) *internalpb.RetrieveResults); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*internalpb.RetrieveResults)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.QueryRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_Query_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Query'
type MockQueryNode_Query_Call struct {
	*mock.Call
}

// Query is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.QueryRequest
func (_e *MockQueryNode_Expecter) Query(ctx interface{}, req interface{}) *MockQueryNode_Query_Call {
	return &MockQueryNode_Query_Call{Call: _e.mock.On("Query", ctx, req)}
}

func (_c *MockQueryNode_Query_Call) Run(run func(ctx context.Context, req *querypb.QueryRequest)) *MockQueryNode_Query_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.QueryRequest))
	})
	return _c
}

func (_c *MockQueryNode_Query_Call) Return(_a0 *internalpb.RetrieveResults, _a1 error) *MockQueryNode_Query_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Register provides a mock function with given fields:
func (_m *MockQueryNode) Register() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockQueryNode_Register_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Register'
type MockQueryNode_Register_Call struct {
	*mock.Call
}

// Register is a helper method to define mock.On call
func (_e *MockQueryNode_Expecter) Register() *MockQueryNode_Register_Call {
	return &MockQueryNode_Register_Call{Call: _e.mock.On("Register")}
}

func (_c *MockQueryNode_Register_Call) Run(run func()) *MockQueryNode_Register_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockQueryNode_Register_Call) Return(_a0 error) *MockQueryNode_Register_Call {
	_c.Call.Return(_a0)
	return _c
}

// ReleaseCollection provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) ReleaseCollection(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.ReleaseCollectionRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.ReleaseCollectionRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_ReleaseCollection_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ReleaseCollection'
type MockQueryNode_ReleaseCollection_Call struct {
	*mock.Call
}

// ReleaseCollection is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.ReleaseCollectionRequest
func (_e *MockQueryNode_Expecter) ReleaseCollection(ctx interface{}, req interface{}) *MockQueryNode_ReleaseCollection_Call {
	return &MockQueryNode_ReleaseCollection_Call{Call: _e.mock.On("ReleaseCollection", ctx, req)}
}

func (_c *MockQueryNode_ReleaseCollection_Call) Run(run func(ctx context.Context, req *querypb.ReleaseCollectionRequest)) *MockQueryNode_ReleaseCollection_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.ReleaseCollectionRequest))
	})
	return _c
}

func (_c *MockQueryNode_ReleaseCollection_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_ReleaseCollection_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// ReleasePartitions provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) ReleasePartitions(ctx context.Context, req *querypb.ReleasePartitionsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.ReleasePartitionsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.ReleasePartitionsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_ReleasePartitions_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ReleasePartitions'
type MockQueryNode_ReleasePartitions_Call struct {
	*mock.Call
}

// ReleasePartitions is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.ReleasePartitionsRequest
func (_e *MockQueryNode_Expecter) ReleasePartitions(ctx interface{}, req interface{}) *MockQueryNode_ReleasePartitions_Call {
	return &MockQueryNode_ReleasePartitions_Call{Call: _e.mock.On("ReleasePartitions", ctx, req)}
}

func (_c *MockQueryNode_ReleasePartitions_Call) Run(run func(ctx context.Context, req *querypb.ReleasePartitionsRequest)) *MockQueryNode_ReleasePartitions_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.ReleasePartitionsRequest))
	})
	return _c
}

func (_c *MockQueryNode_ReleasePartitions_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_ReleasePartitions_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// ReleaseSegments provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) ReleaseSegments(ctx context.Context, req *querypb.ReleaseSegmentsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.ReleaseSegmentsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.ReleaseSegmentsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_ReleaseSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ReleaseSegments'
type MockQueryNode_ReleaseSegments_Call struct {
	*mock.Call
}

// ReleaseSegments is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.ReleaseSegmentsRequest
func (_e *MockQueryNode_Expecter) ReleaseSegments(ctx interface{}, req interface{}) *MockQueryNode_ReleaseSegments_Call {
	return &MockQueryNode_ReleaseSegments_Call{Call: _e.mock.On("ReleaseSegments", ctx, req)}
}

func (_c *MockQueryNode_ReleaseSegments_Call) Run(run func(ctx context.Context, req *querypb.ReleaseSegmentsRequest)) *MockQueryNode_ReleaseSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.ReleaseSegmentsRequest))
	})
	return _c
}

func (_c *MockQueryNode_ReleaseSegments_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_ReleaseSegments_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Search provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) Search(ctx context.Context, req *querypb.SearchRequest) (*internalpb.SearchResults, error) {
	ret := _m.Called(ctx, req)

	var r0 *internalpb.SearchResults
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.SearchRequest) *internalpb.SearchResults); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*internalpb.SearchResults)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.SearchRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_Search_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Search'
type MockQueryNode_Search_Call struct {
	*mock.Call
}

// Search is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.SearchRequest
func (_e *MockQueryNode_Expecter) Search(ctx interface{}, req interface{}) *MockQueryNode_Search_Call {
	return &MockQueryNode_Search_Call{Call: _e.mock.On("Search", ctx, req)}
}

func (_c *MockQueryNode_Search_Call) Run(run func(ctx context.Context, req *querypb.SearchRequest)) *MockQueryNode_Search_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.SearchRequest))
	})
	return _c
}

func (_c *MockQueryNode_Search_Call) Return(_a0 *internalpb.SearchResults, _a1 error) *MockQueryNode_Search_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// ShowConfigurations provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) ShowConfigurations(ctx context.Context, req *internalpb.ShowConfigurationsRequest) (*internalpb.ShowConfigurationsResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *internalpb.ShowConfigurationsResponse
	if rf, ok := ret.Get(0).(func(context.Context, *internalpb.ShowConfigurationsRequest) *internalpb.ShowConfigurationsResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*internalpb.ShowConfigurationsResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *internalpb.ShowConfigurationsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_ShowConfigurations_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ShowConfigurations'
type MockQueryNode_ShowConfigurations_Call struct {
	*mock.Call
}

// ShowConfigurations is a helper method to define mock.On call
//  - ctx context.Context
//  - req *internalpb.ShowConfigurationsRequest
func (_e *MockQueryNode_Expecter) ShowConfigurations(ctx interface{}, req interface{}) *MockQueryNode_ShowConfigurations_Call {
	return &MockQueryNode_ShowConfigurations_Call{Call: _e.mock.On("ShowConfigurations", ctx, req)}
}

func (_c *MockQueryNode_ShowConfigurations_Call) Run(run func(ctx context.Context, req *internalpb.ShowConfigurationsRequest)) *MockQueryNode_ShowConfigurations_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*internalpb.ShowConfigurationsRequest))
	})
	return _c
}

func (_c *MockQueryNode_ShowConfigurations_Call) Return(_a0 *internalpb.ShowConfigurationsResponse, _a1 error) *MockQueryNode_ShowConfigurations_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Start provides a mock function with given fields:
func (_m *MockQueryNode) Start() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockQueryNode_Start_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Start'
type MockQueryNode_Start_Call struct {
	*mock.Call
}

// Start is a helper method to define mock.On call
func (_e *MockQueryNode_Expecter) Start() *MockQueryNode_Start_Call {
	return &MockQueryNode_Start_Call{Call: _e.mock.On("Start")}
}

func (_c *MockQueryNode_Start_Call) Run(run func()) *MockQueryNode_Start_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockQueryNode_Start_Call) Return(_a0 error) *MockQueryNode_Start_Call {
	_c.Call.Return(_a0)
	return _c
}

// Stop provides a mock function with given fields:
func (_m *MockQueryNode) Stop() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockQueryNode_Stop_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Stop'
type MockQueryNode_Stop_Call struct {
	*mock.Call
}

// Stop is a helper method to define mock.On call
func (_e *MockQueryNode_Expecter) Stop() *MockQueryNode_Stop_Call {
	return &MockQueryNode_Stop_Call{Call: _e.mock.On("Stop")}
}

func (_c *MockQueryNode_Stop_Call) Run(run func()) *MockQueryNode_Stop_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockQueryNode_Stop_Call) Return(_a0 error) *MockQueryNode_Stop_Call {
	_c.Call.Return(_a0)
	return _c
}

// SyncDistribution provides a mock function with given fields: _a0, _a1
func (_m *MockQueryNode) SyncDistribution(_a0 context.Context, _a1 *querypb.SyncDistributionRequest) (*commonpb.Status, error) {
	ret := _m.Called(_a0, _a1)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.SyncDistributionRequest) *commonpb.Status); ok {
		r0 = rf(_a0, _a1)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.SyncDistributionRequest) error); ok {
		r1 = rf(_a0, _a1)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_SyncDistribution_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SyncDistribution'
type MockQueryNode_SyncDistribution_Call struct {
	*mock.Call
}

// SyncDistribution is a helper method to define mock.On call
//  - _a0 context.Context
//  - _a1 *querypb.SyncDistributionRequest
func (_e *MockQueryNode_Expecter) SyncDistribution(_a0 interface{}, _a1 interface{}) *MockQueryNode_SyncDistribution_Call {
	return &MockQueryNode_SyncDistribution_Call{Call: _e.mock.On("SyncDistribution", _a0, _a1)}
}

func (_c *MockQueryNode_SyncDistribution_Call) Run(run func(_a0 context.Context, _a1 *querypb.SyncDistributionRequest)) *MockQueryNode_SyncDistribution_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.SyncDistributionRequest))
	})
	return _c
}

func (_c *MockQueryNode_SyncDistribution_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_SyncDistribution_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// SyncReplicaSegments provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) SyncReplicaSegments(ctx context.Context, req *querypb.SyncReplicaSegmentsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.SyncReplicaSegmentsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.SyncReplicaSegmentsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_SyncReplicaSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SyncReplicaSegments'
type MockQueryNode_SyncReplicaSegments_Call struct {
	*mock.Call
}

// SyncReplicaSegments is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.SyncReplicaSegmentsRequest
func (_e *MockQueryNode_Expecter) SyncReplicaSegments(ctx interface{}, req interface{}) *MockQueryNode_SyncReplicaSegments_Call {
	return &MockQueryNode_SyncReplicaSegments_Call{Call: _e.mock.On("SyncReplicaSegments", ctx, req)}
}

func (_c *MockQueryNode_SyncReplicaSegments_Call) Run(run func(ctx context.Context, req *querypb.SyncReplicaSegmentsRequest)) *MockQueryNode_SyncReplicaSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.SyncReplicaSegmentsRequest))
	})
	return _c
}

func (_c *MockQueryNode_SyncReplicaSegments_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_SyncReplicaSegments_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// UnsubDmChannel provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) UnsubDmChannel(ctx context.Context, req *querypb.UnsubDmChannelRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.UnsubDmChannelRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.UnsubDmChannelRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_UnsubDmChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UnsubDmChannel'
type MockQueryNode_UnsubDmChannel_Call struct {
	*mock.Call
}

// UnsubDmChannel is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.UnsubDmChannelRequest
func (_e *MockQueryNode_Expecter) UnsubDmChannel(ctx interface{}, req interface{}) *MockQueryNode_UnsubDmChannel_Call {
	return &MockQueryNode_UnsubDmChannel_Call{Call: _e.mock.On("UnsubDmChannel", ctx, req)}
}

func (_c *MockQueryNode_UnsubDmChannel_Call) Run(run func(ctx context.Context, req *querypb.UnsubDmChannelRequest)) *MockQueryNode_UnsubDmChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.UnsubDmChannelRequest))
	})
	return _c
}

func (_c *MockQueryNode_UnsubDmChannel_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_UnsubDmChannel_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// WatchDmChannels provides a mock function with given fields: ctx, req
func (_m *MockQueryNode) WatchDmChannels(ctx context.Context, req *querypb.WatchDmChannelsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *querypb.WatchDmChannelsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *querypb.WatchDmChannelsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockQueryNode_WatchDmChannels_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'WatchDmChannels'
type MockQueryNode_WatchDmChannels_Call struct {
	*mock.Call
}

// WatchDmChannels is a helper method to define mock.On call
//  - ctx context.Context
//  - req *querypb.WatchDmChannelsRequest
func (_e *MockQueryNode_Expecter) WatchDmChannels(ctx interface{}, req interface{}) *MockQueryNode_WatchDmChannels_Call {
	return &MockQueryNode_WatchDmChannels_Call{Call: _e.mock.On("WatchDmChannels", ctx, req)}
}

func (_c *MockQueryNode_WatchDmChannels_Call) Run(run func(ctx context.Context, req *querypb.WatchDmChannelsRequest)) *MockQueryNode_WatchDmChannels_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*querypb.WatchDmChannelsRequest))
	})
	return _c
}

func (_c *MockQueryNode_WatchDmChannels_Call) Return(_a0 *commonpb.Status, _a1 error) *MockQueryNode_WatchDmChannels_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

type mockConstructorTestingTNewMockQueryNode interface {
	mock.TestingT
	Cleanup(func())
}

// NewMockQueryNode creates a new instance of MockQueryNode. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
func NewMockQueryNode(t mockConstructorTestingTNewMockQueryNode) *MockQueryNode {
	mock := &MockQueryNode{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
