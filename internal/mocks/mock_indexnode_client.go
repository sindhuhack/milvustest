// Code generated by mockery v2.32.4. DO NOT EDIT.

package mocks

import (
	context "context"

	commonpb "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"

	grpc "google.golang.org/grpc"

	indexpb "github.com/milvus-io/milvus/internal/proto/indexpb"

	internalpb "github.com/milvus-io/milvus/internal/proto/internalpb"

	milvuspb "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"

	mock "github.com/stretchr/testify/mock"
)

// MockIndexNodeClient is an autogenerated mock type for the IndexNodeClient type
type MockIndexNodeClient struct {
	mock.Mock
}

type MockIndexNodeClient_Expecter struct {
	mock *mock.Mock
}

func (_m *MockIndexNodeClient) EXPECT() *MockIndexNodeClient_Expecter {
	return &MockIndexNodeClient_Expecter{mock: &_m.Mock}
}

// Close provides a mock function with given fields:
func (_m *MockIndexNodeClient) Close() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockIndexNodeClient_Close_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Close'
type MockIndexNodeClient_Close_Call struct {
	*mock.Call
}

// Close is a helper method to define mock.On call
func (_e *MockIndexNodeClient_Expecter) Close() *MockIndexNodeClient_Close_Call {
	return &MockIndexNodeClient_Close_Call{Call: _e.mock.On("Close")}
}

func (_c *MockIndexNodeClient_Close_Call) Run(run func()) *MockIndexNodeClient_Close_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockIndexNodeClient_Close_Call) Return(_a0 error) *MockIndexNodeClient_Close_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockIndexNodeClient_Close_Call) RunAndReturn(run func() error) *MockIndexNodeClient_Close_Call {
	_c.Call.Return(run)
	return _c
}

// CreateJob provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) CreateJob(ctx context.Context, in *indexpb.CreateJobRequest, opts ...grpc.CallOption) (*commonpb.Status, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *commonpb.Status
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.CreateJobRequest, ...grpc.CallOption) (*commonpb.Status, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.CreateJobRequest, ...grpc.CallOption) *commonpb.Status); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *indexpb.CreateJobRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_CreateJob_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CreateJob'
type MockIndexNodeClient_CreateJob_Call struct {
	*mock.Call
}

// CreateJob is a helper method to define mock.On call
//  - ctx context.Context
//  - in *indexpb.CreateJobRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) CreateJob(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_CreateJob_Call {
	return &MockIndexNodeClient_CreateJob_Call{Call: _e.mock.On("CreateJob",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_CreateJob_Call) Run(run func(ctx context.Context, in *indexpb.CreateJobRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_CreateJob_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*indexpb.CreateJobRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_CreateJob_Call) Return(_a0 *commonpb.Status, _a1 error) *MockIndexNodeClient_CreateJob_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_CreateJob_Call) RunAndReturn(run func(context.Context, *indexpb.CreateJobRequest, ...grpc.CallOption) (*commonpb.Status, error)) *MockIndexNodeClient_CreateJob_Call {
	_c.Call.Return(run)
	return _c
}

// DropJobs provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) DropJobs(ctx context.Context, in *indexpb.DropJobsRequest, opts ...grpc.CallOption) (*commonpb.Status, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *commonpb.Status
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.DropJobsRequest, ...grpc.CallOption) (*commonpb.Status, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.DropJobsRequest, ...grpc.CallOption) *commonpb.Status); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *indexpb.DropJobsRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_DropJobs_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'DropJobs'
type MockIndexNodeClient_DropJobs_Call struct {
	*mock.Call
}

// DropJobs is a helper method to define mock.On call
//  - ctx context.Context
//  - in *indexpb.DropJobsRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) DropJobs(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_DropJobs_Call {
	return &MockIndexNodeClient_DropJobs_Call{Call: _e.mock.On("DropJobs",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_DropJobs_Call) Run(run func(ctx context.Context, in *indexpb.DropJobsRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_DropJobs_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*indexpb.DropJobsRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_DropJobs_Call) Return(_a0 *commonpb.Status, _a1 error) *MockIndexNodeClient_DropJobs_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_DropJobs_Call) RunAndReturn(run func(context.Context, *indexpb.DropJobsRequest, ...grpc.CallOption) (*commonpb.Status, error)) *MockIndexNodeClient_DropJobs_Call {
	_c.Call.Return(run)
	return _c
}

// GetComponentStates provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) GetComponentStates(ctx context.Context, in *milvuspb.GetComponentStatesRequest, opts ...grpc.CallOption) (*milvuspb.ComponentStates, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *milvuspb.ComponentStates
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *milvuspb.GetComponentStatesRequest, ...grpc.CallOption) (*milvuspb.ComponentStates, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *milvuspb.GetComponentStatesRequest, ...grpc.CallOption) *milvuspb.ComponentStates); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.ComponentStates)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *milvuspb.GetComponentStatesRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_GetComponentStates_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetComponentStates'
type MockIndexNodeClient_GetComponentStates_Call struct {
	*mock.Call
}

// GetComponentStates is a helper method to define mock.On call
//  - ctx context.Context
//  - in *milvuspb.GetComponentStatesRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) GetComponentStates(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_GetComponentStates_Call {
	return &MockIndexNodeClient_GetComponentStates_Call{Call: _e.mock.On("GetComponentStates",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_GetComponentStates_Call) Run(run func(ctx context.Context, in *milvuspb.GetComponentStatesRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_GetComponentStates_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*milvuspb.GetComponentStatesRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_GetComponentStates_Call) Return(_a0 *milvuspb.ComponentStates, _a1 error) *MockIndexNodeClient_GetComponentStates_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_GetComponentStates_Call) RunAndReturn(run func(context.Context, *milvuspb.GetComponentStatesRequest, ...grpc.CallOption) (*milvuspb.ComponentStates, error)) *MockIndexNodeClient_GetComponentStates_Call {
	_c.Call.Return(run)
	return _c
}

// GetJobStats provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) GetJobStats(ctx context.Context, in *indexpb.GetJobStatsRequest, opts ...grpc.CallOption) (*indexpb.GetJobStatsResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *indexpb.GetJobStatsResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.GetJobStatsRequest, ...grpc.CallOption) (*indexpb.GetJobStatsResponse, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.GetJobStatsRequest, ...grpc.CallOption) *indexpb.GetJobStatsResponse); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*indexpb.GetJobStatsResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *indexpb.GetJobStatsRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_GetJobStats_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetJobStats'
type MockIndexNodeClient_GetJobStats_Call struct {
	*mock.Call
}

// GetJobStats is a helper method to define mock.On call
//  - ctx context.Context
//  - in *indexpb.GetJobStatsRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) GetJobStats(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_GetJobStats_Call {
	return &MockIndexNodeClient_GetJobStats_Call{Call: _e.mock.On("GetJobStats",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_GetJobStats_Call) Run(run func(ctx context.Context, in *indexpb.GetJobStatsRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_GetJobStats_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*indexpb.GetJobStatsRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_GetJobStats_Call) Return(_a0 *indexpb.GetJobStatsResponse, _a1 error) *MockIndexNodeClient_GetJobStats_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_GetJobStats_Call) RunAndReturn(run func(context.Context, *indexpb.GetJobStatsRequest, ...grpc.CallOption) (*indexpb.GetJobStatsResponse, error)) *MockIndexNodeClient_GetJobStats_Call {
	_c.Call.Return(run)
	return _c
}

// GetMetrics provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) GetMetrics(ctx context.Context, in *milvuspb.GetMetricsRequest, opts ...grpc.CallOption) (*milvuspb.GetMetricsResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *milvuspb.GetMetricsResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *milvuspb.GetMetricsRequest, ...grpc.CallOption) (*milvuspb.GetMetricsResponse, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *milvuspb.GetMetricsRequest, ...grpc.CallOption) *milvuspb.GetMetricsResponse); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.GetMetricsResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *milvuspb.GetMetricsRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_GetMetrics_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetMetrics'
type MockIndexNodeClient_GetMetrics_Call struct {
	*mock.Call
}

// GetMetrics is a helper method to define mock.On call
//  - ctx context.Context
//  - in *milvuspb.GetMetricsRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) GetMetrics(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_GetMetrics_Call {
	return &MockIndexNodeClient_GetMetrics_Call{Call: _e.mock.On("GetMetrics",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_GetMetrics_Call) Run(run func(ctx context.Context, in *milvuspb.GetMetricsRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_GetMetrics_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*milvuspb.GetMetricsRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_GetMetrics_Call) Return(_a0 *milvuspb.GetMetricsResponse, _a1 error) *MockIndexNodeClient_GetMetrics_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_GetMetrics_Call) RunAndReturn(run func(context.Context, *milvuspb.GetMetricsRequest, ...grpc.CallOption) (*milvuspb.GetMetricsResponse, error)) *MockIndexNodeClient_GetMetrics_Call {
	_c.Call.Return(run)
	return _c
}

// GetStatisticsChannel provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) GetStatisticsChannel(ctx context.Context, in *internalpb.GetStatisticsChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *milvuspb.StringResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *internalpb.GetStatisticsChannelRequest, ...grpc.CallOption) (*milvuspb.StringResponse, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *internalpb.GetStatisticsChannelRequest, ...grpc.CallOption) *milvuspb.StringResponse); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.StringResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *internalpb.GetStatisticsChannelRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_GetStatisticsChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetStatisticsChannel'
type MockIndexNodeClient_GetStatisticsChannel_Call struct {
	*mock.Call
}

// GetStatisticsChannel is a helper method to define mock.On call
//  - ctx context.Context
//  - in *internalpb.GetStatisticsChannelRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) GetStatisticsChannel(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_GetStatisticsChannel_Call {
	return &MockIndexNodeClient_GetStatisticsChannel_Call{Call: _e.mock.On("GetStatisticsChannel",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_GetStatisticsChannel_Call) Run(run func(ctx context.Context, in *internalpb.GetStatisticsChannelRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_GetStatisticsChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*internalpb.GetStatisticsChannelRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_GetStatisticsChannel_Call) Return(_a0 *milvuspb.StringResponse, _a1 error) *MockIndexNodeClient_GetStatisticsChannel_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_GetStatisticsChannel_Call) RunAndReturn(run func(context.Context, *internalpb.GetStatisticsChannelRequest, ...grpc.CallOption) (*milvuspb.StringResponse, error)) *MockIndexNodeClient_GetStatisticsChannel_Call {
	_c.Call.Return(run)
	return _c
}

// QueryJobs provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) QueryJobs(ctx context.Context, in *indexpb.QueryJobsRequest, opts ...grpc.CallOption) (*indexpb.QueryJobsResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *indexpb.QueryJobsResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.QueryJobsRequest, ...grpc.CallOption) (*indexpb.QueryJobsResponse, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *indexpb.QueryJobsRequest, ...grpc.CallOption) *indexpb.QueryJobsResponse); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*indexpb.QueryJobsResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *indexpb.QueryJobsRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_QueryJobs_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'QueryJobs'
type MockIndexNodeClient_QueryJobs_Call struct {
	*mock.Call
}

// QueryJobs is a helper method to define mock.On call
//  - ctx context.Context
//  - in *indexpb.QueryJobsRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) QueryJobs(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_QueryJobs_Call {
	return &MockIndexNodeClient_QueryJobs_Call{Call: _e.mock.On("QueryJobs",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_QueryJobs_Call) Run(run func(ctx context.Context, in *indexpb.QueryJobsRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_QueryJobs_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*indexpb.QueryJobsRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_QueryJobs_Call) Return(_a0 *indexpb.QueryJobsResponse, _a1 error) *MockIndexNodeClient_QueryJobs_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_QueryJobs_Call) RunAndReturn(run func(context.Context, *indexpb.QueryJobsRequest, ...grpc.CallOption) (*indexpb.QueryJobsResponse, error)) *MockIndexNodeClient_QueryJobs_Call {
	_c.Call.Return(run)
	return _c
}

// ShowConfigurations provides a mock function with given fields: ctx, in, opts
func (_m *MockIndexNodeClient) ShowConfigurations(ctx context.Context, in *internalpb.ShowConfigurationsRequest, opts ...grpc.CallOption) (*internalpb.ShowConfigurationsResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *internalpb.ShowConfigurationsResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *internalpb.ShowConfigurationsRequest, ...grpc.CallOption) (*internalpb.ShowConfigurationsResponse, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *internalpb.ShowConfigurationsRequest, ...grpc.CallOption) *internalpb.ShowConfigurationsResponse); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*internalpb.ShowConfigurationsResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *internalpb.ShowConfigurationsRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockIndexNodeClient_ShowConfigurations_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ShowConfigurations'
type MockIndexNodeClient_ShowConfigurations_Call struct {
	*mock.Call
}

// ShowConfigurations is a helper method to define mock.On call
//  - ctx context.Context
//  - in *internalpb.ShowConfigurationsRequest
//  - opts ...grpc.CallOption
func (_e *MockIndexNodeClient_Expecter) ShowConfigurations(ctx interface{}, in interface{}, opts ...interface{}) *MockIndexNodeClient_ShowConfigurations_Call {
	return &MockIndexNodeClient_ShowConfigurations_Call{Call: _e.mock.On("ShowConfigurations",
		append([]interface{}{ctx, in}, opts...)...)}
}

func (_c *MockIndexNodeClient_ShowConfigurations_Call) Run(run func(ctx context.Context, in *internalpb.ShowConfigurationsRequest, opts ...grpc.CallOption)) *MockIndexNodeClient_ShowConfigurations_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*internalpb.ShowConfigurationsRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockIndexNodeClient_ShowConfigurations_Call) Return(_a0 *internalpb.ShowConfigurationsResponse, _a1 error) *MockIndexNodeClient_ShowConfigurations_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockIndexNodeClient_ShowConfigurations_Call) RunAndReturn(run func(context.Context, *internalpb.ShowConfigurationsRequest, ...grpc.CallOption) (*internalpb.ShowConfigurationsResponse, error)) *MockIndexNodeClient_ShowConfigurations_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockIndexNodeClient creates a new instance of MockIndexNodeClient. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockIndexNodeClient(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockIndexNodeClient {
	mock := &MockIndexNodeClient{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
