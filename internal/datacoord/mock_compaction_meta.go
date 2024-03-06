// Code generated by mockery v2.32.4. DO NOT EDIT.

package datacoord

import (
	datapb "github.com/milvus-io/milvus/internal/proto/datapb"
	mock "github.com/stretchr/testify/mock"
)

// MockCompactionMeta is an autogenerated mock type for the CompactionMeta type
type MockCompactionMeta struct {
	mock.Mock
}

type MockCompactionMeta_Expecter struct {
	mock *mock.Mock
}

func (_m *MockCompactionMeta) EXPECT() *MockCompactionMeta_Expecter {
	return &MockCompactionMeta_Expecter{mock: &_m.Mock}
}

// CompleteCompactionMutation provides a mock function with given fields: plan, result
func (_m *MockCompactionMeta) CompleteCompactionMutation(plan *datapb.CompactionPlan, result *datapb.CompactionPlanResult) ([]*SegmentInfo, *SegMetricMutation, error) {
	ret := _m.Called(plan, result)

	var r0 []*SegmentInfo
	var r1 *SegMetricMutation
	var r2 error
	if rf, ok := ret.Get(0).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) ([]*SegmentInfo, *SegMetricMutation, error)); ok {
		return rf(plan, result)
	}
	if rf, ok := ret.Get(0).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) []*SegmentInfo); ok {
		r0 = rf(plan, result)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*SegmentInfo)
		}
	}

	if rf, ok := ret.Get(1).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) *SegMetricMutation); ok {
		r1 = rf(plan, result)
	} else {
		if ret.Get(1) != nil {
			r1 = ret.Get(1).(*SegMetricMutation)
		}
	}

	if rf, ok := ret.Get(2).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) error); ok {
		r2 = rf(plan, result)
	} else {
		r2 = ret.Error(2)
	}

	return r0, r1, r2
}

// MockCompactionMeta_CompleteCompactionMutation_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CompleteCompactionMutation'
type MockCompactionMeta_CompleteCompactionMutation_Call struct {
	*mock.Call
}

// CompleteCompactionMutation is a helper method to define mock.On call
//   - plan *datapb.CompactionPlan
//   - result *datapb.CompactionPlanResult
func (_e *MockCompactionMeta_Expecter) CompleteCompactionMutation(plan interface{}, result interface{}) *MockCompactionMeta_CompleteCompactionMutation_Call {
	return &MockCompactionMeta_CompleteCompactionMutation_Call{Call: _e.mock.On("CompleteCompactionMutation", plan, result)}
}

func (_c *MockCompactionMeta_CompleteCompactionMutation_Call) Run(run func(plan *datapb.CompactionPlan, result *datapb.CompactionPlanResult)) *MockCompactionMeta_CompleteCompactionMutation_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*datapb.CompactionPlan), args[1].(*datapb.CompactionPlanResult))
	})
	return _c
}

func (_c *MockCompactionMeta_CompleteCompactionMutation_Call) Return(_a0 []*SegmentInfo, _a1 *SegMetricMutation, _a2 error) *MockCompactionMeta_CompleteCompactionMutation_Call {
	_c.Call.Return(_a0, _a1, _a2)
	return _c
}

func (_c *MockCompactionMeta_CompleteCompactionMutation_Call) RunAndReturn(run func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) ([]*SegmentInfo, *SegMetricMutation, error)) *MockCompactionMeta_CompleteCompactionMutation_Call {
	_c.Call.Return(run)
	return _c
}

// GetHealthySegment provides a mock function with given fields: segID
func (_m *MockCompactionMeta) GetHealthySegment(segID int64) *SegmentInfo {
	ret := _m.Called(segID)

	var r0 *SegmentInfo
	if rf, ok := ret.Get(0).(func(int64) *SegmentInfo); ok {
		r0 = rf(segID)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*SegmentInfo)
		}
	}

	return r0
}

// MockCompactionMeta_GetHealthySegment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetHealthySegment'
type MockCompactionMeta_GetHealthySegment_Call struct {
	*mock.Call
}

// GetHealthySegment is a helper method to define mock.On call
//   - segID int64
func (_e *MockCompactionMeta_Expecter) GetHealthySegment(segID interface{}) *MockCompactionMeta_GetHealthySegment_Call {
	return &MockCompactionMeta_GetHealthySegment_Call{Call: _e.mock.On("GetHealthySegment", segID)}
}

func (_c *MockCompactionMeta_GetHealthySegment_Call) Run(run func(segID int64)) *MockCompactionMeta_GetHealthySegment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockCompactionMeta_GetHealthySegment_Call) Return(_a0 *SegmentInfo) *MockCompactionMeta_GetHealthySegment_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_GetHealthySegment_Call) RunAndReturn(run func(int64) *SegmentInfo) *MockCompactionMeta_GetHealthySegment_Call {
	_c.Call.Return(run)
	return _c
}

// SelectSegments provides a mock function with given fields: selector
func (_m *MockCompactionMeta) SelectSegments(selector SegmentInfoSelector) []*SegmentInfo {
	ret := _m.Called(selector)

	var r0 []*SegmentInfo
	if rf, ok := ret.Get(0).(func(SegmentInfoSelector) []*SegmentInfo); ok {
		r0 = rf(selector)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*SegmentInfo)
		}
	}

	return r0
}

// MockCompactionMeta_SelectSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SelectSegments'
type MockCompactionMeta_SelectSegments_Call struct {
	*mock.Call
}

// SelectSegments is a helper method to define mock.On call
//   - selector SegmentInfoSelector
func (_e *MockCompactionMeta_Expecter) SelectSegments(selector interface{}) *MockCompactionMeta_SelectSegments_Call {
	return &MockCompactionMeta_SelectSegments_Call{Call: _e.mock.On("SelectSegments", selector)}
}

func (_c *MockCompactionMeta_SelectSegments_Call) Run(run func(selector SegmentInfoSelector)) *MockCompactionMeta_SelectSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(SegmentInfoSelector))
	})
	return _c
}

func (_c *MockCompactionMeta_SelectSegments_Call) Return(_a0 []*SegmentInfo) *MockCompactionMeta_SelectSegments_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_SelectSegments_Call) RunAndReturn(run func(SegmentInfoSelector) []*SegmentInfo) *MockCompactionMeta_SelectSegments_Call {
	_c.Call.Return(run)
	return _c
}

// SetSegmentCompacting provides a mock function with given fields: segmentID, compacting
func (_m *MockCompactionMeta) SetSegmentCompacting(segmentID int64, compacting bool) {
	_m.Called(segmentID, compacting)
}

// MockCompactionMeta_SetSegmentCompacting_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetSegmentCompacting'
type MockCompactionMeta_SetSegmentCompacting_Call struct {
	*mock.Call
}

// SetSegmentCompacting is a helper method to define mock.On call
//   - segmentID int64
//   - compacting bool
func (_e *MockCompactionMeta_Expecter) SetSegmentCompacting(segmentID interface{}, compacting interface{}) *MockCompactionMeta_SetSegmentCompacting_Call {
	return &MockCompactionMeta_SetSegmentCompacting_Call{Call: _e.mock.On("SetSegmentCompacting", segmentID, compacting)}
}

func (_c *MockCompactionMeta_SetSegmentCompacting_Call) Run(run func(segmentID int64, compacting bool)) *MockCompactionMeta_SetSegmentCompacting_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64), args[1].(bool))
	})
	return _c
}

func (_c *MockCompactionMeta_SetSegmentCompacting_Call) Return() *MockCompactionMeta_SetSegmentCompacting_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockCompactionMeta_SetSegmentCompacting_Call) RunAndReturn(run func(int64, bool)) *MockCompactionMeta_SetSegmentCompacting_Call {
	_c.Call.Return(run)
	return _c
}

// UpdateSegmentsInfo provides a mock function with given fields: operators
func (_m *MockCompactionMeta) UpdateSegmentsInfo(operators ...UpdateOperator) error {
	_va := make([]interface{}, len(operators))
	for _i := range operators {
		_va[_i] = operators[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 error
	if rf, ok := ret.Get(0).(func(...UpdateOperator) error); ok {
		r0 = rf(operators...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockCompactionMeta_UpdateSegmentsInfo_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UpdateSegmentsInfo'
type MockCompactionMeta_UpdateSegmentsInfo_Call struct {
	*mock.Call
}

// UpdateSegmentsInfo is a helper method to define mock.On call
//   - operators ...UpdateOperator
func (_e *MockCompactionMeta_Expecter) UpdateSegmentsInfo(operators ...interface{}) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	return &MockCompactionMeta_UpdateSegmentsInfo_Call{Call: _e.mock.On("UpdateSegmentsInfo",
		append([]interface{}{}, operators...)...)}
}

func (_c *MockCompactionMeta_UpdateSegmentsInfo_Call) Run(run func(operators ...UpdateOperator)) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]UpdateOperator, len(args)-0)
		for i, a := range args[0:] {
			if a != nil {
				variadicArgs[i] = a.(UpdateOperator)
			}
		}
		run(variadicArgs...)
	})
	return _c
}

func (_c *MockCompactionMeta_UpdateSegmentsInfo_Call) Return(_a0 error) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_UpdateSegmentsInfo_Call) RunAndReturn(run func(...UpdateOperator) error) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockCompactionMeta creates a new instance of MockCompactionMeta. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockCompactionMeta(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockCompactionMeta {
	mock := &MockCompactionMeta{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
