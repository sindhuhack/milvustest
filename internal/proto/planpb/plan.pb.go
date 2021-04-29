// Code generated by protoc-gen-go. DO NOT EDIT.
// source: plan.proto

package planpb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	schemapb "github.com/milvus-io/milvus/internal/proto/schemapb"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type RangeExpr_OpType int32

const (
	RangeExpr_Invalid      RangeExpr_OpType = 0
	RangeExpr_GreaterThan  RangeExpr_OpType = 1
	RangeExpr_GreaterEqual RangeExpr_OpType = 2
	RangeExpr_LessThan     RangeExpr_OpType = 3
	RangeExpr_LessEqual    RangeExpr_OpType = 4
	RangeExpr_Equal        RangeExpr_OpType = 5
	RangeExpr_NotEqual     RangeExpr_OpType = 6
)

var RangeExpr_OpType_name = map[int32]string{
	0: "Invalid",
	1: "GreaterThan",
	2: "GreaterEqual",
	3: "LessThan",
	4: "LessEqual",
	5: "Equal",
	6: "NotEqual",
}

var RangeExpr_OpType_value = map[string]int32{
	"Invalid":      0,
	"GreaterThan":  1,
	"GreaterEqual": 2,
	"LessThan":     3,
	"LessEqual":    4,
	"Equal":        5,
	"NotEqual":     6,
}

func (x RangeExpr_OpType) String() string {
	return proto.EnumName(RangeExpr_OpType_name, int32(x))
}

func (RangeExpr_OpType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{3, 0}
}

type UnaryExpr_UnaryOp int32

const (
	UnaryExpr_Invalid UnaryExpr_UnaryOp = 0
	UnaryExpr_Not     UnaryExpr_UnaryOp = 1
)

var UnaryExpr_UnaryOp_name = map[int32]string{
	0: "Invalid",
	1: "Not",
}

var UnaryExpr_UnaryOp_value = map[string]int32{
	"Invalid": 0,
	"Not":     1,
}

func (x UnaryExpr_UnaryOp) String() string {
	return proto.EnumName(UnaryExpr_UnaryOp_name, int32(x))
}

func (UnaryExpr_UnaryOp) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{5, 0}
}

type BinaryExpr_BinaryOp int32

const (
	BinaryExpr_Invalid    BinaryExpr_BinaryOp = 0
	BinaryExpr_LogicalAnd BinaryExpr_BinaryOp = 1
	BinaryExpr_LogicalOr  BinaryExpr_BinaryOp = 2
)

var BinaryExpr_BinaryOp_name = map[int32]string{
	0: "Invalid",
	1: "LogicalAnd",
	2: "LogicalOr",
}

var BinaryExpr_BinaryOp_value = map[string]int32{
	"Invalid":    0,
	"LogicalAnd": 1,
	"LogicalOr":  2,
}

func (x BinaryExpr_BinaryOp) String() string {
	return proto.EnumName(BinaryExpr_BinaryOp_name, int32(x))
}

func (BinaryExpr_BinaryOp) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{6, 0}
}

type GenericValue struct {
	// Types that are valid to be assigned to Val:
	//	*GenericValue_BoolVal
	//	*GenericValue_Int64Val
	//	*GenericValue_FloatVal
	Val                  isGenericValue_Val `protobuf_oneof:"val"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *GenericValue) Reset()         { *m = GenericValue{} }
func (m *GenericValue) String() string { return proto.CompactTextString(m) }
func (*GenericValue) ProtoMessage()    {}
func (*GenericValue) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{0}
}

func (m *GenericValue) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_GenericValue.Unmarshal(m, b)
}
func (m *GenericValue) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_GenericValue.Marshal(b, m, deterministic)
}
func (m *GenericValue) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GenericValue.Merge(m, src)
}
func (m *GenericValue) XXX_Size() int {
	return xxx_messageInfo_GenericValue.Size(m)
}
func (m *GenericValue) XXX_DiscardUnknown() {
	xxx_messageInfo_GenericValue.DiscardUnknown(m)
}

var xxx_messageInfo_GenericValue proto.InternalMessageInfo

type isGenericValue_Val interface {
	isGenericValue_Val()
}

type GenericValue_BoolVal struct {
	BoolVal bool `protobuf:"varint,1,opt,name=bool_val,json=boolVal,proto3,oneof"`
}

type GenericValue_Int64Val struct {
	Int64Val int64 `protobuf:"varint,2,opt,name=int64_val,json=int64Val,proto3,oneof"`
}

type GenericValue_FloatVal struct {
	FloatVal float64 `protobuf:"fixed64,3,opt,name=float_val,json=floatVal,proto3,oneof"`
}

func (*GenericValue_BoolVal) isGenericValue_Val() {}

func (*GenericValue_Int64Val) isGenericValue_Val() {}

func (*GenericValue_FloatVal) isGenericValue_Val() {}

func (m *GenericValue) GetVal() isGenericValue_Val {
	if m != nil {
		return m.Val
	}
	return nil
}

func (m *GenericValue) GetBoolVal() bool {
	if x, ok := m.GetVal().(*GenericValue_BoolVal); ok {
		return x.BoolVal
	}
	return false
}

func (m *GenericValue) GetInt64Val() int64 {
	if x, ok := m.GetVal().(*GenericValue_Int64Val); ok {
		return x.Int64Val
	}
	return 0
}

func (m *GenericValue) GetFloatVal() float64 {
	if x, ok := m.GetVal().(*GenericValue_FloatVal); ok {
		return x.FloatVal
	}
	return 0
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*GenericValue) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*GenericValue_BoolVal)(nil),
		(*GenericValue_Int64Val)(nil),
		(*GenericValue_FloatVal)(nil),
	}
}

type QueryInfo struct {
	Topk                 int64    `protobuf:"varint,1,opt,name=topk,proto3" json:"topk,omitempty"`
	MetricType           string   `protobuf:"bytes,3,opt,name=metric_type,json=metricType,proto3" json:"metric_type,omitempty"`
	SearchParams         string   `protobuf:"bytes,4,opt,name=search_params,json=searchParams,proto3" json:"search_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *QueryInfo) Reset()         { *m = QueryInfo{} }
func (m *QueryInfo) String() string { return proto.CompactTextString(m) }
func (*QueryInfo) ProtoMessage()    {}
func (*QueryInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{1}
}

func (m *QueryInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_QueryInfo.Unmarshal(m, b)
}
func (m *QueryInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_QueryInfo.Marshal(b, m, deterministic)
}
func (m *QueryInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_QueryInfo.Merge(m, src)
}
func (m *QueryInfo) XXX_Size() int {
	return xxx_messageInfo_QueryInfo.Size(m)
}
func (m *QueryInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_QueryInfo.DiscardUnknown(m)
}

var xxx_messageInfo_QueryInfo proto.InternalMessageInfo

func (m *QueryInfo) GetTopk() int64 {
	if m != nil {
		return m.Topk
	}
	return 0
}

func (m *QueryInfo) GetMetricType() string {
	if m != nil {
		return m.MetricType
	}
	return ""
}

func (m *QueryInfo) GetSearchParams() string {
	if m != nil {
		return m.SearchParams
	}
	return ""
}

type ColumnInfo struct {
	FieldId              int64             `protobuf:"varint,1,opt,name=field_id,json=fieldId,proto3" json:"field_id,omitempty"`
	DataType             schemapb.DataType `protobuf:"varint,2,opt,name=data_type,json=dataType,proto3,enum=milvus.proto.schema.DataType" json:"data_type,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *ColumnInfo) Reset()         { *m = ColumnInfo{} }
func (m *ColumnInfo) String() string { return proto.CompactTextString(m) }
func (*ColumnInfo) ProtoMessage()    {}
func (*ColumnInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{2}
}

func (m *ColumnInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ColumnInfo.Unmarshal(m, b)
}
func (m *ColumnInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ColumnInfo.Marshal(b, m, deterministic)
}
func (m *ColumnInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ColumnInfo.Merge(m, src)
}
func (m *ColumnInfo) XXX_Size() int {
	return xxx_messageInfo_ColumnInfo.Size(m)
}
func (m *ColumnInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_ColumnInfo.DiscardUnknown(m)
}

var xxx_messageInfo_ColumnInfo proto.InternalMessageInfo

func (m *ColumnInfo) GetFieldId() int64 {
	if m != nil {
		return m.FieldId
	}
	return 0
}

func (m *ColumnInfo) GetDataType() schemapb.DataType {
	if m != nil {
		return m.DataType
	}
	return schemapb.DataType_None
}

type RangeExpr struct {
	ColumnInfo           *ColumnInfo        `protobuf:"bytes,1,opt,name=column_info,json=columnInfo,proto3" json:"column_info,omitempty"`
	Ops                  []RangeExpr_OpType `protobuf:"varint,2,rep,packed,name=ops,proto3,enum=milvus.proto.plan.RangeExpr_OpType" json:"ops,omitempty"`
	Values               []*GenericValue    `protobuf:"bytes,3,rep,name=values,proto3" json:"values,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *RangeExpr) Reset()         { *m = RangeExpr{} }
func (m *RangeExpr) String() string { return proto.CompactTextString(m) }
func (*RangeExpr) ProtoMessage()    {}
func (*RangeExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{3}
}

func (m *RangeExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RangeExpr.Unmarshal(m, b)
}
func (m *RangeExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RangeExpr.Marshal(b, m, deterministic)
}
func (m *RangeExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RangeExpr.Merge(m, src)
}
func (m *RangeExpr) XXX_Size() int {
	return xxx_messageInfo_RangeExpr.Size(m)
}
func (m *RangeExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_RangeExpr.DiscardUnknown(m)
}

var xxx_messageInfo_RangeExpr proto.InternalMessageInfo

func (m *RangeExpr) GetColumnInfo() *ColumnInfo {
	if m != nil {
		return m.ColumnInfo
	}
	return nil
}

func (m *RangeExpr) GetOps() []RangeExpr_OpType {
	if m != nil {
		return m.Ops
	}
	return nil
}

func (m *RangeExpr) GetValues() []*GenericValue {
	if m != nil {
		return m.Values
	}
	return nil
}

type TermExpr struct {
	ColumnInfo           *ColumnInfo     `protobuf:"bytes,1,opt,name=column_info,json=columnInfo,proto3" json:"column_info,omitempty"`
	Values               []*GenericValue `protobuf:"bytes,2,rep,name=values,proto3" json:"values,omitempty"`
	XXX_NoUnkeyedLiteral struct{}        `json:"-"`
	XXX_unrecognized     []byte          `json:"-"`
	XXX_sizecache        int32           `json:"-"`
}

func (m *TermExpr) Reset()         { *m = TermExpr{} }
func (m *TermExpr) String() string { return proto.CompactTextString(m) }
func (*TermExpr) ProtoMessage()    {}
func (*TermExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{4}
}

func (m *TermExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TermExpr.Unmarshal(m, b)
}
func (m *TermExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TermExpr.Marshal(b, m, deterministic)
}
func (m *TermExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TermExpr.Merge(m, src)
}
func (m *TermExpr) XXX_Size() int {
	return xxx_messageInfo_TermExpr.Size(m)
}
func (m *TermExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_TermExpr.DiscardUnknown(m)
}

var xxx_messageInfo_TermExpr proto.InternalMessageInfo

func (m *TermExpr) GetColumnInfo() *ColumnInfo {
	if m != nil {
		return m.ColumnInfo
	}
	return nil
}

func (m *TermExpr) GetValues() []*GenericValue {
	if m != nil {
		return m.Values
	}
	return nil
}

type UnaryExpr struct {
	Op                   UnaryExpr_UnaryOp `protobuf:"varint,1,opt,name=op,proto3,enum=milvus.proto.plan.UnaryExpr_UnaryOp" json:"op,omitempty"`
	Child                *Expr             `protobuf:"bytes,2,opt,name=child,proto3" json:"child,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *UnaryExpr) Reset()         { *m = UnaryExpr{} }
func (m *UnaryExpr) String() string { return proto.CompactTextString(m) }
func (*UnaryExpr) ProtoMessage()    {}
func (*UnaryExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{5}
}

func (m *UnaryExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_UnaryExpr.Unmarshal(m, b)
}
func (m *UnaryExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_UnaryExpr.Marshal(b, m, deterministic)
}
func (m *UnaryExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_UnaryExpr.Merge(m, src)
}
func (m *UnaryExpr) XXX_Size() int {
	return xxx_messageInfo_UnaryExpr.Size(m)
}
func (m *UnaryExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_UnaryExpr.DiscardUnknown(m)
}

var xxx_messageInfo_UnaryExpr proto.InternalMessageInfo

func (m *UnaryExpr) GetOp() UnaryExpr_UnaryOp {
	if m != nil {
		return m.Op
	}
	return UnaryExpr_Invalid
}

func (m *UnaryExpr) GetChild() *Expr {
	if m != nil {
		return m.Child
	}
	return nil
}

type BinaryExpr struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *BinaryExpr) Reset()         { *m = BinaryExpr{} }
func (m *BinaryExpr) String() string { return proto.CompactTextString(m) }
func (*BinaryExpr) ProtoMessage()    {}
func (*BinaryExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{6}
}

func (m *BinaryExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BinaryExpr.Unmarshal(m, b)
}
func (m *BinaryExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BinaryExpr.Marshal(b, m, deterministic)
}
func (m *BinaryExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BinaryExpr.Merge(m, src)
}
func (m *BinaryExpr) XXX_Size() int {
	return xxx_messageInfo_BinaryExpr.Size(m)
}
func (m *BinaryExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_BinaryExpr.DiscardUnknown(m)
}

var xxx_messageInfo_BinaryExpr proto.InternalMessageInfo

type Expr struct {
	// Types that are valid to be assigned to Expr:
	//	*Expr_RangeExpr
	//	*Expr_TermExpr
	//	*Expr_UnaryExpr
	//	*Expr_BinaryExpr
	Expr                 isExpr_Expr `protobuf_oneof:"expr"`
	XXX_NoUnkeyedLiteral struct{}    `json:"-"`
	XXX_unrecognized     []byte      `json:"-"`
	XXX_sizecache        int32       `json:"-"`
}

func (m *Expr) Reset()         { *m = Expr{} }
func (m *Expr) String() string { return proto.CompactTextString(m) }
func (*Expr) ProtoMessage()    {}
func (*Expr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{7}
}

func (m *Expr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Expr.Unmarshal(m, b)
}
func (m *Expr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Expr.Marshal(b, m, deterministic)
}
func (m *Expr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Expr.Merge(m, src)
}
func (m *Expr) XXX_Size() int {
	return xxx_messageInfo_Expr.Size(m)
}
func (m *Expr) XXX_DiscardUnknown() {
	xxx_messageInfo_Expr.DiscardUnknown(m)
}

var xxx_messageInfo_Expr proto.InternalMessageInfo

type isExpr_Expr interface {
	isExpr_Expr()
}

type Expr_RangeExpr struct {
	RangeExpr *RangeExpr `protobuf:"bytes,1,opt,name=range_expr,json=rangeExpr,proto3,oneof"`
}

type Expr_TermExpr struct {
	TermExpr *TermExpr `protobuf:"bytes,2,opt,name=term_expr,json=termExpr,proto3,oneof"`
}

type Expr_UnaryExpr struct {
	UnaryExpr *UnaryExpr `protobuf:"bytes,3,opt,name=unary_expr,json=unaryExpr,proto3,oneof"`
}

type Expr_BinaryExpr struct {
	BinaryExpr *BinaryExpr `protobuf:"bytes,4,opt,name=binary_expr,json=binaryExpr,proto3,oneof"`
}

func (*Expr_RangeExpr) isExpr_Expr() {}

func (*Expr_TermExpr) isExpr_Expr() {}

func (*Expr_UnaryExpr) isExpr_Expr() {}

func (*Expr_BinaryExpr) isExpr_Expr() {}

func (m *Expr) GetExpr() isExpr_Expr {
	if m != nil {
		return m.Expr
	}
	return nil
}

func (m *Expr) GetRangeExpr() *RangeExpr {
	if x, ok := m.GetExpr().(*Expr_RangeExpr); ok {
		return x.RangeExpr
	}
	return nil
}

func (m *Expr) GetTermExpr() *TermExpr {
	if x, ok := m.GetExpr().(*Expr_TermExpr); ok {
		return x.TermExpr
	}
	return nil
}

func (m *Expr) GetUnaryExpr() *UnaryExpr {
	if x, ok := m.GetExpr().(*Expr_UnaryExpr); ok {
		return x.UnaryExpr
	}
	return nil
}

func (m *Expr) GetBinaryExpr() *BinaryExpr {
	if x, ok := m.GetExpr().(*Expr_BinaryExpr); ok {
		return x.BinaryExpr
	}
	return nil
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*Expr) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*Expr_RangeExpr)(nil),
		(*Expr_TermExpr)(nil),
		(*Expr_UnaryExpr)(nil),
		(*Expr_BinaryExpr)(nil),
	}
}

type VectorANNS struct {
	IsBinary             bool       `protobuf:"varint,1,opt,name=is_binary,json=isBinary,proto3" json:"is_binary,omitempty"`
	FieldId              int64      `protobuf:"varint,2,opt,name=field_id,json=fieldId,proto3" json:"field_id,omitempty"`
	Predicates           *Expr      `protobuf:"bytes,3,opt,name=predicates,proto3" json:"predicates,omitempty"`
	QueryInfo            *QueryInfo `protobuf:"bytes,4,opt,name=query_info,json=queryInfo,proto3" json:"query_info,omitempty"`
	PlaceholderTag       string     `protobuf:"bytes,5,opt,name=placeholder_tag,json=placeholderTag,proto3" json:"placeholder_tag,omitempty"`
	XXX_NoUnkeyedLiteral struct{}   `json:"-"`
	XXX_unrecognized     []byte     `json:"-"`
	XXX_sizecache        int32      `json:"-"`
}

func (m *VectorANNS) Reset()         { *m = VectorANNS{} }
func (m *VectorANNS) String() string { return proto.CompactTextString(m) }
func (*VectorANNS) ProtoMessage()    {}
func (*VectorANNS) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{8}
}

func (m *VectorANNS) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_VectorANNS.Unmarshal(m, b)
}
func (m *VectorANNS) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_VectorANNS.Marshal(b, m, deterministic)
}
func (m *VectorANNS) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VectorANNS.Merge(m, src)
}
func (m *VectorANNS) XXX_Size() int {
	return xxx_messageInfo_VectorANNS.Size(m)
}
func (m *VectorANNS) XXX_DiscardUnknown() {
	xxx_messageInfo_VectorANNS.DiscardUnknown(m)
}

var xxx_messageInfo_VectorANNS proto.InternalMessageInfo

func (m *VectorANNS) GetIsBinary() bool {
	if m != nil {
		return m.IsBinary
	}
	return false
}

func (m *VectorANNS) GetFieldId() int64 {
	if m != nil {
		return m.FieldId
	}
	return 0
}

func (m *VectorANNS) GetPredicates() *Expr {
	if m != nil {
		return m.Predicates
	}
	return nil
}

func (m *VectorANNS) GetQueryInfo() *QueryInfo {
	if m != nil {
		return m.QueryInfo
	}
	return nil
}

func (m *VectorANNS) GetPlaceholderTag() string {
	if m != nil {
		return m.PlaceholderTag
	}
	return ""
}

type PlanNode struct {
	// Types that are valid to be assigned to Node:
	//	*PlanNode_VectorAnns
	Node                 isPlanNode_Node `protobuf_oneof:"node"`
	XXX_NoUnkeyedLiteral struct{}        `json:"-"`
	XXX_unrecognized     []byte          `json:"-"`
	XXX_sizecache        int32           `json:"-"`
}

func (m *PlanNode) Reset()         { *m = PlanNode{} }
func (m *PlanNode) String() string { return proto.CompactTextString(m) }
func (*PlanNode) ProtoMessage()    {}
func (*PlanNode) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{9}
}

func (m *PlanNode) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_PlanNode.Unmarshal(m, b)
}
func (m *PlanNode) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_PlanNode.Marshal(b, m, deterministic)
}
func (m *PlanNode) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PlanNode.Merge(m, src)
}
func (m *PlanNode) XXX_Size() int {
	return xxx_messageInfo_PlanNode.Size(m)
}
func (m *PlanNode) XXX_DiscardUnknown() {
	xxx_messageInfo_PlanNode.DiscardUnknown(m)
}

var xxx_messageInfo_PlanNode proto.InternalMessageInfo

type isPlanNode_Node interface {
	isPlanNode_Node()
}

type PlanNode_VectorAnns struct {
	VectorAnns *VectorANNS `protobuf:"bytes,1,opt,name=vector_anns,json=vectorAnns,proto3,oneof"`
}

func (*PlanNode_VectorAnns) isPlanNode_Node() {}

func (m *PlanNode) GetNode() isPlanNode_Node {
	if m != nil {
		return m.Node
	}
	return nil
}

func (m *PlanNode) GetVectorAnns() *VectorANNS {
	if x, ok := m.GetNode().(*PlanNode_VectorAnns); ok {
		return x.VectorAnns
	}
	return nil
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*PlanNode) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*PlanNode_VectorAnns)(nil),
	}
}

func init() {
	proto.RegisterEnum("milvus.proto.plan.RangeExpr_OpType", RangeExpr_OpType_name, RangeExpr_OpType_value)
	proto.RegisterEnum("milvus.proto.plan.UnaryExpr_UnaryOp", UnaryExpr_UnaryOp_name, UnaryExpr_UnaryOp_value)
	proto.RegisterEnum("milvus.proto.plan.BinaryExpr_BinaryOp", BinaryExpr_BinaryOp_name, BinaryExpr_BinaryOp_value)
	proto.RegisterType((*GenericValue)(nil), "milvus.proto.plan.GenericValue")
	proto.RegisterType((*QueryInfo)(nil), "milvus.proto.plan.QueryInfo")
	proto.RegisterType((*ColumnInfo)(nil), "milvus.proto.plan.ColumnInfo")
	proto.RegisterType((*RangeExpr)(nil), "milvus.proto.plan.RangeExpr")
	proto.RegisterType((*TermExpr)(nil), "milvus.proto.plan.TermExpr")
	proto.RegisterType((*UnaryExpr)(nil), "milvus.proto.plan.UnaryExpr")
	proto.RegisterType((*BinaryExpr)(nil), "milvus.proto.plan.BinaryExpr")
	proto.RegisterType((*Expr)(nil), "milvus.proto.plan.Expr")
	proto.RegisterType((*VectorANNS)(nil), "milvus.proto.plan.VectorANNS")
	proto.RegisterType((*PlanNode)(nil), "milvus.proto.plan.PlanNode")
}

func init() { proto.RegisterFile("plan.proto", fileDescriptor_2d655ab2f7683c23) }

var fileDescriptor_2d655ab2f7683c23 = []byte{
	// 805 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xac, 0x54, 0x4f, 0x6f, 0xdb, 0x36,
	0x14, 0xb7, 0x24, 0xff, 0x91, 0x9e, 0x5d, 0xd7, 0xe3, 0x65, 0xde, 0xb2, 0x22, 0x86, 0x3a, 0x60,
	0xbe, 0xd4, 0x06, 0xbc, 0xae, 0x05, 0x3a, 0xb4, 0x68, 0xb2, 0x16, 0x4d, 0x80, 0xc2, 0xe9, 0x34,
	0x2f, 0x87, 0x5d, 0x04, 0x5a, 0x62, 0x6c, 0x62, 0x34, 0xc9, 0x50, 0x94, 0xd1, 0xf4, 0xba, 0xdb,
	0x6e, 0xfb, 0x1c, 0xfb, 0x58, 0xfb, 0x22, 0x03, 0x49, 0x45, 0x4e, 0x0a, 0x27, 0xc0, 0x80, 0xdd,
	0xf8, 0xfe, 0xfc, 0xf8, 0xe3, 0xfb, 0xbd, 0xc7, 0x07, 0x20, 0x19, 0xe6, 0x13, 0xa9, 0x84, 0x16,
	0xe8, 0x8b, 0x0d, 0x65, 0xdb, 0xb2, 0x70, 0xd6, 0xc4, 0x04, 0xbe, 0xee, 0x15, 0xd9, 0x9a, 0x6c,
	0xb0, 0x73, 0xc5, 0x12, 0x7a, 0xef, 0x08, 0x27, 0x8a, 0x66, 0xe7, 0x98, 0x95, 0x04, 0x1d, 0x40,
	0xb8, 0x14, 0x82, 0xa5, 0x5b, 0xcc, 0x86, 0xde, 0xc8, 0x1b, 0x87, 0x27, 0x8d, 0xa4, 0x63, 0x3c,
	0xe7, 0x98, 0xa1, 0x47, 0x10, 0x51, 0xae, 0x9f, 0x3d, 0xb5, 0x51, 0x7f, 0xe4, 0x8d, 0x83, 0x93,
	0x46, 0x12, 0x5a, 0x57, 0x15, 0xbe, 0x60, 0x02, 0x6b, 0x1b, 0x0e, 0x46, 0xde, 0xd8, 0x33, 0x61,
	0xeb, 0x3a, 0xc7, 0xec, 0xb8, 0x05, 0xc1, 0x16, 0xb3, 0x98, 0x40, 0xf4, 0x73, 0x49, 0xd4, 0xd5,
	0x29, 0xbf, 0x10, 0x08, 0x41, 0x53, 0x0b, 0xf9, 0xbb, 0xa5, 0x0a, 0x12, 0x7b, 0x46, 0x87, 0xd0,
	0xdd, 0x10, 0xad, 0x68, 0x96, 0xea, 0x2b, 0x49, 0xec, 0x45, 0x51, 0x02, 0xce, 0xb5, 0xb8, 0x92,
	0x04, 0x3d, 0x86, 0x07, 0x05, 0xc1, 0x2a, 0x5b, 0xa7, 0x12, 0x2b, 0xbc, 0x29, 0x86, 0x4d, 0x9b,
	0xd2, 0x73, 0xce, 0x0f, 0xd6, 0x17, 0x67, 0x00, 0x3f, 0x09, 0x56, 0x6e, 0xb8, 0xe5, 0xf9, 0x0a,
	0xc2, 0x0b, 0x4a, 0x58, 0x9e, 0xd2, 0xbc, 0xe2, 0xea, 0x58, 0xfb, 0x34, 0x47, 0x2f, 0x20, 0xca,
	0xb1, 0xc6, 0x8e, 0xcc, 0x14, 0xd5, 0x9f, 0x3d, 0x9a, 0xdc, 0x92, 0xad, 0x12, 0xec, 0x0d, 0xd6,
	0xd8, 0xf0, 0x27, 0x61, 0x5e, 0x9d, 0xe2, 0xbf, 0x7d, 0x88, 0x12, 0xcc, 0x57, 0xe4, 0xed, 0x47,
	0xa9, 0xd0, 0x2b, 0xe8, 0x66, 0x96, 0x32, 0xa5, 0xfc, 0x42, 0x58, 0x9e, 0xee, 0xe7, 0x77, 0xd9,
	0xde, 0xec, 0x1e, 0x96, 0x40, 0xb6, 0x7b, 0xe4, 0x0f, 0x10, 0x08, 0x59, 0x0c, 0xfd, 0x51, 0x30,
	0xee, 0xcf, 0x1e, 0xef, 0xc1, 0xd5, 0x54, 0x93, 0x33, 0x69, 0x5f, 0x62, 0xf2, 0xd1, 0x73, 0x68,
	0x6f, 0x4d, 0xef, 0x8a, 0x61, 0x30, 0x0a, 0xc6, 0xdd, 0xd9, 0xe1, 0x1e, 0xe4, 0xcd, 0x1e, 0x27,
	0x55, 0x7a, 0xcc, 0xa1, 0xed, 0xee, 0x41, 0x5d, 0xe8, 0x9c, 0xf2, 0x2d, 0x66, 0x34, 0x1f, 0x34,
	0xd0, 0x43, 0xe8, 0xbe, 0x53, 0x04, 0x6b, 0xa2, 0x16, 0x6b, 0xcc, 0x07, 0x1e, 0x1a, 0x40, 0xaf,
	0x72, 0xbc, 0xbd, 0x2c, 0x31, 0x1b, 0xf8, 0xa8, 0x07, 0xe1, 0x7b, 0x52, 0x14, 0x36, 0x1e, 0xa0,
	0x07, 0x10, 0x19, 0xcb, 0x05, 0x9b, 0x28, 0x82, 0x96, 0x3b, 0xb6, 0x4c, 0xde, 0x5c, 0x68, 0x67,
	0xb5, 0xe3, 0x3f, 0x3c, 0x08, 0x17, 0x44, 0x6d, 0xfe, 0x17, 0xb1, 0x76, 0x55, 0xfb, 0xff, 0xad,
	0xea, 0xbf, 0x3c, 0x88, 0x7e, 0xe5, 0x58, 0x5d, 0xd9, 0x67, 0x3c, 0x05, 0x5f, 0x48, 0xcb, 0xde,
	0x9f, 0x7d, 0xbb, 0xe7, 0x8a, 0x3a, 0xd3, 0x9d, 0xce, 0x64, 0xe2, 0x0b, 0x89, 0x9e, 0x40, 0x2b,
	0x5b, 0x53, 0x96, 0xdb, 0x79, 0xe9, 0xce, 0xbe, 0xdc, 0x03, 0x34, 0x98, 0xc4, 0x65, 0xc5, 0x87,
	0xd0, 0xa9, 0xd0, 0xb7, 0x95, 0xee, 0x40, 0x30, 0x17, 0x7a, 0xe0, 0xc5, 0x6f, 0x00, 0x8e, 0xe9,
	0x35, 0x53, 0xfc, 0x0c, 0x42, 0x67, 0x7d, 0x9e, 0xdf, 0x07, 0x78, 0x2f, 0x56, 0x34, 0xc3, 0xec,
	0x88, 0xe7, 0x03, 0xcf, 0x0a, 0xef, 0xec, 0x33, 0x35, 0xf0, 0xe3, 0x3f, 0x7d, 0x68, 0xda, 0xa2,
	0x5e, 0x02, 0x28, 0x33, 0x2a, 0x29, 0xf9, 0x28, 0x55, 0x25, 0xed, 0x37, 0xf7, 0xcd, 0xd3, 0x49,
	0x23, 0x89, 0x54, 0x3d, 0xc7, 0x2f, 0x20, 0xd2, 0x44, 0x6d, 0x1c, 0xda, 0x55, 0x78, 0xb0, 0x07,
	0x7d, 0xdd, 0x4a, 0xf3, 0xc9, 0xf5, 0x75, 0x5b, 0x5f, 0x02, 0x94, 0xe6, 0xe9, 0x0e, 0x1c, 0xdc,
	0x49, 0x5d, 0xeb, 0x6a, 0xa8, 0xcb, 0xba, 0x1d, 0xaf, 0xa1, 0xbb, 0xa4, 0x3b, 0x7c, 0xf3, 0xce,
	0xa9, 0xd8, 0xc9, 0x75, 0xd2, 0x48, 0x60, 0x59, 0x5b, 0xc7, 0x6d, 0x68, 0x1a, 0x68, 0xfc, 0x8f,
	0x07, 0x70, 0x4e, 0x32, 0x2d, 0xd4, 0xd1, 0x7c, 0xfe, 0x0b, 0x3a, 0x80, 0x88, 0x16, 0xa9, 0xcb,
	0x73, 0x8b, 0x2d, 0x09, 0x69, 0xe1, 0x6e, 0xb9, 0xb5, 0x1d, 0xfc, 0xdb, 0xdb, 0xe1, 0x39, 0x80,
	0x54, 0x24, 0xa7, 0x19, 0xd6, 0xf6, 0x83, 0xdd, 0xdb, 0xee, 0x1b, 0xa9, 0xe8, 0x47, 0x80, 0x4b,
	0xb3, 0xe6, 0xdc, 0x78, 0x37, 0xef, 0x14, 0xa2, 0xde, 0x85, 0x49, 0x74, 0x59, 0xaf, 0xc5, 0xef,
	0xe0, 0xa1, 0x64, 0x38, 0x23, 0x6b, 0xc1, 0x72, 0xa2, 0x52, 0x8d, 0x57, 0xc3, 0x96, 0xdd, 0x71,
	0xfd, 0x1b, 0xee, 0x05, 0x5e, 0xc5, 0x0b, 0x08, 0x3f, 0x30, 0xcc, 0xe7, 0x22, 0x27, 0x46, 0xbb,
	0xad, 0x2d, 0x38, 0xc5, 0x9c, 0x17, 0xf7, 0xfc, 0xa8, 0x9d, 0x2c, 0x46, 0x3b, 0x87, 0x39, 0xe2,
	0xbc, 0x30, 0xda, 0x71, 0x91, 0x93, 0xe3, 0xd7, 0xbf, 0xbd, 0x5a, 0x51, 0xbd, 0x2e, 0x97, 0x93,
	0x4c, 0x6c, 0xa6, 0x9f, 0x28, 0x63, 0xf4, 0x93, 0x26, 0xd9, 0x7a, 0xea, 0xee, 0x7a, 0x92, 0xd3,
	0x42, 0x2b, 0xba, 0x2c, 0x35, 0xc9, 0xa7, 0x94, 0x6b, 0xa2, 0x38, 0x66, 0x53, 0x4b, 0x30, 0x35,
	0x04, 0x72, 0xb9, 0x6c, 0x5b, 0xeb, 0xfb, 0x7f, 0x03, 0x00, 0x00, 0xff, 0xff, 0x4d, 0x20, 0xbd,
	0x88, 0x8c, 0x06, 0x00, 0x00,
}
