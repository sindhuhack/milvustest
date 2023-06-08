// Code generated by protoc-gen-go. DO NOT EDIT.
// source: segcore.proto

package segcorepb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	commonpb "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	schemapb "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
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

type RetrieveResults struct {
	Ids                  *schemapb.IDs         `protobuf:"bytes,1,opt,name=ids,proto3" json:"ids,omitempty"`
	Offset               []int64               `protobuf:"varint,2,rep,packed,name=offset,proto3" json:"offset,omitempty"`
	FieldsData           []*schemapb.FieldData `protobuf:"bytes,3,rep,name=fields_data,json=fieldsData,proto3" json:"fields_data,omitempty"`
	XXX_NoUnkeyedLiteral struct{}              `json:"-"`
	XXX_unrecognized     []byte                `json:"-"`
	XXX_sizecache        int32                 `json:"-"`
}

func (m *RetrieveResults) Reset()         { *m = RetrieveResults{} }
func (m *RetrieveResults) String() string { return proto.CompactTextString(m) }
func (*RetrieveResults) ProtoMessage()    {}
func (*RetrieveResults) Descriptor() ([]byte, []int) {
	return fileDescriptor_1d79fce784797357, []int{0}
}

func (m *RetrieveResults) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RetrieveResults.Unmarshal(m, b)
}
func (m *RetrieveResults) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RetrieveResults.Marshal(b, m, deterministic)
}
func (m *RetrieveResults) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RetrieveResults.Merge(m, src)
}
func (m *RetrieveResults) XXX_Size() int {
	return xxx_messageInfo_RetrieveResults.Size(m)
}
func (m *RetrieveResults) XXX_DiscardUnknown() {
	xxx_messageInfo_RetrieveResults.DiscardUnknown(m)
}

var xxx_messageInfo_RetrieveResults proto.InternalMessageInfo

func (m *RetrieveResults) GetIds() *schemapb.IDs {
	if m != nil {
		return m.Ids
	}
	return nil
}

func (m *RetrieveResults) GetOffset() []int64 {
	if m != nil {
		return m.Offset
	}
	return nil
}

func (m *RetrieveResults) GetFieldsData() []*schemapb.FieldData {
	if m != nil {
		return m.FieldsData
	}
	return nil
}

type LoadFieldMeta struct {
	MinTimestamp         int64    `protobuf:"varint,1,opt,name=min_timestamp,json=minTimestamp,proto3" json:"min_timestamp,omitempty"`
	MaxTimestamp         int64    `protobuf:"varint,2,opt,name=max_timestamp,json=maxTimestamp,proto3" json:"max_timestamp,omitempty"`
	RowCount             int64    `protobuf:"varint,3,opt,name=row_count,json=rowCount,proto3" json:"row_count,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *LoadFieldMeta) Reset()         { *m = LoadFieldMeta{} }
func (m *LoadFieldMeta) String() string { return proto.CompactTextString(m) }
func (*LoadFieldMeta) ProtoMessage()    {}
func (*LoadFieldMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_1d79fce784797357, []int{1}
}

func (m *LoadFieldMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_LoadFieldMeta.Unmarshal(m, b)
}
func (m *LoadFieldMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_LoadFieldMeta.Marshal(b, m, deterministic)
}
func (m *LoadFieldMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LoadFieldMeta.Merge(m, src)
}
func (m *LoadFieldMeta) XXX_Size() int {
	return xxx_messageInfo_LoadFieldMeta.Size(m)
}
func (m *LoadFieldMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_LoadFieldMeta.DiscardUnknown(m)
}

var xxx_messageInfo_LoadFieldMeta proto.InternalMessageInfo

func (m *LoadFieldMeta) GetMinTimestamp() int64 {
	if m != nil {
		return m.MinTimestamp
	}
	return 0
}

func (m *LoadFieldMeta) GetMaxTimestamp() int64 {
	if m != nil {
		return m.MaxTimestamp
	}
	return 0
}

func (m *LoadFieldMeta) GetRowCount() int64 {
	if m != nil {
		return m.RowCount
	}
	return 0
}

type LoadSegmentMeta struct {
	// TODOs
	Metas                []*LoadFieldMeta `protobuf:"bytes,1,rep,name=metas,proto3" json:"metas,omitempty"`
	TotalSize            int64            `protobuf:"varint,2,opt,name=total_size,json=totalSize,proto3" json:"total_size,omitempty"`
	XXX_NoUnkeyedLiteral struct{}         `json:"-"`
	XXX_unrecognized     []byte           `json:"-"`
	XXX_sizecache        int32            `json:"-"`
}

func (m *LoadSegmentMeta) Reset()         { *m = LoadSegmentMeta{} }
func (m *LoadSegmentMeta) String() string { return proto.CompactTextString(m) }
func (*LoadSegmentMeta) ProtoMessage()    {}
func (*LoadSegmentMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_1d79fce784797357, []int{2}
}

func (m *LoadSegmentMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_LoadSegmentMeta.Unmarshal(m, b)
}
func (m *LoadSegmentMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_LoadSegmentMeta.Marshal(b, m, deterministic)
}
func (m *LoadSegmentMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LoadSegmentMeta.Merge(m, src)
}
func (m *LoadSegmentMeta) XXX_Size() int {
	return xxx_messageInfo_LoadSegmentMeta.Size(m)
}
func (m *LoadSegmentMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_LoadSegmentMeta.DiscardUnknown(m)
}

var xxx_messageInfo_LoadSegmentMeta proto.InternalMessageInfo

func (m *LoadSegmentMeta) GetMetas() []*LoadFieldMeta {
	if m != nil {
		return m.Metas
	}
	return nil
}

func (m *LoadSegmentMeta) GetTotalSize() int64 {
	if m != nil {
		return m.TotalSize
	}
	return 0
}

type InsertRecord struct {
	FieldsData           []*schemapb.FieldData `protobuf:"bytes,1,rep,name=fields_data,json=fieldsData,proto3" json:"fields_data,omitempty"`
	NumRows              int64                 `protobuf:"varint,2,opt,name=num_rows,json=numRows,proto3" json:"num_rows,omitempty"`
	XXX_NoUnkeyedLiteral struct{}              `json:"-"`
	XXX_unrecognized     []byte                `json:"-"`
	XXX_sizecache        int32                 `json:"-"`
}

func (m *InsertRecord) Reset()         { *m = InsertRecord{} }
func (m *InsertRecord) String() string { return proto.CompactTextString(m) }
func (*InsertRecord) ProtoMessage()    {}
func (*InsertRecord) Descriptor() ([]byte, []int) {
	return fileDescriptor_1d79fce784797357, []int{3}
}

func (m *InsertRecord) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_InsertRecord.Unmarshal(m, b)
}
func (m *InsertRecord) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_InsertRecord.Marshal(b, m, deterministic)
}
func (m *InsertRecord) XXX_Merge(src proto.Message) {
	xxx_messageInfo_InsertRecord.Merge(m, src)
}
func (m *InsertRecord) XXX_Size() int {
	return xxx_messageInfo_InsertRecord.Size(m)
}
func (m *InsertRecord) XXX_DiscardUnknown() {
	xxx_messageInfo_InsertRecord.DiscardUnknown(m)
}

var xxx_messageInfo_InsertRecord proto.InternalMessageInfo

func (m *InsertRecord) GetFieldsData() []*schemapb.FieldData {
	if m != nil {
		return m.FieldsData
	}
	return nil
}

func (m *InsertRecord) GetNumRows() int64 {
	if m != nil {
		return m.NumRows
	}
	return 0
}

type FieldIndexMeta struct {
	FieldID              int64                    `protobuf:"varint,1,opt,name=fieldID,proto3" json:"fieldID,omitempty"`
	CollectionID         int64                    `protobuf:"varint,2,opt,name=collectionID,proto3" json:"collectionID,omitempty"`
	IndexName            string                   `protobuf:"bytes,3,opt,name=index_name,json=indexName,proto3" json:"index_name,omitempty"`
	TypeParams           []*commonpb.KeyValuePair `protobuf:"bytes,4,rep,name=type_params,json=typeParams,proto3" json:"type_params,omitempty"`
	IndexParams          []*commonpb.KeyValuePair `protobuf:"bytes,5,rep,name=index_params,json=indexParams,proto3" json:"index_params,omitempty"`
	IsAutoIndex          bool                     `protobuf:"varint,6,opt,name=is_auto_index,json=isAutoIndex,proto3" json:"is_auto_index,omitempty"`
	UserIndexParams      []*commonpb.KeyValuePair `protobuf:"bytes,7,rep,name=user_index_params,json=userIndexParams,proto3" json:"user_index_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                 `json:"-"`
	XXX_unrecognized     []byte                   `json:"-"`
	XXX_sizecache        int32                    `json:"-"`
}

func (m *FieldIndexMeta) Reset()         { *m = FieldIndexMeta{} }
func (m *FieldIndexMeta) String() string { return proto.CompactTextString(m) }
func (*FieldIndexMeta) ProtoMessage()    {}
func (*FieldIndexMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_1d79fce784797357, []int{4}
}

func (m *FieldIndexMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FieldIndexMeta.Unmarshal(m, b)
}
func (m *FieldIndexMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FieldIndexMeta.Marshal(b, m, deterministic)
}
func (m *FieldIndexMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FieldIndexMeta.Merge(m, src)
}
func (m *FieldIndexMeta) XXX_Size() int {
	return xxx_messageInfo_FieldIndexMeta.Size(m)
}
func (m *FieldIndexMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_FieldIndexMeta.DiscardUnknown(m)
}

var xxx_messageInfo_FieldIndexMeta proto.InternalMessageInfo

func (m *FieldIndexMeta) GetFieldID() int64 {
	if m != nil {
		return m.FieldID
	}
	return 0
}

func (m *FieldIndexMeta) GetCollectionID() int64 {
	if m != nil {
		return m.CollectionID
	}
	return 0
}

func (m *FieldIndexMeta) GetIndexName() string {
	if m != nil {
		return m.IndexName
	}
	return ""
}

func (m *FieldIndexMeta) GetTypeParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.TypeParams
	}
	return nil
}

func (m *FieldIndexMeta) GetIndexParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.IndexParams
	}
	return nil
}

func (m *FieldIndexMeta) GetIsAutoIndex() bool {
	if m != nil {
		return m.IsAutoIndex
	}
	return false
}

func (m *FieldIndexMeta) GetUserIndexParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.UserIndexParams
	}
	return nil
}

type CollectionIndexMeta struct {
	MaxIndexRowCount     int64             `protobuf:"varint,1,opt,name=maxIndexRowCount,proto3" json:"maxIndexRowCount,omitempty"`
	IndexMetas           []*FieldIndexMeta `protobuf:"bytes,2,rep,name=index_metas,json=indexMetas,proto3" json:"index_metas,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *CollectionIndexMeta) Reset()         { *m = CollectionIndexMeta{} }
func (m *CollectionIndexMeta) String() string { return proto.CompactTextString(m) }
func (*CollectionIndexMeta) ProtoMessage()    {}
func (*CollectionIndexMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_1d79fce784797357, []int{5}
}

func (m *CollectionIndexMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_CollectionIndexMeta.Unmarshal(m, b)
}
func (m *CollectionIndexMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_CollectionIndexMeta.Marshal(b, m, deterministic)
}
func (m *CollectionIndexMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CollectionIndexMeta.Merge(m, src)
}
func (m *CollectionIndexMeta) XXX_Size() int {
	return xxx_messageInfo_CollectionIndexMeta.Size(m)
}
func (m *CollectionIndexMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_CollectionIndexMeta.DiscardUnknown(m)
}

var xxx_messageInfo_CollectionIndexMeta proto.InternalMessageInfo

func (m *CollectionIndexMeta) GetMaxIndexRowCount() int64 {
	if m != nil {
		return m.MaxIndexRowCount
	}
	return 0
}

func (m *CollectionIndexMeta) GetIndexMetas() []*FieldIndexMeta {
	if m != nil {
		return m.IndexMetas
	}
	return nil
}

func init() {
	proto.RegisterType((*RetrieveResults)(nil), "milvus.proto.segcore.RetrieveResults")
	proto.RegisterType((*LoadFieldMeta)(nil), "milvus.proto.segcore.LoadFieldMeta")
	proto.RegisterType((*LoadSegmentMeta)(nil), "milvus.proto.segcore.LoadSegmentMeta")
	proto.RegisterType((*InsertRecord)(nil), "milvus.proto.segcore.InsertRecord")
	proto.RegisterType((*FieldIndexMeta)(nil), "milvus.proto.segcore.FieldIndexMeta")
	proto.RegisterType((*CollectionIndexMeta)(nil), "milvus.proto.segcore.CollectionIndexMeta")
}

func init() { proto.RegisterFile("segcore.proto", fileDescriptor_1d79fce784797357) }

var fileDescriptor_1d79fce784797357 = []byte{
	// 564 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x52, 0x5d, 0x6f, 0xd3, 0x30,
	0x14, 0x55, 0x1b, 0xf6, 0xd1, 0x9b, 0x94, 0x81, 0x41, 0x28, 0x0c, 0x81, 0x4a, 0xc6, 0x43, 0x35,
	0x89, 0x56, 0x1a, 0x08, 0x89, 0x27, 0xc4, 0x56, 0x90, 0x22, 0x18, 0x9a, 0x3c, 0xc4, 0x03, 0x2f,
	0x91, 0x97, 0xdc, 0x6d, 0x86, 0xd8, 0xae, 0x6c, 0x67, 0xed, 0xf6, 0x0b, 0xf8, 0x05, 0xfc, 0x48,
	0x7e, 0x05, 0xb2, 0xe3, 0xb1, 0x15, 0xfa, 0x30, 0xde, 0x7c, 0x4f, 0xee, 0x39, 0xe7, 0xde, 0x93,
	0x0b, 0x7d, 0x83, 0x27, 0xa5, 0xd2, 0x38, 0x9a, 0x6a, 0x65, 0x15, 0xb9, 0x2f, 0x78, 0x7d, 0xd6,
	0x98, 0xb6, 0x1a, 0x85, 0x6f, 0x9b, 0x89, 0x29, 0x4f, 0x51, 0xb0, 0x16, 0xdd, 0x4c, 0x4a, 0x25,
	0x84, 0x92, 0x6d, 0x95, 0xfd, 0xec, 0xc0, 0x06, 0x45, 0xab, 0x39, 0x9e, 0x21, 0x45, 0xd3, 0xd4,
	0xd6, 0x90, 0x6d, 0x88, 0x78, 0x65, 0xd2, 0xce, 0xa0, 0x33, 0x8c, 0x77, 0xd2, 0xd1, 0xa2, 0x66,
	0x2b, 0x95, 0x4f, 0x0c, 0x75, 0x4d, 0xe4, 0x01, 0xac, 0xaa, 0xe3, 0x63, 0x83, 0x36, 0xed, 0x0e,
	0xa2, 0x61, 0x44, 0x43, 0x45, 0xde, 0x40, 0x7c, 0xcc, 0xb1, 0xae, 0x4c, 0x51, 0x31, 0xcb, 0xd2,
	0x68, 0x10, 0x0d, 0xe3, 0x9d, 0x27, 0x4b, 0xb5, 0xde, 0xbb, 0xbe, 0x09, 0xb3, 0x8c, 0x42, 0x4b,
	0x71, 0xef, 0xec, 0x0c, 0xfa, 0x1f, 0x15, 0xab, 0xfc, 0xc7, 0x7d, 0xb4, 0x8c, 0x6c, 0x41, 0x5f,
	0x70, 0x59, 0x58, 0x2e, 0xd0, 0x58, 0x26, 0xa6, 0x7e, 0xbe, 0x88, 0x26, 0x82, 0xcb, 0xcf, 0x97,
	0x98, 0x6f, 0x62, 0xf3, 0x6b, 0x4d, 0xdd, 0xd0, 0xc4, 0xe6, 0x57, 0x4d, 0x8f, 0xa0, 0xa7, 0xd5,
	0xac, 0x28, 0x55, 0x23, 0x6d, 0x1a, 0xf9, 0x86, 0x75, 0xad, 0x66, 0x7b, 0xae, 0xce, 0xbe, 0xc3,
	0x86, 0xf3, 0x3d, 0xc4, 0x13, 0x81, 0xd2, 0x7a, 0xe7, 0xd7, 0xb0, 0x22, 0xd0, 0x32, 0x97, 0x88,
	0xdb, 0x62, 0x6b, 0xb4, 0x2c, 0xe5, 0xd1, 0xc2, 0xb4, 0xb4, 0x65, 0x90, 0xc7, 0x00, 0x56, 0x59,
	0x56, 0x17, 0x86, 0x5f, 0x60, 0x18, 0xa6, 0xe7, 0x91, 0x43, 0x7e, 0x81, 0xd9, 0x37, 0x48, 0x72,
	0x69, 0x50, 0x5b, 0x8a, 0xa5, 0xd2, 0xd5, 0xdf, 0xa9, 0x75, 0xfe, 0x37, 0x35, 0xf2, 0x10, 0xd6,
	0x65, 0x23, 0x0a, 0xad, 0x66, 0x26, 0xb8, 0xad, 0xc9, 0x46, 0x50, 0x35, 0x33, 0xd9, 0xaf, 0x2e,
	0xdc, 0xf6, 0xa4, 0x5c, 0x56, 0x38, 0xf7, 0x8b, 0xa5, 0xb0, 0xe6, 0xb9, 0xf9, 0x24, 0x84, 0x79,
	0x59, 0x92, 0x0c, 0x92, 0x52, 0xd5, 0x35, 0x96, 0x96, 0x2b, 0x99, 0x4f, 0x2e, 0x63, 0xbc, 0x8e,
	0xb9, 0xdd, 0xb8, 0x93, 0x2a, 0x24, 0x13, 0xe8, 0x73, 0xec, 0xd1, 0x9e, 0x47, 0x3e, 0x31, 0x81,
	0x64, 0x17, 0x62, 0x7b, 0x3e, 0xc5, 0x62, 0xca, 0x34, 0x13, 0x26, 0xbd, 0xe5, 0x77, 0x79, 0xba,
	0xb8, 0x4b, 0x38, 0xc5, 0x0f, 0x78, 0xfe, 0x85, 0xd5, 0x0d, 0x1e, 0x30, 0xae, 0x29, 0x38, 0xd6,
	0x81, 0x27, 0x91, 0x09, 0x24, 0xad, 0x45, 0x10, 0x59, 0xb9, 0xa9, 0x48, 0xec, 0x69, 0x41, 0x25,
	0x83, 0x3e, 0x37, 0x05, 0x6b, 0xac, 0x2a, 0x3c, 0x9c, 0xae, 0x0e, 0x3a, 0xc3, 0x75, 0x1a, 0x73,
	0xf3, 0xb6, 0xb1, 0xca, 0xc7, 0x41, 0xf6, 0xe1, 0x6e, 0x63, 0x50, 0x17, 0x0b, 0x76, 0x6b, 0x37,
	0xb5, 0xdb, 0x70, 0xdc, 0xfc, 0xca, 0x32, 0xfb, 0xd1, 0x81, 0x7b, 0x7b, 0x57, 0x61, 0xfd, 0x49,
	0x7c, 0x1b, 0xee, 0x08, 0x36, 0xf7, 0x35, 0x0d, 0x17, 0x17, 0xa2, 0xff, 0x07, 0x27, 0xef, 0xa0,
	0xdd, 0xa2, 0x68, 0x8f, 0xaf, 0xeb, 0x87, 0x79, 0xb6, 0xfc, 0xf8, 0x16, 0x7f, 0x2c, 0x6d, 0x7f,
	0x8c, 0x7b, 0x9a, 0xdd, 0x57, 0x5f, 0x5f, 0x9e, 0x70, 0x7b, 0xda, 0x1c, 0xb9, 0xc9, 0xc7, 0x2d,
	0xfb, 0x39, 0x57, 0xe1, 0x35, 0xe6, 0xd2, 0xa2, 0x96, 0xac, 0x1e, 0x7b, 0xc1, 0x71, 0x10, 0x9c,
	0x1e, 0x1d, 0xad, 0x7a, 0xe0, 0xc5, 0xef, 0x00, 0x00, 0x00, 0xff, 0xff, 0xdf, 0x46, 0x25, 0x42,
	0x63, 0x04, 0x00, 0x00,
}
