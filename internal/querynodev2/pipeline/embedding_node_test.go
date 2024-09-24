// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pipeline

import (
	"testing"

	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/mq/msgstream"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

// test of embedding node
type EmbeddingNodeSuite struct {
	suite.Suite
	// datas
	collectionID     int64
	collectionSchema *schemapb.CollectionSchema
	channel          string
	msgs             []*InsertMsg

	// mocks
	manager    *segments.Manager
	segManager *segments.MockSegmentManager
	colManager *segments.MockCollectionManager
}

func (suite *EmbeddingNodeSuite) SetupSuite() {
	paramtable.Init()
	suite.collectionID = 111
	suite.channel = "test-channel"
	suite.collectionSchema = &schemapb.CollectionSchema{
		Name: "test-collection",
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:  common.TimeStampField,
				Name:     common.TimeStampFieldName,
				DataType: schemapb.DataType_Int64,
			}, {
				Name:         "pk",
				FieldID:      100,
				IsPrimaryKey: true,
				DataType:     schemapb.DataType_Int64,
			}, {
				Name:       "text",
				FieldID:    101,
				DataType:   schemapb.DataType_VarChar,
				TypeParams: []*commonpb.KeyValuePair{},
			}, {
				Name:             "sparse",
				FieldID:          102,
				DataType:         schemapb.DataType_SparseFloatVector,
				IsFunctionOutput: true,
			},
		},
		Functions: []*schemapb.FunctionSchema{{
			Name:           "BM25",
			Type:           schemapb.FunctionType_BM25,
			InputFieldIds:  []int64{101},
			OutputFieldIds: []int64{102},
		}},
	}

	suite.msgs = []*msgstream.InsertMsg{{
		BaseMsg: msgstream.BaseMsg{},
		InsertRequest: &msgpb.InsertRequest{
			SegmentID:  1,
			NumRows:    3,
			Version:    msgpb.InsertDataVersion_ColumnBased,
			Timestamps: []uint64{1, 1, 1},
			FieldsData: []*schemapb.FieldData{
				{
					FieldId: 100,
					Type:    schemapb.DataType_Int64,
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
					},
				}, {
					FieldId: 101,
					Type:    schemapb.DataType_VarChar,
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: []string{"test1", "test2", "test3"}}}},
					},
				},
			},
		},
	}}

	suite.segManager = segments.NewMockSegmentManager(suite.T())
	suite.colManager = segments.NewMockCollectionManager(suite.T())

	suite.manager = &segments.Manager{
		Collection: suite.colManager,
		Segment:    suite.segManager,
	}
}

func (suite *EmbeddingNodeSuite) TestCreateEmbeddingNode() {
	suite.Run("collection not found", func() {
		suite.colManager.EXPECT().Get(suite.collectionID).Return(nil).Once()
		_, err := newEmbeddingNode(suite.collectionID, suite.channel, suite.manager, 128)
		suite.Error(err)
	})

	suite.Run("function invalid", func() {
		collSchema := proto.Clone(suite.collectionSchema).(*schemapb.CollectionSchema)
		collection := segments.NewCollectionWithoutSegcore(suite.collectionID, collSchema)
		collection.Schema().Functions = []*schemapb.FunctionSchema{{}}
		suite.colManager.EXPECT().Get(suite.collectionID).Return(collection).Once()
		_, err := newEmbeddingNode(suite.collectionID, suite.channel, suite.manager, 128)
		suite.Error(err)
	})

	suite.Run("normal case", func() {
		collSchema := proto.Clone(suite.collectionSchema).(*schemapb.CollectionSchema)
		collection := segments.NewCollectionWithoutSegcore(suite.collectionID, collSchema)
		suite.colManager.EXPECT().Get(suite.collectionID).Return(collection).Once()
		_, err := newEmbeddingNode(suite.collectionID, suite.channel, suite.manager, 128)
		suite.NoError(err)
	})
}

func (suite *EmbeddingNodeSuite) TestOperator() {
	suite.Run("collection not found", func() {
		suite.colManager.EXPECT().Get(suite.collectionID).Return(&segments.Collection{}).Once()
		node, err := newEmbeddingNode(suite.collectionID, suite.channel, suite.manager, 128)
		suite.NoError(err)

		suite.colManager.EXPECT().Get(suite.collectionID).Return(nil).Once()
		suite.Panics(func() {
			node.Operate(&insertNodeMsg{})
		})
	})

	suite.Run("add InsertData Failed", func() {
		collection := segments.NewCollectionWithoutSegcore(suite.collectionID, suite.collectionSchema)
		suite.colManager.EXPECT().Get(suite.collectionID).Return(collection).Times(2)
		node, err := newEmbeddingNode(suite.collectionID, suite.channel, suite.manager, 128)
		suite.NoError(err)

		suite.Panics(func() {
			node.Operate(&insertNodeMsg{
				insertMsgs: []*msgstream.InsertMsg{{
					BaseMsg: msgstream.BaseMsg{},
					InsertRequest: &msgpb.InsertRequest{
						SegmentID: 1,
						NumRows:   3,
						Version:   msgpb.InsertDataVersion_ColumnBased,
						FieldsData: []*schemapb.FieldData{
							{
								FieldId: 100,
								Type:    schemapb.DataType_Int64,
								Field: &schemapb.FieldData_Scalars{
									Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
								},
							},
						},
					},
				}},
			})
		})
	})

	suite.Run("normal case", func() {
		collection := segments.NewCollectionWithoutSegcore(suite.collectionID, suite.collectionSchema)
		suite.colManager.EXPECT().Get(suite.collectionID).Return(collection).Times(2)
		node, err := newEmbeddingNode(suite.collectionID, suite.channel, suite.manager, 128)
		suite.NoError(err)

		suite.NotPanics(func() {
			output := node.Operate(&insertNodeMsg{
				insertMsgs: suite.msgs,
			})

			msg, ok := output.(*insertNodeMsg)
			suite.Require().True(ok)
			suite.Require().NotNil(msg.insertDatas)
			suite.Require().Equal(int64(3), msg.insertDatas[1].BM25Stats[102].NumRow())
			suite.Require().Equal(int64(3), msg.insertDatas[1].InsertRecord.GetNumRows())
		})
	})
}

func TestEmbeddingNode(t *testing.T) {
	suite.Run(t, new(EmbeddingNodeSuite))
}
