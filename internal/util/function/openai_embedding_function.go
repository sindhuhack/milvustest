/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
 * # distributed with this work for additional information
 * # regarding copyright ownership. The ASF licenses this file
 * # to you under the Apache License, Version 2.0 (the
 * # "License"); you may not use this file except in compliance
 * # with the License. You may obtain a copy of the License at
 * #
 * #     http://www.apache.org/licenses/LICENSE-2.0
 * #
 * # Unless required by applicable law or agreed to in writing, software
 * # distributed under the License is distributed on an "AS IS" BASIS,
 * # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * # See the License for the specific language governing permissions and
 * # limitations under the License.
 */

package function

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/models"
	"github.com/milvus-io/milvus/pkg/util/typeutil"	
)


const (
	TextEmbeddingAda002  string = "text-embedding-ada-002"
	TextEmbedding3Small string = "text-embedding-3-small"
	TextEmbedding3Large string = "text-embedding-3-large"
)

const (
	maxBatch = 128
	timeoutSec = 60
	maxRowNum = 60 * maxBatch
)


type OpenAIEmbeddingRunner struct {
	base *FunctionBase
	fieldDim int64
	
	client *models.OpenAIEmbeddingClient
	modelName string
	embedDimParam int64
	user string
}

func createOpenAIEmbeddingClient() (*models.OpenAIEmbeddingClient, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("The apiKey configuration was not found in the environment variables")
	}

	url := os.Getenv("OPENAI_EMBEDDING_URL")
	if url == "" {
		url = "https://api.openai.com/v1/embeddings"
	}
	c := models.NewOpenAIEmbeddingClient(apiKey, url)
	return &c, nil
}


func NewOpenAIEmbeddingRunner(coll *schemapb.CollectionSchema, schema *schemapb.FunctionSchema) (*OpenAIEmbeddingRunner, error) {
	if len(schema.GetOutputFieldIds()) != 1 {
		return nil, fmt.Errorf("OpenAIEmbedding function should only have one output field, but now %d", len(schema.GetOutputFieldIds()))
	}

	base, err := NewBase(coll, schema)
	if err != nil {
		return nil, err
	}

	fieldDim, err := typeutil.GetDim(base.outputFields[0])
	if err != nil {
		return nil, err
	}	

	c, err := createOpenAIEmbeddingClient()
	if err != nil {
		return nil, err
	}

	runner := OpenAIEmbeddingRunner{
		base: base,
		client: c,
		fieldDim: fieldDim,
	}

	for _, param := range schema.Params {
		if strings.ToLower(param.Key) == "modelName" {
			runner.modelName = param.Value
		}
		if strings.ToLower(param.Key) == "dim" {
			dim, err := strconv.ParseInt(param.Value, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("dim [%s] is not int", param.Value)
			}

			if dim != 0 && dim != runner.fieldDim {
				return nil, fmt.Errorf("Dim in field's schema is [%d], but embeding dim is [%d]", fieldDim, dim)
			}
			runner.fieldDim = dim
		}

		if strings.ToLower(param.Key) == "user" {
			runner.user = param.Value
		}
		
	}
	
	if runner.modelName != TextEmbeddingAda002 && runner.modelName != TextEmbedding3Small && runner.modelName != TextEmbedding3Large {
		return nil, fmt.Errorf("Unsupported model: %s, only support [%s, %s, %s]",
			runner.modelName, TextEmbeddingAda002, TextEmbedding3Small, TextEmbedding3Large)
	}
	return &runner, nil
}

func (runner *OpenAIEmbeddingRunner) Run(inputs []*schemapb.FieldData) ([]*schemapb.FieldData, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("OpenAIEmbedding function only receives one input, bug got [%d]", len(inputs))
	}

	if inputs[0].Type != schemapb.DataType_VarChar {
		return nil, fmt.Errorf("OpenAIEmbedding only supports varchar field, the input is not varchar")
	}

	texts := inputs[0].GetScalars().GetStringData().GetData()
	if texts == nil {
		return nil, fmt.Errorf("Input texts is empty")
	}

	if len(texts) > maxRowNum {
		return nil, fmt.Errorf("OpenAI embedding supports up to [%d] pieces of data at a time, got [%d]", maxRowNum, len(texts))
	}
	
	var output_field schemapb.FieldData
	output_field.FieldId = runner.base.outputFields[0].FieldID
	output_field.FieldName = runner.base.outputFields[0].Name
	output_field.Type = runner.base.outputFields[0].DataType
	output_field.IsDynamic = false
	output_field.GetVectors().Dim = runner.fieldDim
	numRows := len(texts)
	output_field.GetVectors().GetFloatVector().Data = make([]float32, numRows * int(runner.fieldDim))

	for i := 0; i < numRows; i += maxBatch {
		end := i + maxBatch
		if end > numRows {
			end = numRows
		}
		resp, err := runner.client.Embedding(runner.modelName, texts[i:end], int(runner.embedDimParam), runner.user, timeoutSec)
		if err != nil {
			return nil, err
		}

		if len(resp.Data[0].Embedding) != int(runner.fieldDim) {
			return nil, fmt.Errorf("Dim in field's schema is [%d], but embeding dim is [%d]", runner.fieldDim, len(resp.Data[0].Embedding))
		}

		for _, item := range resp.Data {
			output_field.GetVectors().GetFloatVector().Data = append(output_field.GetVectors().GetFloatVector().Data, item.Embedding...)
		}
	}
	return []*schemapb.FieldData{&output_field}, nil
}
