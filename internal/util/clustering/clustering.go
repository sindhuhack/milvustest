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

package clustering

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"strconv"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
)

const (
	ClusteringCentroid    = "clustering.centroid"
	ClusteringSize        = "clustering.size"
	ClusteringId          = "clustering.id"
	ClusteringOperationid = "clustering.operationID"

	SearchEnableClustering      = "clustering.enable"
	SearchClusteringFilterRatio = "clustering.filter_ratio"
)

func ClusteringInfoFromKV(kv []*commonpb.KeyValuePair) (*internalpb.ClusteringInfo, error) {
	kvMap := funcutil.KeyValuePair2Map(kv)
	if v, ok := kvMap[ClusteringCentroid]; ok {
		var floatSlice []float32
		err := json.Unmarshal([]byte(v), &floatSlice)
		if err != nil {
			log.Error("Failed to parse cluster center value:", zap.String("value", v), zap.Error(err))
			return nil, err
		}
		clusterInfo := &internalpb.ClusteringInfo{
			Centroid: floatSlice,
		}
		if sizeStr, ok := kvMap[ClusteringSize]; ok {
			size, err := strconv.ParseInt(sizeStr, 10, 64)
			if err != nil {
				log.Error("Failed to parse cluster size value:", zap.String("value", sizeStr), zap.Error(err))
				return nil, err
			}
			clusterInfo.Size = size
		}
		if clusterIDStr, ok := kvMap[ClusteringId]; ok {
			clusterID, err := strconv.ParseInt(clusterIDStr, 10, 64)
			if err != nil {
				log.Error("Failed to parse cluster id value:", zap.String("value", clusterIDStr), zap.Error(err))
				return nil, err
			}
			clusterInfo.Id = clusterID
		}
		if operationIDStr, ok := kvMap[ClusteringOperationid]; ok {
			operationID, err := strconv.ParseInt(operationIDStr, 10, 64)
			if err != nil {
				log.Error("Failed to parse cluster group id value:", zap.String("value", operationIDStr), zap.Error(err))
				return nil, err
			}
			clusterInfo.OperationID = operationID
		}
		return clusterInfo, nil
	}
	return nil, nil
}

func SearchClusteringOptions(kv []*commonpb.KeyValuePair) (*internalpb.SearchClusteringOptions, error) {
	kvMap := funcutil.KeyValuePair2Map(kv)

	clusteringOptions := &internalpb.SearchClusteringOptions{
		Enable:      false,
		FilterRatio: 0.5, // default
	}

	if enable, ok := kvMap[SearchEnableClustering]; ok {
		b, err := strconv.ParseBool(enable)
		if err != nil {
			return nil, errors.New("illegal search params clustering.enable value, should be true or false")
		}
		clusteringOptions.Enable = b
	}

	if clusterBasedFilterRatio, ok := kvMap[SearchClusteringFilterRatio]; ok {
		b, err := strconv.ParseFloat(clusterBasedFilterRatio, 32)
		if err != nil {
			return nil, errors.New("illegal search params clustering.filter_ratio value, should be a float in range (0.0, 1.0]")
		}
		if b <= 0.0 || b > 1.0 {
			return nil, errors.New("invalid clustering.filter_ratio value, should be a float in range (0.0, 1.0]")
		}
		clusteringOptions.FilterRatio = float32(b)
	}

	return clusteringOptions, nil
}

func DeserializeFloatVector(data []byte) []float32 {
	vectorLen := len(data) / 4 // Each float32 occupies 4 bytes
	fv := make([]float32, vectorLen)

	for i := 0; i < vectorLen; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		fv[i] = math.Float32frombits(bits)
	}

	return fv
}

func SerializeFloatVector(fv []float32) []byte {
	data := make([]byte, 0, 4*len(fv)) // float32 occupies 4 bytes
	buf := make([]byte, 4)
	for _, f := range fv {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(f))
		data = append(data, buf...)
	}
	return data
}
