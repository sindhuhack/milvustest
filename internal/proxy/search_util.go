package proxy

import (
	"context"
	"fmt"
	"strconv"

	"github.com/cockroachdb/errors"
	"github.com/golang/protobuf/proto"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/util/exprutil"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

func processPlaceholderGroup(t *searchTask) ([]byte, error) {
	annsField, ok := t.schema.GetFieldByName(t.annsFieldName)
	if !ok {
		return nil, merr.WrapErrParameterInvalidMsg("Search %s field does not exist in collection schema", t.annsFieldName)
	}

	dim, err := t.schema.schemaHelper.GetVectorDimFromID(annsField.GetFieldID())
	if err != nil {
		return nil, err
	}
	fieldType := annsField.GetDataType()

	phg := &commonpb.PlaceholderGroup{}
	phgBytes := t.request.GetPlaceholderGroup()
	err = proto.Unmarshal(phgBytes, phg)
	if err != nil {
		return nil, err
	}
	for _, phv := range phg.GetPlaceholders() {
		// TODO fp32, fp16 and bf16 can be converted here
		if int32(phv.GetType()) != int32(fieldType) {
			return nil, merr.WrapErrParameterInvalidMsg("ANNS field %s type %s cannot be searched by input %s", t.annsFieldName, fieldType.String(), phv.GetType().String)
		}
		// sparse vector length is variable
		if fieldType != schemapb.DataType_SparseFloatVector {
			expectLength := typeutil.VectorBytesLength(fieldType, dim)
			for _, value := range phv.Values {
				if expectLength != len(value) {
					return nil, merr.WrapErrParameterInvalidMsg("ANNS vector length not valid for %s, dim=%d, expect %d bytes, get %d bytes",
						annsField.GetName(), dim, expectLength, len(value))
				}
			}
		}
	}
	return phgBytes, nil
}

func initSearchRequest(ctx context.Context, t *searchTask, isHybrid bool) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "init search request")
	defer sp.End()

	log := log.Ctx(ctx).With(zap.Int64("collID", t.GetCollectionID()), zap.String("collName", t.collectionName))
	// fetch search_growing from search param
	var ignoreGrowing bool
	var err error
	for i, kv := range t.request.GetSearchParams() {
		if kv.GetKey() == IgnoreGrowingKey {
			ignoreGrowing, err = strconv.ParseBool(kv.GetValue())
			if err != nil {
				return errors.New("parse search growing failed")
			}
			t.request.SearchParams = append(t.request.GetSearchParams()[:i], t.request.GetSearchParams()[i+1:]...)
			break
		}
	}
	t.SearchRequest.IgnoreGrowing = ignoreGrowing

	// Manually update nq if not set.
	nq, err := getNq(t.request)
	if err != nil {
		log.Warn("failed to get nq", zap.Error(err))
		return err
	}
	// Check if nq is valid:
	// https://milvus.io/docs/limitations.md
	if err := validateNQLimit(nq); err != nil {
		return fmt.Errorf("%s [%d] is invalid, %w", NQKey, nq, err)
	}
	t.SearchRequest.Nq = nq
	log = log.With(zap.Int64("nq", nq))

	outputFieldIDs, err := getOutputFieldIDs(t.schema, t.request.GetOutputFields())
	if err != nil {
		log.Warn("fail to get output field ids", zap.Error(err))
		return err
	}
	t.SearchRequest.OutputFieldsId = outputFieldIDs

	if t.request.GetDslType() == commonpb.DslType_BoolExprV1 {
		annsFieldName, err := funcutil.GetAttrByKeyFromRepeatedKV(AnnsFieldKey, t.request.GetSearchParams())
		if err != nil || len(annsFieldName) == 0 {
			vecFields := typeutil.GetVectorFieldSchemas(t.schema.CollectionSchema)
			if len(vecFields) == 0 {
				return errors.New(AnnsFieldKey + " not found in schema")
			}

			if enableMultipleVectorFields && len(vecFields) > 1 {
				return errors.New("multiple anns_fields exist, please specify a anns_field in search_params")
			}

			annsFieldName = vecFields[0].Name
		}
		t.annsFieldName = annsFieldName
		queryInfo, offset, err := parseSearchInfo(t.request.GetSearchParams(), t.schema.CollectionSchema)
		annField := typeutil.GetFieldByName(t.schema.CollectionSchema, annsFieldName)
		if queryInfo.GetGroupByFieldId() != -1 && isHybrid {
			return errors.New("not support search_group_by operation in the hybrid search")
		}
		if queryInfo.GetGroupByFieldId() != -1 && annField.GetDataType() == schemapb.DataType_BinaryVector {
			return errors.New("not support search_group_by operation based on binary vector column")
		}

		if err != nil {
			return err
		}
		t.offset = offset

		plan, err := planparserv2.CreateSearchPlan(t.schema.schemaHelper, t.request.Dsl, annsFieldName, queryInfo)
		if err != nil {
			log.Warn("failed to create query plan", zap.Error(err),
				zap.String("dsl", t.request.Dsl), // may be very large if large term passed.
				zap.String("anns field", annsFieldName), zap.Any("query info", queryInfo))
			return merr.WrapErrParameterInvalidMsg("failed to create query plan: %v", err)
		}
		log.Debug("create query plan",
			zap.String("dsl", t.request.Dsl), // may be very large if large term passed.
			zap.String("anns field", annsFieldName), zap.Any("query info", queryInfo))

		if t.partitionKeyMode {
			expr, err := exprutil.ParseExprFromPlan(plan)
			if err != nil {
				log.Warn("failed to parse expr", zap.Error(err))
				return err
			}
			partitionKeys := exprutil.ParseKeys(expr, exprutil.PartitionKey)
			hashedPartitionNames, err := assignPartitionKeys(ctx, t.request.GetDbName(), t.collectionName, partitionKeys)
			if err != nil {
				log.Warn("failed to assign partition keys", zap.Error(err))
				return err
			}

			if len(hashedPartitionNames) > 0 {
				// translate partition name to partition ids. Use regex-pattern to match partition name.
				t.SearchRequest.PartitionIDs, err = getPartitionIDs(ctx, t.request.GetDbName(), t.collectionName, hashedPartitionNames)
				if err != nil {
					log.Warn("failed to get partition ids", zap.Error(err))
					return err
				}

				if t.enableMaterializedView {
					if planPtr := plan.GetVectorAnns(); planPtr != nil {
						planPtr.QueryInfo.MaterializedViewInvolved = true
					}
				}
			}
		}

		plan.OutputFieldIds = outputFieldIDs

		t.SearchRequest.Topk = queryInfo.GetTopk()
		t.SearchRequest.MetricType = queryInfo.GetMetricType()
		t.queryInfo = queryInfo
		t.SearchRequest.DslType = commonpb.DslType_BoolExprV1

		estimateSize, err := t.estimateResultSize(nq, t.SearchRequest.Topk)
		if err != nil {
			log.Warn("failed to estimate result size", zap.Error(err))
			return err
		}
		if estimateSize >= requeryThreshold {
			t.requery = true
			plan.OutputFieldIds = nil
		}

		t.SearchRequest.SerializedExprPlan, err = proto.Marshal(plan)
		if err != nil {
			return err
		}

		t.SearchRequest.PlaceholderGroup, err = processPlaceholderGroup(t)
		if err != nil {
			return err
		}

		log.Debug("proxy init search request",
			zap.Int64s("plan.OutputFieldIds", plan.GetOutputFieldIds()),
			zap.Stringer("plan", plan)) // may be very large if large term passed.
	}

	if deadline, ok := t.TraceCtx().Deadline(); ok {
		t.SearchRequest.TimeoutTimestamp = tsoutil.ComposeTSByTime(deadline, 0)
	}

	// Set username of this search request for feature like task scheduling.
	if username, _ := GetCurUserFromContext(ctx); username != "" {
		t.SearchRequest.Username = username
	}

	return nil
}
