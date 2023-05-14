package segments

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type defaultLimitReducer struct {
	req    *querypb.QueryRequest
	schema *schemapb.CollectionSchema
}

func (r *defaultLimitReducer) Reduce(ctx context.Context, results []*internalpb.RetrieveResults) (*internalpb.RetrieveResults, error) {
	return mergeInternalRetrieveResultsAndFillIfEmpty(ctx, results, r.req.GetReq().GetLimit(), r.req.GetReq().GetOutputFieldsId(), r.schema)
}

func newDefaultLimitReducer(req *querypb.QueryRequest, schema *schemapb.CollectionSchema) *defaultLimitReducer {
	return &defaultLimitReducer{
		req:    req,
		schema: schema,
	}
}

type extensionLimitReducer struct {
	req           *querypb.QueryRequest
	schema        *schemapb.CollectionSchema
	extendedLimit int64
}

func (r *extensionLimitReducer) Reduce(ctx context.Context, results []*internalpb.RetrieveResults) (*internalpb.RetrieveResults, error) {
	return mergeInternalRetrieveResultsAndFillIfEmpty(ctx, results, typeutil.Unlimited, r.req.GetReq().GetOutputFieldsId(), r.schema)
}

func newExtensionLimitReducer(req *querypb.QueryRequest, schema *schemapb.CollectionSchema, etdLimit int64) *extensionLimitReducer {
	return &extensionLimitReducer{
		req:           req,
		schema:        schema,
		extendedLimit: etdLimit,
	}
}

type defaultLimitReducerSegcore struct {
	req    *querypb.QueryRequest
	schema *schemapb.CollectionSchema
}

func (r *defaultLimitReducerSegcore) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults) (*segcorepb.RetrieveResults, error) {
	return mergeSegcoreRetrieveResultsAndFillIfEmpty(ctx, results, r.req.GetReq().GetLimit(), r.req.GetReq().GetOutputFieldsId(), r.schema)
}

func newDefaultLimitReducerSegcore(req *querypb.QueryRequest, schema *schemapb.CollectionSchema) *defaultLimitReducerSegcore {
	return &defaultLimitReducerSegcore{
		req:    req,
		schema: schema,
	}
}

type extensionLimitSegcoreReducer struct {
	req    *querypb.QueryRequest
	schema *schemapb.CollectionSchema
}

func (r *extensionLimitSegcoreReducer) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults) (*segcorepb.RetrieveResults, error) {
	return mergeSegcoreRetrieveResultsAndFillIfEmpty(ctx, results, typeutil.Unlimited, r.req.GetReq().GetOutputFieldsId(), r.schema)
}

func newExtensionLimitSegcoreReducer(req *querypb.QueryRequest, schema *schemapb.CollectionSchema) *extensionLimitSegcoreReducer {
	return &extensionLimitSegcoreReducer{
		req:    req,
		schema: schema,
	}
}
