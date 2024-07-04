package balancer

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/resolver"

	"github.com/milvus-io/milvus/internal/mocks/google.golang.org/grpc/mock_balancer"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/attributes"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/internal/util/streamingutil/status"
	"github.com/milvus-io/milvus/pkg/util/interceptor"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

func TestServerIDPickerBuilder(t *testing.T) {
	builder := &serverIDPickerBuilder{}
	picker := builder.Build(base.PickerBuildInfo{})
	assert.NotNil(t, picker)
	_, err := picker.Pick(balancer.PickInfo{})
	assert.Error(t, err)
	assert.ErrorIs(t, err, balancer.ErrNoSubConnAvailable)

	picker = builder.Build(base.PickerBuildInfo{
		ReadySCs: map[balancer.SubConn]base.SubConnInfo{
			mock_balancer.NewMockSubConn(t): {
				Address: resolver.Address{
					Addr: "localhost:1",
					BalancerAttributes: attributes.WithServerID(
						new(attributes.Attributes),
						1,
					),
				},
			},
			mock_balancer.NewMockSubConn(t): {
				Address: resolver.Address{
					Addr: "localhost:2",
					BalancerAttributes: attributes.WithServerID(
						new(attributes.Attributes),
						2,
					),
				},
			},
		},
	})
	// Test round-robin
	serverIDSet := typeutil.NewSet[string]()
	info, err := picker.Pick(balancer.PickInfo{Ctx: context.Background()})
	assert.NoError(t, err)
	serverIDSet.Insert(info.Metadata.Get(interceptor.ServerIDKey)[0])
	info, err = picker.Pick(balancer.PickInfo{Ctx: context.Background()})
	assert.NoError(t, err)
	serverIDSet.Insert(info.Metadata.Get(interceptor.ServerIDKey)[0])
	serverIDSet.Insert(info.Metadata.Get(interceptor.ServerIDKey)[0])
	assert.Equal(t, 2, serverIDSet.Len())

	// Test force address
	info, err = picker.Pick(balancer.PickInfo{
		Ctx: contextutil.WithPickServerID(context.Background(), 1),
	})
	assert.NoError(t, err)
	assert.Equal(t, "1", info.Metadata.Get(interceptor.ServerIDKey)[0])

	// Test pick not exists
	info, err = picker.Pick(balancer.PickInfo{
		Ctx: contextutil.WithPickServerID(context.Background(), 3),
	})
	assert.Error(t, err)
	assert.NotNil(t, info)
}

func TestIsErrNoSubConnForPick(t *testing.T) {
	assert.True(t, IsErrNoSubConnForPick(ErrNoSubConnForPick))
	assert.False(t, IsErrNoSubConnForPick(errors.New("test")))
	err := status.ConvertStreamingError("test", ErrNoSubConnForPick)
	assert.True(t, IsErrNoSubConnForPick(err))
}
