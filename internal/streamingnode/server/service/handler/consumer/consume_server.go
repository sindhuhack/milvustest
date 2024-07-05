package consumer

import (
	"io"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/streamingpb"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal"
	"github.com/milvus-io/milvus/internal/streamingnode/server/walmanager"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/internal/util/streamingutil/typeconverter"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/syncutil"
)

// CreateConsumeServer create a new consumer.
// Expected message sequence:
// CreateConsumeServer:
// -> ConsumeResponse 1
// -> ConsumeResponse 2
// -> ConsumeResponse 3
// CloseConsumer:
func CreateConsumeServer(walManager walmanager.Manager, streamServer streamingpb.StreamingNodeHandlerService_ConsumeServer) (*ConsumeServer, error) {
	createReq, err := contextutil.GetCreateConsumer(streamServer.Context())
	if err != nil {
		return nil, errors.Wrap(err, "at get create consumer request")
	}

	pchanelInfo := typeconverter.NewPChannelInfoFromProto(createReq.Pchannel)
	l, err := walManager.GetAvailableWAL(pchanelInfo)
	if err != nil {
		return nil, errors.Wrap(err, "at get available wal")
	}
	deliverPolicy, err := typeconverter.NewDeliverPolicyFromProto(l.WALName(), createReq.GetDeliverPolicy())
	if err != nil {
		return nil, errors.Wrap(err, "at convert deliver policy")
	}
	deliverFilters, err := newMessageFilter(createReq.DeliverFilters)
	if err != nil {
		return nil, errors.Wrap(err, "at convert deliver filters")
	}
	scanner, err := l.Read(streamServer.Context(), wal.ReadOption{
		DeliverPolicy: deliverPolicy,
		MessageFilter: deliverFilters,
	})
	if err != nil {
		return nil, errors.Wrap(err, "at create scanner")
	}

	consumeServer := &consumeGrpcServerHelper{
		StreamingNodeHandlerService_ConsumeServer: streamServer,
	}
	if err := consumeServer.SendCreated(&streamingpb.CreateConsumerResponse{}); err != nil {
		// release the scanner to avoid resource leak.
		if err := scanner.Close(); err != nil {
			log.Warn("close scanner failed at create consume server", zap.Error(err))
		}
		return nil, errors.Wrap(err, "at send created")
	}
	return &ConsumeServer{
		scanner:       scanner,
		consumeServer: consumeServer,
		logger:        log.With(zap.String("channel", l.Channel().Name), zap.Int64("term", l.Channel().Term)), // Add trace info for all log.
	}, nil
}

// ConsumeServer is a ConsumeServer of log messages.
type ConsumeServer struct {
	scanner       wal.Scanner
	consumeServer *consumeGrpcServerHelper
	logger        *log.MLogger
}

// Execute executes the consumer.
func (c *ConsumeServer) Execute() error {
	// sender: recv arm, receiver: send arm, with buffer 1 to avoid block.
	recvFailureSignal := syncutil.NewFuture[error]()

	// Start a recv arm to handle the control message on background.
	go func() {
		// recv loop will be blocked until the stream is closed.
		// 1. close by client.
		// 2. close by server context cancel by return of outside Execute.
		_ = c.recvLoop(recvFailureSignal)
	}()

	// Start a send loop on current main goroutine.
	// the loop will be blocked until:
	// 1. the stream is broken.
	// 2. recv arm recv close signal.
	// 3. scanner is quit with expected error.
	return c.sendLoop(recvFailureSignal)
}

// sendLoop sends the message to client.
func (c *ConsumeServer) sendLoop(recvChanSignal *syncutil.Future[error]) (err error) {
	defer func() {
		if err != nil {
			c.logger.Warn("send arm of stream closed by unexpected error", zap.Error(err))
		} else {
			c.logger.Info("send arm of stream closed")
		}
		if err := c.scanner.Close(); err != nil {
			c.logger.Warn("close scanner failed", zap.Error(err))
		}
	}()
	// Read ahead buffer is implemented by scanner.
	// Do not add buffer here.
	for {
		select {
		case msg, ok := <-c.scanner.Chan():
			if !ok {
				return errors.Wrap(c.scanner.Error(), "at scanner")
			}
			// Send Consumed message to client and do metrics.
			messageSize := msg.EstimateSize()
			if err := c.consumeServer.SendConsumeMessage(&streamingpb.ConsumeMessageReponse{
				Id: &streamingpb.MessageID{
					Id: msg.MessageID().Marshal(),
				},
				Message: &streamingpb.Message{
					Payload:    msg.Payload(),
					Properties: msg.Properties().ToRawMap(),
				},
			}); err != nil {
				return errors.Wrap(err, "at send consume message")
			}
			metrics.StreamingNodeConsumeBytes.WithLabelValues(paramtable.GetStringNodeID()).Observe(float64(messageSize))
		case <-recvChanSignal.Done():
			err := recvChanSignal.Get()
			c.logger.Info("recv channel notified", zap.Error(err))
			if err := c.consumeServer.SendClosed(); err != nil {
				c.logger.Warn("send close failed", zap.Error(err))
				return errors.Wrap(err, "at send close")
			}
			return errors.Wrap(err, "at recv failure channel")
		case <-c.consumeServer.Context().Done():
			return errors.Wrap(c.consumeServer.Context().Err(), "at grpc context done")
		}
	}
}

// recvLoop receives messages from client.
func (c *ConsumeServer) recvLoop(recvFailureCh *syncutil.Future[error]) (err error) {
	defer func() {
		recvFailureCh.Set(err)
		if err != nil {
			c.logger.Warn("recv arm of stream closed by unexpected error", zap.Error(err))
		} else {
			c.logger.Info("recv arm of stream closed")
		}
	}()

	for {
		req, err := c.consumeServer.Recv()
		if err == io.EOF {
			// should always return by ConsumeRequest_Close message.
			c.logger.Warn("stream closed by client unexpectedly")
			return io.ErrUnexpectedEOF
		}
		if err != nil {
			return err
		}
		switch req := req.Request.(type) {
		case *streamingpb.ConsumeRequest_Close:
			return nil
			// should be eof next.
		default:
			// skip unknown message here, to keep the forward compatibility.
			c.logger.Warn("unknown request type", zap.Any("request", req))
		}
	}
}

func newMessageFilter(filters []*streamingpb.DeliverFilter) (wal.MessageFilter, error) {
	fs, err := typeconverter.NewDeliverFiltersFromProtos(filters)
	if err != nil {
		return nil, err
	}
	return func(msg message.ImmutableMessage) bool {
		for _, f := range fs {
			if !f.Filter(msg) {
				return false
			}
		}
		return true
	}, nil
}
