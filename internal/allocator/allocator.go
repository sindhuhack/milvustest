package allocator

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/zilliztech/milvus-distributed/internal/proto/masterpb"
	"google.golang.org/grpc"
)

const (
	maxConcurrentRequests = 10000
)

type request interface {
	Wait()
	Notify(error)
	IsValid() bool
}

type baseRequest struct {
	done  chan error
	valid bool
}

func (req *baseRequest) Wait() {
	err := <-req.done
	req.valid = err == nil
}

func (req *baseRequest) IsValid() bool {
	return req.valid
}

func (req *baseRequest) Notify(err error) {
	req.done <- err
}

type idRequest struct {
	baseRequest
	id    UniqueID
	count uint32
}

type tsoRequest struct {
	baseRequest
	timestamp Timestamp
	count     uint32
}

type segRequest struct {
	baseRequest
	count     uint32
	colName   string
	partition string
	segID     UniqueID
	channelID int32
}

type syncRequest struct {
	baseRequest
}

type tickerChan interface {
	Chan() <-chan time.Time
	Close()
	Init()
	Reset()
}

type emptyTicker struct {
	tChan <-chan time.Time
}

func (t *emptyTicker) Chan() <-chan time.Time {
	return t.tChan
}

func (t *emptyTicker) Init() {
}

func (t *emptyTicker) Reset() {
}

func (t *emptyTicker) Close() {
}

type ticker struct {
	ticker         *time.Ticker
	updateInterval time.Duration //
}

func (t *ticker) Init() {
	t.ticker = time.NewTicker(t.updateInterval)
}

func (t *ticker) Reset() {
	t.ticker.Reset(t.updateInterval)
}

func (t *ticker) Close() {
	t.ticker.Stop()
}

func (t *ticker) Chan() <-chan time.Time {
	return t.ticker.C
}

type Allocator struct {
	reqs chan request

	wg     sync.WaitGroup
	ctx    context.Context
	cancel context.CancelFunc

	masterAddress string
	masterConn    *grpc.ClientConn
	masterClient  masterpb.MasterClient
	countPerRPC   uint32

	toDoReqs []request

	syncReqs []request

	tChan         tickerChan
	forceSyncChan chan request

	syncFunc    func()
	processFunc func(req request) error
	checkFunc   func(timeout bool) bool
}

func (ta *Allocator) Start() error {
	err := ta.connectMaster()
	if err != nil {
		panic("connect to master failed")
	}
	ta.tChan.Init()
	ta.wg.Add(1)
	go ta.mainLoop()
	return nil
}

func (ta *Allocator) connectMaster() error {
	log.Printf("Connected to master, master_addr=%s", ta.masterAddress)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(ctx, ta.masterAddress, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Printf("Connect to master failed, error= %v", err)
		return err
	}
	log.Printf("Connected to master, master_addr=%s", ta.masterAddress)
	ta.masterConn = conn
	ta.masterClient = masterpb.NewMasterClient(conn)
	return nil
}

func (ta *Allocator) init() {
	ta.forceSyncChan = make(chan request, maxConcurrentRequests)
}

func (ta *Allocator) mainLoop() {
	defer ta.wg.Done()

	loopCtx, loopCancel := context.WithCancel(ta.ctx)
	defer loopCancel()

	for {
		select {

		case first := <-ta.forceSyncChan:
			ta.syncReqs = append(ta.syncReqs, first)
			pending := len(ta.forceSyncChan)
			for i := 0; i < pending; i++ {
				ta.syncReqs = append(ta.syncReqs, <-ta.forceSyncChan)
			}
			ta.sync(true)
			ta.finishSyncRequest()

		case <-ta.tChan.Chan():
			ta.sync(true)

		case first := <-ta.reqs:
			ta.toDoReqs = append(ta.toDoReqs, first)
			pending := len(ta.reqs)
			for i := 0; i < pending; i++ {
				ta.toDoReqs = append(ta.toDoReqs, <-ta.reqs)
			}
			ta.sync(false)

			ta.finishRequest()

		case <-loopCtx.Done():
			return
		}

	}
}

func (ta *Allocator) sync(timeout bool) {
	if ta.syncFunc == nil {
		return
	}
	if ta.checkFunc == nil || !ta.checkFunc(timeout) {
		return
	}

	ta.syncFunc()

	if !timeout {
		ta.tChan.Reset()
	}
	fmt.Println("synced")
}

func (ta *Allocator) finishSyncRequest() {
	for _, req := range ta.syncReqs {
		if req != nil {
			req.Notify(nil)
		}
	}
	ta.syncReqs = ta.syncReqs[0:0]
}

func (ta *Allocator) finishRequest() {
	for _, req := range ta.toDoReqs {
		if req != nil {
			err := ta.processFunc(req)
			req.Notify(err)
		}
	}
	ta.toDoReqs = ta.toDoReqs[0:0]
}

func (ta *Allocator) revokeRequest(err error) {
	n := len(ta.reqs)
	for i := 0; i < n; i++ {
		req := <-ta.reqs
		req.Notify(err)
	}
}

func (ta *Allocator) Close() {
	ta.cancel()
	ta.wg.Wait()
	ta.tChan.Close()
	ta.revokeRequest(errors.New("closing"))
}

func (ta *Allocator) CleanCache() {
	req := &syncRequest{baseRequest: baseRequest{done: make(chan error), valid: false}}
	ta.forceSyncChan <- req
	req.Wait()
}
