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

package tikv

import (
	"context"
	"os"
	"testing"

	"github.com/tikv/client-go/v2/rawkv"
	"github.com/tikv/client-go/v2/testutils"
	tilib "github.com/tikv/client-go/v2/tikv"
	"github.com/tikv/client-go/v2/txnkv"

	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

var TestParams paramtable.ComponentParam
var txnClient *txnkv.Client
var rawClient *rawkv.Client

// creates a locak TiKV Store for testing purpose.
func setupLocalTiKV() {
	setupLocalTxn()
	setupLocalRaw()
}
func setupLocalTxn() {
	client, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	if err != nil {
		panic(err)
	}
	testutils.BootstrapWithSingleStore(cluster)
	store, err := tilib.NewTestTiKVStore(client, pdClient, nil, nil, 0)
	if err != nil {
		panic(err)
	}
	txnClient = &txnkv.Client{KVStore: store}
}

func setupLocalRaw() {
	client, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	if err != nil {
		panic(err)
	}
	testutils.BootstrapWithSingleStore(cluster)
	rawClient = &rawkv.Client{}
	p := rawkv.ClientProbe{Client: rawClient}
	p.SetPDClient(pdClient)
	p.SetRegionCache(tilib.NewRegionCache(pdClient))
	p.SetRPCClient(client)
}

// Connects to a remote TiKV service for testing purpose. By default, it assumes the TiKV is from localhost.
func setupRemoteTiKV() {
	pdsn := "127.0.0.1:2379"
	var err error
	txnClient, err = txnkv.NewClient([]string{pdsn})
	if err != nil {
		panic(err)
	}
	rawClient, err = rawkv.NewClientWithOpts(context.Background(), []string{pdsn})
	if err != nil {
		panic(err)
	}
}

func setupTiKV(use_remote bool) {
	if use_remote {
		setupRemoteTiKV()
	} else {
		setupLocalTiKV()
	}
}

func TestMain(m *testing.M) {
	TestParams.Init()
	setupTiKV(false)
	code := m.Run()
	os.Exit(code)
}
