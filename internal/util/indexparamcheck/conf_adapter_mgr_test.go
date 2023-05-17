// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package indexparamcheck

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_GetConfAdapterMgrInstance(t *testing.T) {
	adapterMgr := GetIndexCheckerMgrInstance()

	var adapter IndexChecker
	var err error
	var ok bool

	adapter, err = adapterMgr.GetChecker("invalid")
	assert.NotEqual(t, nil, err)
	assert.Equal(t, nil, adapter)

	adapter, err = adapterMgr.GetChecker(IndexFaissIDMap)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*flatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissIvfFlat)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ivfBaseChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissIvfPQ)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ivfPQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissIvfSQ8)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ivfSQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissBinIDMap)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*binFlatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissBinIvfFlat)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*binIVFFlatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexHNSW)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*hnswChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexNSG)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*nsgChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexANNOY)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*annoyChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexRHNSWFlat)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*rHnswFlatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexRHNSWPQ)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*rHnswPQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexRHNSWSQ)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*rHnswSQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexNGTPANNG)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ngtPANNGChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexNGTONNG)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ngtONNGChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexDISKANN)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*diskannChecker)
	assert.Equal(t, true, ok)
}

func TestConfAdapterMgrImpl_GetAdapter(t *testing.T) {
	adapterMgr := newIndexCheckerMgr()

	var adapter IndexChecker
	var err error
	var ok bool

	adapter, err = adapterMgr.GetChecker("invalid")
	assert.NotEqual(t, nil, err)
	assert.Equal(t, nil, adapter)

	adapter, err = adapterMgr.GetChecker(IndexFaissIDMap)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*flatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissIvfFlat)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ivfBaseChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissIvfPQ)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ivfPQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissIvfSQ8)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ivfSQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissBinIDMap)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*binFlatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexFaissBinIvfFlat)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*binIVFFlatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexHNSW)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*hnswChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexNSG)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*nsgChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexANNOY)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*annoyChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexRHNSWFlat)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*rHnswFlatChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexRHNSWPQ)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*rHnswPQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexRHNSWSQ)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*rHnswSQChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexNGTPANNG)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ngtPANNGChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexNGTONNG)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*ngtONNGChecker)
	assert.Equal(t, true, ok)

	adapter, err = adapterMgr.GetChecker(IndexDISKANN)
	assert.Equal(t, nil, err)
	assert.NotEqual(t, nil, adapter)
	_, ok = adapter.(*diskannChecker)
	assert.Equal(t, true, ok)
}

func TestConfAdapterMgrImpl_GetAdapter_multiple_threads(t *testing.T) {
	num := 4
	mgr := newIndexCheckerMgr()
	var wg sync.WaitGroup
	for i := 0; i < num; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			adapter, err := mgr.GetChecker(IndexHNSW)
			assert.NoError(t, err)
			assert.NotNil(t, adapter)
		}()
	}
	wg.Wait()
}
