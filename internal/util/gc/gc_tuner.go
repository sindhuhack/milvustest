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

package gc

import (
	"math"
	"os"
	"runtime"
	"strconv"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/hardware"
	"go.uber.org/zap"
)

var defaultGOGC int
var previousGOGC uint32

var minGOGC uint32
var maxGOGC uint32

var memoryThreshold uint64
var action func(uint32)

type finalizer struct {
	ref *finalizerRef
}

type finalizerRef struct {
	parent *finalizer
}

// just a finializer to handle go gc
func finalizerHandler(f *finalizerRef) {
	optimizeGOGC()
	runtime.SetFinalizer(f, finalizerHandler)
}

func optimizeGOGC() {
	var m runtime.MemStats
	// This will trigger a STW so be careful
	runtime.ReadMemStats(&m)

	heapuse := m.HeapInuse

	totaluse := hardware.GetUsedMemoryCount()

	var newGoGC uint32
	if totaluse > memoryThreshold {
		newGoGC = minGOGC
	} else {
		heapTarget := memoryThreshold - (totaluse - heapuse)
		newGoGC = uint32(math.Floor(float64(heapTarget-heapuse) / float64(heapuse) * 100))
		if newGoGC < minGOGC {
			newGoGC = minGOGC
		} else if newGoGC > maxGOGC {
			newGoGC = maxGOGC
		}
	}

	action(newGoGC)

	toMB := func(mem uint64) uint64 {
		return mem / 1024 / 1024
	}

	log.Info("GC Tune done", zap.Uint32("previous GOGC", previousGOGC),
		zap.Uint64("heapuse ", toMB(heapuse)),
		zap.Uint64("total memory", toMB(totaluse)),
		zap.Uint64("next GC", toMB(m.NextGC)),
		zap.Uint32("new GOGC", newGoGC),
	)

	previousGOGC = newGoGC
}

func NewTuner(targetPercent float64, minimumGOGCConfig uint32, maximumGOGCConfig uint32, fn func(uint322 uint32)) *finalizer {
	// initiate GOGC parameter
	if envGOGC := os.Getenv("GOGC"); envGOGC != "" {
		n, err := strconv.Atoi(envGOGC)
		if err == nil {
			defaultGOGC = n
		}
	} else {
		// the default value of GOGC is 100 for now
		defaultGOGC = 100
	}
	action = fn
	minGOGC = minimumGOGCConfig
	maxGOGC = maximumGOGCConfig

	previousGOGC = uint32(defaultGOGC)

	totalMemory := hardware.GetMemoryCount()
	if totalMemory == 0 {
		log.Warn("Failed to get memory count, disable gc auto tune", zap.Int("Initial GoGC", defaultGOGC))
		// noop
		action = func(uint32) {}
		return nil
	}
	memoryThreshold = uint64(float64(totalMemory) * targetPercent)
	log.Info("GC Helper initialized.", zap.Uint32("Initial GoGC", previousGOGC),
		zap.Uint32("minimumGOGC", minGOGC),
		zap.Uint32("maximumGOGC", maxGOGC),
		zap.Uint64("memoryThreshold", memoryThreshold))
	f := &finalizer{}

	f.ref = &finalizerRef{parent: f}
	runtime.SetFinalizer(f.ref, finalizerHandler)
	f.ref = nil
	return f
}
