/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Transpose.cuh>

namespace faiss { namespace gpu {

FlatIndex::FlatIndex(GpuResources* res,
                     int dim,
                     bool useFloat16,
                     bool storeTransposed,
                     MemorySpace space) :
    resources_(res),
    dim_(dim),
    useFloat16_(useFloat16),
    storeTransposed_(storeTransposed),
    space_(space),
    num_(0),
    rawData_(space) {
#ifndef FAISS_USE_FLOAT16
    FAISS_ASSERT(!useFloat16_);
#endif
}

bool
FlatIndex::getUseFloat16() const {
  return useFloat16_;
}

/// Returns the number of vectors we contain
int FlatIndex::getSize() const {
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    return vectorsHalf_.getSize(0);
  } else {
    return vectors_.getSize(0);
  }
#else
  return vectors_.getSize(0);
#endif
}

int FlatIndex::getDim() const {
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    return vectorsHalf_.getSize(1);
  } else {
    return vectors_.getSize(1);
  }
#else
    return vectors_.getSize(1);
#endif
}

void
FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    rawData_.reserve(numVecs * dim_ * sizeof(half), stream);
  } else {
        rawData_.reserve(numVecs * dim_ * sizeof(float), stream);
  }
#else
    rawData_.reserve(numVecs * dim_ * sizeof(float), stream);
#endif
}

template <>
Tensor<float, 2, true>&
FlatIndex::getVectorsRef<float>() {
  // Should not call this unless we are in float32 mode
  FAISS_ASSERT(!useFloat16_);
  return getVectorsFloat32Ref();
}

#ifdef FAISS_USE_FLOAT16
template <>
Tensor<half, 2, true>&
FlatIndex::getVectorsRef<half>() {
  // Should not call this unless we are in float16 mode
  FAISS_ASSERT(useFloat16_);
  return getVectorsFloat16Ref();
}
#endif

Tensor<float, 2, true>&
FlatIndex::getVectorsFloat32Ref() {
  // Should not call this unless we are in float32 mode
  FAISS_ASSERT(!useFloat16_);

  return vectors_;
}

#ifdef FAISS_USE_FLOAT16
Tensor<half, 2, true>&
FlatIndex::getVectorsFloat16Ref() {
  // Should not call this unless we are in float16 mode
  FAISS_ASSERT(useFloat16_);

  return vectorsHalf_;
}
#endif

DeviceTensor<float, 2, true>
FlatIndex::getVectorsFloat32Copy(cudaStream_t stream) {
  return getVectorsFloat32Copy(0, num_, stream);
}

DeviceTensor<float, 2, true>
FlatIndex::getVectorsFloat32Copy(int from, int num, cudaStream_t stream) {
  DeviceTensor<float, 2, true> vecFloat32({num, dim_}, space_);

#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    auto halfNarrow = vectorsHalf_.narrowOutermost(from, num);
    convertTensor<half, float, 2>(stream, halfNarrow, vecFloat32);
  } else {
    vectors_.copyTo(vecFloat32, stream);
  }
#else
    vectors_.copyTo(vecFloat32, stream);
#endif

  return vecFloat32;
}

void
FlatIndex::query(Tensor<float, 2, true>& input,
                 Tensor<uint8_t, 1, true>& bitset,
                 int k,
                 faiss::MetricType metric,
                 float metricArg,
                 Tensor<float, 2, true>& outDistances,
                 Tensor<int, 2, true>& outIndices,
                 bool exactDistance) {
  auto stream = resources_->getDefaultStreamCurrentDevice();
  auto& mem = resources_->getMemoryManagerCurrentDevice();

#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    // We need to convert the input to float16 for comparison to ourselves

    auto inputHalf =
      convertTensor<float, half, 2>(resources_, stream, input);

    query(inputHalf, bitset, k, metric, metricArg,
          outDistances, outIndices, exactDistance);

  } else {
    bfKnnOnDevice(resources_,
                  getCurrentDevice(),
                  stream,
                  storeTransposed_ ? vectorsTransposed_ : vectors_,
                  !storeTransposed_, // is vectors row major?
                  &norms_,
                  input,
                  true, // input is row major
                  bitset,
                  k,
                  metric,
                  metricArg,
                  outDistances,
                  outIndices,
                  !exactDistance);
  }
#else
    bfKnnOnDevice(resources_,
                  getCurrentDevice(),
                  stream,
                  storeTransposed_ ? vectorsTransposed_ : vectors_,
                  !storeTransposed_, // is vectors row major?
                  &norms_,
                  input,
                  true, // input is row major
                  bitset,
                  k,
                  metric,
                  metricArg,
                  outDistances,
                  outIndices,
                  !exactDistance);
#endif
}

#ifdef FAISS_USE_FLOAT16
void
FlatIndex::query(Tensor<half, 2, true>& input,
                 Tensor<uint8_t, 1, true>& bitset,
                 int k,
                 faiss::MetricType metric,
                 float metricArg,
                 Tensor<float, 2, true>& outDistances,
                 Tensor<int, 2, true>& outIndices,
                 bool exactDistance) {
  FAISS_ASSERT(useFloat16_);

  bfKnnOnDevice(resources_,
                getCurrentDevice(),
                resources_->getDefaultStreamCurrentDevice(),
                storeTransposed_ ? vectorsHalfTransposed_ : vectorsHalf_,
                !storeTransposed_, // is vectors row major?
                &norms_,
                input,
                true, // input is row major
                bitset,
                k,
                metric,
                metricArg,
                outDistances,
                outIndices,
                !exactDistance);
}
#endif

void
FlatIndex::computeResidual(Tensor<float, 2, true>& vecs,
                           Tensor<int, 1, true>& listIds,
                           Tensor<float, 2, true>& residuals) {
#ifdef FAISS_USE_FLOAT16
    if (useFloat16_) {
    runCalcResidual(vecs,
                    getVectorsFloat16Ref(),
                    listIds,
                    residuals,
                    resources_->getDefaultStreamCurrentDevice());
  } else {
    runCalcResidual(vecs,
                    getVectorsFloat32Ref(),
                    listIds,
                    residuals,
                    resources_->getDefaultStreamCurrentDevice());
  }
#else
    runCalcResidual(vecs,
                    getVectorsFloat32Ref(),
                    listIds,
                    residuals,
                    resources_->getDefaultStreamCurrentDevice());
#endif
}

void
FlatIndex::reconstruct(Tensor<int, 1, true>& listIds,
                       Tensor<float, 2, true>& vecs) {
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    runReconstruct(listIds,
                   getVectorsFloat16Ref(),
                   vecs,
                   resources_->getDefaultStreamCurrentDevice());
  } else {
    runReconstruct(listIds,
                   getVectorsFloat32Ref(),
                   vecs,
                   resources_->getDefaultStreamCurrentDevice());
  }
#else
    runReconstruct(listIds,
                   getVectorsFloat32Ref(),
                   vecs,
                   resources_->getDefaultStreamCurrentDevice());
#endif
}

void
FlatIndex::reconstruct(Tensor<int, 2, true>& listIds,
                       Tensor<float, 3, true>& vecs) {
  auto listIds1 = listIds.downcastOuter<1>();
  auto vecs2 = vecs.downcastOuter<2>();

  reconstruct(listIds1, vecs2);
}

void
FlatIndex::add(const float* data, int numVecs, cudaStream_t stream) {
  if (numVecs == 0) {
    return;
  }

#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    // Make sure that `data` is on our device; we'll run the
    // conversion on our device
    auto devData = toDevice<float, 2>(resources_,
                                      getCurrentDevice(),
                                      (float*) data,
                                      stream,
                                      {numVecs, dim_});

    auto devDataHalf =
      convertTensor<float, half, 2>(resources_, stream, devData);

    rawData_.append((char*) devDataHalf.data(),
                    devDataHalf.getSizeInBytes(),
                    stream,
                    true /* reserve exactly */);
  } else {
    rawData_.append((char*) data,
                    (size_t) dim_ * numVecs * sizeof(float),
                    stream,
                    true /* reserve exactly */);
  }
#else
  rawData_.append((char*) data,
          (size_t) dim_ * numVecs * sizeof(float),
          stream,
          true /* reserve exactly */);
#endif
  num_ += numVecs;

#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    DeviceTensor<half, 2, true> vectorsHalf(
      (half*) rawData_.data(), {(int) num_, dim_}, space_);
    vectorsHalf_ = std::move(vectorsHalf);
  } else {
    DeviceTensor<float, 2, true> vectors(
      (float*) rawData_.data(), {(int) num_, dim_}, space_);
    vectors_ = std::move(vectors);
  }
#else
  DeviceTensor<float, 2, true> vectors(
     (float*) rawData_.data(), {(int) num_, dim_}, space_);
    vectors_ = std::move(vectors);
#endif

  if (storeTransposed_) {
#ifdef FAISS_USE_FLOAT16
    if (useFloat16_) {
      vectorsHalfTransposed_ =
        std::move(DeviceTensor<half, 2, true>({dim_, (int) num_}, space_));
      runTransposeAny(vectorsHalf_, 0, 1, vectorsHalfTransposed_, stream);
    } else {
      vectorsTransposed_ =
        std::move(DeviceTensor<float, 2, true>({dim_, (int) num_}, space_));
      runTransposeAny(vectors_, 0, 1, vectorsTransposed_, stream);
    }
#else
  vectorsTransposed_ =
    std::move(DeviceTensor<float, 2, true>({dim_, (int) num_}, space_));
  runTransposeAny(vectors_, 0, 1, vectorsTransposed_, stream);
#endif
  }

  // Precompute L2 norms of our database
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_) {
    DeviceTensor<float, 1, true> norms({(int) num_}, space_);
    runL2Norm(vectorsHalf_, true, norms, true, stream);
    norms_ = std::move(norms);
  } else {
    DeviceTensor<float, 1, true> norms({(int) num_}, space_);
    runL2Norm(vectors_, true, norms, true, stream);
    norms_ = std::move(norms);
  }
#else
  DeviceTensor<float, 1, true> norms({(int) num_}, space_);
  runL2Norm(vectors_, true, norms, true, stream);
  norms_ = std::move(norms);
#endif
}

void
FlatIndex::reset() {
  rawData_.clear();
  vectors_ = std::move(DeviceTensor<float, 2, true>());
  vectorsTransposed_ = std::move(DeviceTensor<float, 2, true>());
#ifdef FAISS_USE_FLOAT16
  vectorsHalf_ = std::move(DeviceTensor<half, 2, true>());
  vectorsHalfTransposed_ = std::move(DeviceTensor<half, 2, true>());
#endif
  norms_ = std::move(DeviceTensor<float, 1, true>());
  num_ = 0;
}

} }
