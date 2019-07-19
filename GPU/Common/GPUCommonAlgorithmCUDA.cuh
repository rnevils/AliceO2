// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonAlgorithmCUDA.cuh
/// \author David Rohr

#ifndef GPUCOMMONALGORITHMCUDA_CUH
#define GPUCOMMONALGORITHMCUDA_CUH

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda.h>

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUCommonAlgorithm
{
 public:
  template <class T>
  GPUd() static void sort(T* begin, T* end);
  template <class T>
  GPUd() static void sortInBlock(T* begin, T* end);
  template <class T, class S>
  GPUd() static void sort(T* begin, T* end, const S& comp);
  template <class T, class S>
  GPUd() static void sortInBlock(T* begin, T* end, const S& comp);
};

template <class T>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end)
{
	thrust::sort(thrust::seq,begin,end);
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sort(T* begin, T* end, const S& comp)
{
	thrust::sort(thrust::seq,begin,end,comp);
}

template <class T>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end)
{
	if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
		thrust::sort(thrust::cuda::par,begin,end);
	}
}

template <class T, class S>
GPUdi() void GPUCommonAlgorithm::sortInBlock(T* begin, T* end, const S& comp)
{
	thrust::sort(thrust::cuda::par,begin,end,comp);
}

typedef GPUCommonAlgorithm CAAlgo;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
