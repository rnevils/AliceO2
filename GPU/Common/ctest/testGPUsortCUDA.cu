// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testGPUsortCUDA.cu
/// \author ...

#define GPUCA_GPUTYPE_PASCAL

#define BOOST_TEST_MODULE Test GPU sort CUDA
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

//#ifndef GPUCA_O2_LIB
//#define GPUCA_O2_LIB
//#endif
//#ifndef HAVE_O2HEADERS
//#define HAVE_O2HEADERS
//#endif
//#ifndef GPUCA_TPC_GEOMETRY_O2
//#define GPUCA_TPC_GEOMETRY_O2
//#endif

#include <boost/test/unit_test.hpp>
#include "GPUCommonAlgorithm.h"

using namespace o2::gpu;

__global__ void testSort()
{
  float tmp[10];
  for (int i = 0; i < 10; i++) {
    tmp[i] = i;
  }
  CAAlgo::sort(&tmp[0], &tmp[10]);
}

/// @brief Basic test if we can create the interface
BOOST_AUTO_TEST_CASE(GPUsortCUDA1)
{
  testSort<<<1, 256>>>();
  BOOST_CHECK_EQUAL(cudaDeviceSynchronize(), CUDA_SUCCESS);
}
