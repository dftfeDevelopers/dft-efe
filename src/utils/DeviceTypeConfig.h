// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------

/*
 * @author Ian C. Lin., Sambit Das
 */
#ifndef dftefeDeviceTypeConfig_h
#define dftefeDeviceTypeConfig_h

#ifdef DFTEFE_WITH_DEVICE
#  ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
#    include "DeviceTypeConfig.cu.h"
#  elif DFTEFE_WITH_DEVICE_LANG_HIP
#    include "DeviceTypeConfig.hip.h"
#  elif DFTEFE_WITH_DEVICE_LANG_SYCL
#    include "DeviceTypeConfig.sycl.h"
#  endif
#else
namespace dftefe
{
  namespace utils
  {
    typedef int           deviceError_t;
    typedef int           deviceStream_t;
    typedef int           deviceEvent_t;
    typedef int           deviceBlasStatus_t;
    typedef int           deviceBlasHandle_t;
    static deviceStream_t defaultStream = 0;
  } // namespace utils
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
#endif // dftefeDeviceTypeConfig_h
