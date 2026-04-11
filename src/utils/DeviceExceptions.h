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

/**
 * @author Gourab Panigrahi
 *
 */

#ifdef DFTEFE_WITH_DEVICE
#  ifndef dftefeDeviceExceptions_h
#    define dftefeDeviceExceptions_h

#    ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
#      include "DeviceExceptions.cu.h"
#    elif DFTEFE_WITH_DEVICE_LANG_HIP
#      include "DeviceExceptions.hip.h"
#    elif DFTEFE_WITH_DEVICE_LANG_SYCL
#      include "DeviceExceptions.sycl.h"
#    endif

#  endif // dftefeDeviceExceptions_h
#endif   // DFTEFE_WITH_DEVICE
