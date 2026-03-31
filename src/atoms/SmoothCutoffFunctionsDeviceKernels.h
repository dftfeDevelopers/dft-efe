/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * Inline DFTEFE_DEVICE_FUNC definitions for smooth cutoff device functions.
 * Included by SmoothCutoffFunctions.h so that every translation unit that
 * includes SmoothCutoffFunctions.h gets its own inline copy — no cross-TU
 * __device__ linkage required.
 *
 * @author Avirup Sircar
 */

#ifndef dftefe_SmoothCutoffFunctionsDeviceKernels_h
#define dftefe_SmoothCutoffFunctionsDeviceKernels_h

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    //=========================================================================
    // smoothCutoffValueDevice
    // Scalar per-point device function — inline body so every TU that
    // includes this header gets its own copy (no cross-TU __device__ call).
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    smoothCutoffValueDevice(const double x, const double r, const double d)
    {
      const double y        = 1.0 - d * (x - r) / r;
      const double f1_y     = (y <= 0.0) ? 0.0 : exp(-1.0 / y);
      const double omy      = 1.0 - y;
      const double f1_1my   = (omy <= 0.0) ? 0.0 : exp(-1.0 / omy);
      return f1_y / (f1_y + f1_1my);
    }

    //=========================================================================
    // smoothCutoffDerivativeDevice
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    smoothCutoffDerivativeDevice(const double x,
                                 const double r,
                                 const double d,
                                 const double tolerance)
    {
      const double y = 1.0 - d * (x - r) / r;
      if (fabs(y) < tolerance || fabs(1.0 - y) < tolerance)
        return 0.0;
      const double f1_y      = (y <= 0.0) ? 0.0 : exp(-1.0 / y);
      const double omy       = 1.0 - y;
      const double f1_1my    = (omy <= 0.0) ? 0.0 : exp(-1.0 / omy);
      const double f1Der_y   = f1_y / (y * y);
      const double f1Der_1my = f1_1my / (omy * omy);
      const double denom     = f1_y + f1_1my;
      const double f2Der =
        (f1Der_y * f1_1my + f1_y * f1Der_1my) / (denom * denom);
      return f2Der * (-d / r);
    }

  } // namespace atoms
} // namespace dftefe

#endif // DFTEFE_WITH_DEVICE
#endif // dftefe_SmoothCutoffFunctionsDeviceKernels_h
