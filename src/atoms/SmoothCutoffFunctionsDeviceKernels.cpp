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
 * @author Avirup Sircar
 *
 * Batch (all-points) kernel launchers for smooth cutoff functions on DEVICE.
 * Scalar smoothCutoffValue/smoothCutoffDerivative are inline DFTEFE_HOST_DEVICE_FUNC
 * in SmoothCutoffFunctions.h and are called directly from the device kernels below.
 */

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/Exceptions.h>
#  include <atoms/SmoothCutoffFunctions.h>
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      DFTEFE_CREATE_KERNEL(
        void,
        SmoothCutoffValueKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = smoothCutoffValue(x[i], r, d);
        },
        const size_type nPoints,
        const double *  x,
        const double    r,
        const double    d,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        SmoothCutoffDerivativeKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = smoothCutoffDerivative(x[i], r, d, tolerance);
        },
        const size_type nPoints,
        const double *  x,
        const double    r,
        const double    d,
        const double    tolerance,
        double *        out);

    } // anonymous namespace

    template <>
    void
    smoothCutoffValue<utils::MemorySpace::DEVICE>(
      size_type             numPoints,
      const double *        x,
      const double          r,
      const double          d,
      double *              out,
      utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(SmoothCutoffValueKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           x,
                           r,
                           d,
                           out);
    }

    template <>
    void
    smoothCutoffDerivative<utils::MemorySpace::DEVICE>(
      size_type             numPoints,
      const double *        x,
      const double          r,
      const double          d,
      const double          tolerance,
      double *              out,
      utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(SmoothCutoffDerivativeKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           x,
                           r,
                           d,
                           tolerance,
                           out);
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
