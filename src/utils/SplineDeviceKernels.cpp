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
 * Derived from spline.h by Tino Kluge (ttk448 at gmail.com), licensed under
 * GNU GPL v2+.  Adapted for DFT-EFE memorySpace templating.
 *
 * Batch (all-points) kernel launchers for Spline on DEVICE.
 * Scalar per-point helpers (SplineEvalKernel, SplineDerivKernel,
 * SPLINE_FIND_IDX, getValueDevice, getDerivDevice) live in
 * SplineDeviceKernels.h which is pulled in via Spline.h.
 *
 * @author Avirup Sircar
 */

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/Exceptions.h>
#  include <utils/Spline.h>   // also pulls in SplineDeviceKernels.h

namespace dftefe
{
  namespace utils
  {
    namespace
    {
      //-----------------------------------------------------------------------
      // Batch kernel: evaluate spline at nQuery points.
      // Calls SplineEvalKernel scalar helper from SplineDeviceKernels.h.
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        SplineEvalAllKernel,
        {
          for (size_type i = globalThreadId; i < nQuery;
               i += nThreadsPerBlock * nThreadBlock)
            y[i] = SplineEvalKernel(static_cast<double>(x[i]),
                                    knotX,
                                    knotY,
                                    coefB,
                                    coefC,
                                    coefD,
                                    nKnots,
                                    c0,
                                    isSubdivGrid,
                                    a_param,
                                    r_param,
                                    numSubDiv);
        },
        const size_type    nQuery,
        const double *     x,
        double *           y,
        const double *     knotX,
        const double *     knotY,
        const double *     coefB,
        const double *     coefC,
        const double *     coefD,
        const size_type    nKnots,
        const double       c0,
        const bool         isSubdivGrid,
        const double       a_param,
        const double       r_param,
        const dftefe::size_type numSubDiv);

      //-----------------------------------------------------------------------
      // Batch kernel: evaluate spline derivative of given order at nQuery pts.
      // Calls SplineDerivKernel scalar helper from SplineDeviceKernels.h.
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        SplineDerivAllKernel,
        {
          for (size_type i = globalThreadId; i < nQuery;
               i += nThreadsPerBlock * nThreadBlock)
            y[i] = SplineDerivKernel(derivOrder,
                                     static_cast<double>(x[i]),
                                     knotX,
                                     coefB,
                                     coefC,
                                     coefD,
                                     nKnots,
                                     c0,
                                     isSubdivGrid,
                                     a_param,
                                     r_param,
                                     numSubDiv);
        },
        const size_type    nQuery,
        const int          derivOrder,
        const double *     x,
        double *           y,
        const double *     knotX,
        const double *     coefB,
        const double *     coefC,
        const double *     coefD,
        const size_type    nKnots,
        const double       c0,
        const bool         isSubdivGrid,
        const double       a_param,
        const double       r_param,
        const dftefe::size_type numSubDiv);

    } // anonymous namespace

    //-------------------------------------------------------------------------
    // evalAll<DEVICE>
    //-------------------------------------------------------------------------
    template <>
    void
    Spline::evalAll<utils::MemorySpace::DEVICE>(
      size_type             n,
      const double *        x,
      double *              y,
      utils::deviceStream_t streamId) const
    {
      DFTEFE_LAUNCH_KERNEL(
        SplineEvalAllKernel,
        n / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        streamId,
        n,
        x,
        y,
        d_x_device.data(),
        d_y_device.data(),
        d_b_device.data(),
        d_c_device.data(),
        d_d_device.data(),
        static_cast<size_type>(d_x_device.size()),
        d_c0,
        d_isSubdivPowerLawGrid,
        d_a,
        d_r,
        d_numSubDiv);
    }

    //-------------------------------------------------------------------------
    // derivAll<DEVICE>
    //-------------------------------------------------------------------------
    template <>
    void
    Spline::derivAll<utils::MemorySpace::DEVICE>(
      size_type             n,
      int                   order,
      const double *        x,
      double *              y,
      utils::deviceStream_t streamId) const
    {
      DFTEFE_LAUNCH_KERNEL(
        SplineDerivAllKernel,
        n / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        streamId,
        n,
        order,
        x,
        y,
        d_x_device.data(),
        d_b_device.data(),
        d_c_device.data(),
        d_d_device.data(),
        static_cast<size_type>(d_x_device.size()),
        d_c0,
        d_isSubdivPowerLawGrid,
        d_a,
        d_r,
        d_numSubDiv);
    }

  } // namespace utils
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
