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
 */

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/Exceptions.h>
#  include <utils/Spline.h> // also pulls in SplineDeviceKernels.h
#  include <utils/MemoryStorage.h>
#  include <atoms/SphericalDataMixed.h>
#  include <atoms/SphericalHarmonicFunctions.h> // also pulls in SphericalHarmonicFunctionsDeviceKernels.h
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      //-----------------------------------------------------------------------
      // SHIFT kernel: shiftedPoints[i] = points[i] - origin
      // Needed to prepare input for funcAfterRadialGrid.evalDevice,
      // which expects coordinates relative to SphericalDataMixed's origin.
      // All kernels in getValueDevice use stream 0 to serialise with
      // evalDevice.
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        SphericalDataMixedShiftKernel,
        {
          for (size_type i = globalThreadId; i < numPoints;
               i += nThreadsPerBlock * nThreadBlock)
            {
              shiftedPoints[3 * i]     = points[3 * i] - origin[0];
              shiftedPoints[3 * i + 1] = points[3 * i + 1] - origin[1];
              shiftedPoints[3 * i + 2] = points[3 * i + 2] - origin[2];
            }
        },
        const size_type numPoints,
        const double *  points,
        const double *  origin,
        double *        shiftedPoints);

      //-----------------------------------------------------------------------
      // VALUE kernel
      // Shift computed inline (like SphericalDataNumericalValueKernel).
      // For r <= lastRadialGridPoint: spline interpolation * Ylm.
      // For r >  lastRadialGridPoint: analyticalVals[i] (precomputed on device
      //   via funcAfterRadialGrid.eval<DEVICE>) * Ylm.
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        SphericalDataMixedValueKernel,
        {
          for (size_type i = globalThreadId; i < numPoints;
               i += nThreadsPerBlock * nThreadBlock)
            {
              double shifted[3];
              shifted[0] = points[3 * i] - origin[0];
              shifted[1] = points[3 * i + 1] - origin[1];
              shifted[2] = points[3 * i + 2] - origin[2];

              double r;
              double theta;
              double phi;
              convertCartesianToSpherical(
                shifted, r, theta, phi, polarAngleTolerance);

              const double cosTheta = cos(theta);
              const double plmVal   = plm(l, mEff, cosTheta);
              const double qm       = Qm(m, phi);
              const double Ylm      = ylmConstant * plmVal * qm;

              if (r <= lastRadialGridPoint)
                out[i] = spline.eval(r) * Ylm;
              else
                out[i] = analyticalVals[i] * Ylm;
            }
        },
        const size_type numPoints,
        const double *  points,
        const double *  origin,
        const double    polarAngleTolerance,
        const int       l,
        const int       m,
        const int       mEff,
        const double    ylmConstant,
        const double    lastRadialGridPoint,
        const utils::Spline::Func<utils::MemorySpace::DEVICE> spline,
        const double *                                        analyticalVals,
        double *                                              out);

    } // anonymous namespace

    //=========================================================================
    // getValueDevice
    //=========================================================================
    void
    SphericalDataMixed::getValueDevice(const size_type       numPoints,
                                       const double *        points,
                                       const double *        origin,
                                       double *              out,
                                       utils::deviceStream_t streamId)
    {
      utils::MemoryStorage<double, utils::MemorySpace::DEVICE> shiftedPoints(
        numPoints * 3);
      utils::MemoryStorage<double, utils::MemorySpace::DEVICE> analyticalVals(
        numPoints);

      // Step 1: compute shiftedPoints = points - origin (stream 0)
      DFTEFE_LAUNCH_KERNEL(SphericalDataMixedShiftKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           0,
                           numPoints,
                           points,
                           origin,
                           shiftedPoints.data());

      // Step 2: evaluate funcAfterRadialGrid on device for all points.
      // evalDevice internally uses stream 0, so it serialises with Step 1.
      d_funcAfterRadialGrid.eval<utils::MemorySpace::DEVICE>(
        numPoints, shiftedPoints.data(), analyticalVals.data());

      // Step 3: blend spline and analytical values, multiply by Ylm (stream 0)
      const int    l          = d_qNumbers[1];
      const int    m          = d_qNumbers[2];
      const int    mEff       = std::abs(m);
      const double ylmConst   = Clm(l, m) * Dm(m);
      const double lastRPoint = d_radialPoints.back();

      DFTEFE_LAUNCH_KERNEL(SphericalDataMixedValueKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           0,
                           numPoints,
                           points,
                           origin,
                           d_polarAngleTolerance,
                           l,
                           m,
                           mEff,
                           ylmConst,
                           lastRPoint,
                           d_spline->getFunc<utils::MemorySpace::DEVICE>(),
                           analyticalVals.data(),
                           out);
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
