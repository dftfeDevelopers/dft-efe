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
#  include <utils/Spline.h>   // also pulls in SplineDeviceKernels.h
#  include <atoms/SphericalDataNumerical.h>
#  include <atoms/SphericalHarmonicFunctions.h>   // also pulls in SphericalHarmonicFunctionsDeviceKernels.h
#  include <atoms/SmoothCutoffFunctions.h>         // also pulls in SmoothCutoffFunctionsDeviceKernels.h
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      //-----------------------------------------------------------------------
      // VALUE kernel
      // spline is passed by value (Func<DEVICE>, safe for GPU).
      // All spherical harmonic / cutoff functions called directly — no object ptr.
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        SphericalDataNumericalValueKernel,
        {
          for (size_type i = globalThreadId; i < numPoints;
               i += nThreadsPerBlock * nThreadBlock)
            {
              double shifted[3];
              shifted[0] = points[3 * i]     - origin[0];
              shifted[1] = points[3 * i + 1] - origin[1];
              shifted[2] = points[3 * i + 2] - origin[2];
              double r;
              double theta;
              double phi;
              convertCartesianToSpherical(
                shifted, r, theta, phi, polarAngleTolerance);

              if (r > cutoff + cutoff / smoothness)
                {
                  out[i] = 0.0;
                  continue;
                }

              const double radialValue = spline.eval(r);
              const double cutoffValue =
                smoothCutoffValue(r, cutoff, smoothness);
              const double cosTheta = cos(theta);
              const double plmVal   = plm(l, mEff, cosTheta);
              const double qm       = Qm(m, phi);

              out[i] = radialValue * cutoffValue * constant * plmVal * qm;
            }
        },
        const size_type             numPoints,
        const double *              points,
        const double *              origin,
        const double                cutoff,
        const double                smoothness,
        const double                polarAngleTolerance,
        const int                   l,
        const int                   m,
        const int                   mEff,
        const double                constant,
        const utils::Spline::Func<utils::MemorySpace::DEVICE> spline,
        double *                    out);

      //-----------------------------------------------------------------------
      // GRADIENT kernel  (output: 3*numPoints, layout [gx0,gy0,gz0,...])
      // Same design: Func<DEVICE> by value, direct function calls.
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        SphericalDataNumericalGradientKernel,
        {
          for (size_type i = globalThreadId; i < numPoints;
               i += nThreadsPerBlock * nThreadBlock)
            {
              double shifted[3];
              shifted[0] = points[3 * i]     - origin[0];
              shifted[1] = points[3 * i + 1] - origin[1];
              shifted[2] = points[3 * i + 2] - origin[2];
              double r;
              double theta;
              double phi;
              convertCartesianToSpherical(
                shifted, r, theta, phi, polarAngleTolerance);

              if (r > cutoff + cutoff / smoothness || r < radiusTolerance)
                {
                  out[3 * i]     = 0.0;
                  out[3 * i + 1] = 0.0;
                  out[3 * i + 2] = 0.0;
                  continue;
                }

              const double cosTheta = cos(theta);
              const double sinTheta = sin(theta);
              const double cosPhi   = cos(phi);
              const double sinPhi   = sin(phi);

              const double radialValue = spline.eval(r);
              const double radialDeriv = spline.deriv(1, r);
              const double cutoffValue =
                smoothCutoffValue(r, cutoff, smoothness);
              const double cutoffDerv = smoothCutoffDerivative(
                r, cutoff, smoothness, cutoffTolerance);

              const double plmVal  = plm(l, mEff, cosTheta);
              const double dPlmVal = dplmDTheta(l, mEff, cosTheta);
              const double qm      = Qm(m, phi);

              const double Ylm        = constant * plmVal * qm;
              const double dYlmDTheta = constant * dPlmVal * qm;

              double dYlmDPhiBysinTheta = 0.0;
              if (m != 0)
                {
                  const double d2PlmVal = d2plmDTheta2(l, mEff, cosTheta);
                  const double dqm      = dQmDPhi(m, phi);
                  dYlmDPhiBysinTheta =
                    constant *
                    (sinTheta * d2PlmVal + cosTheta * dPlmVal +
                     sinTheta * (double)(l * (l + 1)) * plmVal) *
                    (1.0 / ((double)m * (double)m)) * dqm;
                }

              double dValueDR =
                (radialDeriv * cutoffValue + cutoffDerv * radialValue) * Ylm;
              double dValueDThetaByr =
                (radialValue / r) * cutoffValue * dYlmDTheta;
              double dValueDPhiByrsinTheta =
                (radialValue / r) * cutoffValue * dYlmDPhiBysinTheta;

              if (r < 1e-4 && l > 0)
                {
                  dValueDThetaByr       = dValueDR * dYlmDTheta;
                  dValueDPhiByrsinTheta = dValueDR * dYlmDPhiBysinTheta;
                }

              out[3 * i]     = dValueDR * (sinTheta * cosPhi) +
                               dValueDThetaByr * (cosTheta * cosPhi) -
                               sinPhi * dValueDPhiByrsinTheta;
              out[3 * i + 1] = dValueDR * (sinTheta * sinPhi) +
                               dValueDThetaByr * (cosTheta * sinPhi) +
                               cosPhi * dValueDPhiByrsinTheta;
              out[3 * i + 2] =
                dValueDR * cosTheta - dValueDThetaByr * sinTheta;
            }
        },
        const size_type               numPoints,
        const double *                points,
        const double *                origin,
        const double                  cutoff,
        const double                  smoothness,
        const double                  polarAngleTolerance,
        const double                  cutoffTolerance,
        const double                  radiusTolerance,
        const int                     l,
        const int                     m,
        const int                     mEff,
        const double                  constant,
        const utils::Spline::Func<utils::MemorySpace::DEVICE> spline,
        double *                      out);

    } // anonymous namespace

    //=========================================================================
    // getValueDevice
    //=========================================================================
    void
    SphericalDataNumerical::getValueDevice(
      const size_type       numPoints,
      const double *        points,
      const double *        origin,
      double *              out,
      utils::deviceStream_t streamId)
    {
      const int    l        = d_qNumbers[1];
      const int    m        = d_qNumbers[2];
      const int    mEff     = std::abs(m);
      const double constant = Clm(l, m) * Dm(m);

      DFTEFE_LAUNCH_KERNEL(SphericalDataNumericalValueKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           points,
                           origin,
                           d_cutoff,
                           d_smoothness,
                           d_polarAngleTolerance,
                           l,
                           m,
                           mEff,
                           constant,
                           d_spline->getFunc<utils::MemorySpace::DEVICE>(),
                           out);
    }

    //=========================================================================
    // getGradientValueDevice
    //=========================================================================
    void
    SphericalDataNumerical::getGradientValueDevice(
      const size_type       numPoints,
      const double *        points,
      const double *        origin,
      double *              out,
      utils::deviceStream_t streamId)
    {
      const int    l        = d_qNumbers[1];
      const int    m        = d_qNumbers[2];
      const int    mEff     = std::abs(m);
      const double constant = Clm(l, m) * Dm(m);

      DFTEFE_LAUNCH_KERNEL(SphericalDataNumericalGradientKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           points,
                           origin,
                           d_cutoff,
                           d_smoothness,
                           d_polarAngleTolerance,
                           d_cutoffTolerance,
                           d_radiusTolerance,
                           l,
                           m,
                           mEff,
                           constant,
                           d_spline->getFunc<utils::MemorySpace::DEVICE>(),
                           out);
    }

    //=========================================================================
    // getHessianValueDevice — not implemented
    //=========================================================================
    void
    SphericalDataNumerical::getHessianValueDevice(
      const size_type       numPoints,
      const double *        points,
      const double *        origin,
      double *              out,
      utils::deviceStream_t streamId)
    {
      utils::throwException(
        false,
        "getHessianValueDevice not implemented for SphericalDataNumerical.");
    }

    //=========================================================================
    // getFunc<DEVICE>
    //=========================================================================
    template <>
    SphericalDataNumerical::Func<utils::MemorySpace::DEVICE>
    SphericalDataNumerical::getFunc<utils::MemorySpace::DEVICE>() const
    {
      const int l = d_qNumbers[1];
      const int m = d_qNumbers[2];
      return Func<utils::MemorySpace::DEVICE>(
        d_spline->getFunc<utils::MemorySpace::DEVICE>(),
        l, m, std::abs(m),
        Clm(l, m) * Dm(m),
        d_cutoff,
        d_smoothness,
        d_polarAngleTolerance,
        d_cutoffTolerance,
        d_radiusTolerance);
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
