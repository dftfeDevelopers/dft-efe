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
 * Inline DFTEFE_HOST_DEVICE_FUNC definitions for
 * SphericalDataNumerical::Func<memorySpace>.
 * Included unconditionally by SphericalDataNumerical.h so every translation
 * unit (CPU or GPU) gets its own inline copy — no cross-TU linkage required.
 *
 * @author Avirup Sircar
 */

#ifndef dftefe_SphericalDataNumericalDeviceKernels_h
#define dftefe_SphericalDataNumericalDeviceKernels_h

#include <atoms/SphericalDataNumerical.h>
#include <atoms/SphericalHarmonicFunctions.h>
#include <atoms/SmoothCutoffFunctions.h>
#include <utils/Spline.h>
#include <utils/DeviceKernelLauncherHelpers.h>
#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    //=========================================================================
    // SphericalDataNumerical::Func<memorySpace> — default constructor
    //=========================================================================
    template <dftefe::utils::MemorySpace memorySpace>
    SphericalDataNumerical::Func<memorySpace>::Func()
      : d_radialSpline()
      , d_l(0)
      , d_m(0)
      , d_mEff(0)
      , d_constant(0.0)
      , d_cutoff(0.0)
      , d_smoothness(1.0)
      , d_polarAngleTolerance(1e-12)
      , d_cutoffTolerance(1e-12)
      , d_radiusTolerance(1e-10)
    {}

    //=========================================================================
    // SphericalDataNumerical::Func<memorySpace> — full constructor
    //=========================================================================
    template <dftefe::utils::MemorySpace memorySpace>
    SphericalDataNumerical::Func<memorySpace>::Func(
      utils::Spline::Func<memorySpace> radialSpline,
      int    l,
      int    m,
      int    mEff,
      double constant,
      double cutoff,
      double smoothness,
      double polarAngleTolerance,
      double cutoffTolerance,
      double radiusTolerance)
      : d_radialSpline(radialSpline)
      , d_l(l)
      , d_m(m)
      , d_mEff(mEff)
      , d_constant(constant)
      , d_cutoff(cutoff)
      , d_smoothness(smoothness)
      , d_polarAngleTolerance(polarAngleTolerance)
      , d_cutoffTolerance(cutoffTolerance)
      , d_radiusTolerance(radiusTolerance)
    {}

    //=========================================================================
    // SphericalDataNumerical::Func<memorySpace>::getValue
    //
    // Single-point evaluation of f(r)*Y_lm(theta,phi)*smoothCutoff(r).
    // Uses only standard math + Func::eval — callable from both host
    // and device code.
    //=========================================================================
    template <dftefe::utils::MemorySpace memorySpace>
    DFTEFE_HOST_DEVICE_FUNC double
    SphericalDataNumerical::Func<memorySpace>::getValue(
      const double *point,
      const double *origin) const
    {
      double shifted[3] = {point[0] - origin[0],
                           point[1] - origin[1],
                           point[2] - origin[2]};

      double r, theta, phi;
      convertCartesianToSpherical(shifted, r, theta, phi, d_polarAngleTolerance);

      if (r > d_cutoff + d_cutoff / d_smoothness)
        return 0.0;

      const double radialValue = d_radialSpline.eval(r);
      const double cutoffValue =
        smoothCutoffValue(r, d_cutoff, d_smoothness);
      const double plmVal = plm(d_l, d_mEff, cos(theta));
      const double qm  = Qm(d_m, phi);

      return radialValue * cutoffValue * d_constant * plmVal * qm;
    }

    //=========================================================================
    // SphericalDataNumerical::Func<memorySpace>::getGradientValue
    //
    // Single-point gradient of f(r)*Y_lm(theta,phi)*smoothCutoff(r).
    // Writes 3 Cartesian components into grad[0..2].
    //=========================================================================
    template <dftefe::utils::MemorySpace memorySpace>
    DFTEFE_HOST_DEVICE_FUNC void
    SphericalDataNumerical::Func<memorySpace>::getGradientValue(
      const double *point,
      const double *origin,
      double *      grad) const
    {
      double shifted[3] = {point[0] - origin[0],
                           point[1] - origin[1],
                           point[2] - origin[2]};

      double r, theta, phi;
      convertCartesianToSpherical(shifted, r, theta, phi, d_polarAngleTolerance);

      if (r > d_cutoff + d_cutoff / d_smoothness || r < d_radiusTolerance)
        {
          grad[0] = 0.0;
          grad[1] = 0.0;
          grad[2] = 0.0;
          return;
        }

      const double cosTheta = cos(theta);
      const double sinTheta = sin(theta);
      const double cosPhi   = cos(phi);
      const double sinPhi   = sin(phi);

      const double radialValue = d_radialSpline.eval(r);
      const double radialDeriv = d_radialSpline.deriv(1, r);
      const double cutoffValue = smoothCutoffValue(r, d_cutoff, d_smoothness);
      const double cutoffDeriv =
        smoothCutoffDerivative(r, d_cutoff, d_smoothness, d_cutoffTolerance);

      const double plmVal  = plm(d_l, d_mEff, cosTheta);
      const double dPlmVal = dplmDTheta(d_l, d_mEff, cosTheta);
      const double qm      = Qm(d_m, phi);

      const double Ylm        = d_constant * plmVal * qm;
      const double dYlmDTheta = d_constant * dPlmVal * qm;

      double dYlmDPhiBysinTheta = 0.0;
      if (d_m != 0)
        {
          const double d2PlmVal = d2plmDTheta2(d_l, d_mEff, cosTheta);
          const double dqm      = dQmDPhi(d_m, phi);
          dYlmDPhiBysinTheta =
            d_constant *
            (sinTheta * d2PlmVal + cosTheta * dPlmVal +
             sinTheta * (double)(d_l * (d_l + 1)) * plmVal) *
            (1.0 / ((double)d_m * (double)d_m)) * dqm;
        }

      double dValueDR =
        (radialDeriv * cutoffValue + cutoffDeriv * radialValue) * Ylm;
      double dValueDThetaByr =
        (radialValue / r) * cutoffValue * dYlmDTheta;
      double dValueDPhiByrsinTheta =
        (radialValue / r) * cutoffValue * dYlmDPhiBysinTheta;

      if (r < 1e-4 && d_l > 0)
        {
          dValueDThetaByr       = dValueDR * dYlmDTheta;
          dValueDPhiByrsinTheta = dValueDR * dYlmDPhiBysinTheta;
        }

      grad[0] = dValueDR * (sinTheta * cosPhi) +
                dValueDThetaByr * (cosTheta * cosPhi) -
                sinPhi * dValueDPhiByrsinTheta;
      grad[1] = dValueDR * (sinTheta * sinPhi) +
                dValueDThetaByr * (cosTheta * sinPhi) +
                cosPhi * dValueDPhiByrsinTheta;
      grad[2] = dValueDR * cosTheta - dValueDThetaByr * sinTheta;
    }

  } // namespace atoms
} // namespace dftefe

#endif // dftefe_SphericalDataNumericalDeviceKernels_h
