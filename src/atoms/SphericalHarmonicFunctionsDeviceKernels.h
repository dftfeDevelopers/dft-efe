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
 * Inline DFTEFE_DEVICE_FUNC definitions for SphericalHarmonicFunctions device
 * evaluation.  Included by SphericalHarmonicFunctions.h so that every
 * translation unit gets its own inline copy — no cross-TU __device__ linkage.
 *
 * @author Avirup Sircar
 */

#ifndef dftefe_SphericalHarmonicFunctionsDeviceKernels_h
#define dftefe_SphericalHarmonicFunctionsDeviceKernels_h

#ifdef DFTEFE_WITH_DEVICE
#  include <atoms/SphericalHarmonicFunctions.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      //-----------------------------------------------------------------------
      // RlmDevice — recursive device helper for negative-m scaling.
      // Note: __forceinline__ on a recursive function is advisory; the
      // compiler will not fully inline but the function is still callable.
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      RlmDevice(const int l, const int m)
      {
        if (m == 0)
          return 1.0;
        return RlmDevice(l, m - 1) / ((l - m + 1.0) * (l + m));
      }

      //-----------------------------------------------------------------------
      // deviceCartesianToSpherical
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC void
      deviceCartesianToSpherical(const double *x,
                                 double &      r,
                                 double &      theta,
                                 double &      phi,
                                 double        polarAngleTolerance)
      {
        double px = x[0];
        double py = x[1];
        double pz = x[2];
        r         = sqrt(px * px + py * py + pz * pz);
        if (r == 0.0)
          {
            theta = 0.0;
            phi   = 0.0;
          }
        else
          {
            theta = acos(pz / r);
            if (fabs(theta - 0.0) >= polarAngleTolerance &&
                fabs(theta - M_PI) >= polarAngleTolerance)
              phi = atan2(py, px);
            else
              phi = 0.0;
          }
      }

      //-----------------------------------------------------------------------
      // deviceQm
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      deviceQm(const int m, const double phi)
      {
        double v = 0.0;
        if (m > 0)
          v = cos((double)m * phi);
        else if (m == 0)
          v = 1.0;
        else
          v = sin((double)(-m) * phi);
        return v;
      }

      //-----------------------------------------------------------------------
      // deviceDQmDPhi
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      deviceDQmDPhi(const int m, const double phi)
      {
        double v;
        if (m > 0)
          v = -(double)m * sin((double)m * phi);
        else if (m == 0)
          v = 0.0;
        else
          v = (double)(-m) * cos((double)(-m) * phi);
        return v;
      }

      //-----------------------------------------------------------------------
      // devicePlm — associated Legendre polynomial P_l^mEff(cos theta),
      // mEff >= 0.
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      devicePlm(int l, int mEff, double cosTheta)
      {
        if (mEff > l)
          return 0.0;
        double somx2 = sqrt(1.0 - cosTheta * cosTheta);
        double cxM   = 1.0;
        double fact  = 1.0;
        for (int i = 0; i < mEff; i++)
          {
            cxM  = -cxM * fact * somx2;
            fact = fact + 2.0;
          }
        double cx = cxM;
        if (mEff != l)
          {
            double cxMPlus1   = cosTheta * (2 * mEff + 1) * cxM;
            cx                = cxMPlus1;
            double cxPrev     = cxMPlus1;
            double cxPrevPrev = cxM;
            for (int jj = mEff + 2; jj < l + 1; jj++)
              {
                cx = ((2 * jj - 1) * cosTheta * cxPrev +
                      (-jj - mEff + 1) * cxPrevPrev) /
                     (jj - mEff);
                cxPrevPrev = cxPrev;
                cxPrev     = cx;
              }
          }
        return ((mEff % 2 == 0) ? 1.0 : -1.0) * cx;
      }

      //-----------------------------------------------------------------------
      // deviceDPlmDTheta — first derivative d(P_l^mEff)/d(theta).
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      deviceDPlmDTheta(int l, int mEff, double cosTheta)
      {
        if (mEff > l)
          return 0.0;
        if (l == 0)
          return 0.0;
        if (mEff == 0)
          return -1.0 * devicePlm(l, 1, cosTheta);
        if (mEff == l)
          return (double)l * devicePlm(l, l - 1, cosTheta);
        double term1 =
          (double)((l + mEff) * (l - mEff + 1)) *
          devicePlm(l, mEff - 1, cosTheta);
        double term2 = devicePlm(l, mEff + 1, cosTheta);
        return 0.5 * (term1 - term2);
      }

      //-----------------------------------------------------------------------
      // deviceD2PlmDTheta2 — second derivative d^2(P_l^mEff)/d(theta)^2.
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      deviceD2PlmDTheta2(int l, int mEff, double cosTheta)
      {
        if (mEff > l)
          return 0.0;
        if (l == 0)
          return 0.0;
        if (mEff == 0)
          return -1.0 * deviceDPlmDTheta(l, 1, cosTheta);
        if (mEff == l)
          return (double)l * deviceDPlmDTheta(l, l - 1, cosTheta);
        double term1 = (double)((l + mEff) * (l - mEff + 1)) *
                       deviceDPlmDTheta(l, mEff - 1, cosTheta);
        double term2 = deviceDPlmDTheta(l, mEff + 1, cosTheta);
        return 0.5 * (term1 - term2);
      }

    } // anonymous namespace

    //=========================================================================
    // convertCartesianToSphericalDevice
    //=========================================================================
    DFTEFE_DEVICE_FUNC void
    convertCartesianToSphericalDevice(const double *x,
                                      double &      r,
                                      double &      theta,
                                      double &      phi,
                                      double        polarAngleTolerance)
    {
      deviceCartesianToSpherical(x, r, theta, phi, polarAngleTolerance);
    }

    //=========================================================================
    // QmDevice
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    QmDevice(const int m, const double phi)
    {
      return deviceQm(m, phi);
    }

    //=========================================================================
    // dQmDPhiDevice
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    dQmDPhiDevice(const int m, const double phi)
    {
      return deviceDQmDPhi(m, phi);
    }

    //=========================================================================
    // SphericalHarmonicFunctions::PlmDevice
    // Always uses analytical recurrence — the spline path (d_assocLegendreSpline)
    // requires std::vector/shared_ptr which are __host__ only and cannot be
    // used inside __device__ functions.
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    SphericalHarmonicFunctions::PlmDevice(const int    l,
                                          const int    m,
                                          const double theta) const
    {
      const int    mEff   = (m < 0) ? -m : m;
      const double factor = (m < 0) ? pow(-1.0, m) * RlmDevice(l, mEff) : 1.0;
      if (mEff > l)
        return 0.0;
      return factor * devicePlm(l, mEff, cos(theta));
    }

    //=========================================================================
    // SphericalHarmonicFunctions::dPlmDThetaDevice
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    SphericalHarmonicFunctions::dPlmDThetaDevice(const int    l,
                                                  const int    m,
                                                  const double theta) const
    {
      const int absm = (m < 0) ? -m : m;
      if (absm > l || l == 0)
        return 0.0;
      const int    mEff   = absm;
      const double factor = (m < 0) ? pow(-1.0, m) * RlmDevice(l, mEff) : 1.0;
      return factor * deviceDPlmDTheta(l, mEff, cos(theta));
    }

    //=========================================================================
    // SphericalHarmonicFunctions::d2PlmDTheta2Device
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    SphericalHarmonicFunctions::d2PlmDTheta2Device(const int    l,
                                                    const int    m,
                                                    const double theta) const
    {
      const int absm = (m < 0) ? -m : m;
      if (absm > l || l == 0)
        return 0.0;
      const int    mEff   = absm;
      const double factor = (m < 0) ? pow(-1.0, m) * RlmDevice(l, mEff) : 1.0;
      return factor * deviceD2PlmDTheta2(l, mEff, cos(theta));
    }

  } // namespace atoms
} // namespace dftefe

#endif // DFTEFE_WITH_DEVICE
#endif // dftefe_SphericalHarmonicFunctionsDeviceKernels_h
