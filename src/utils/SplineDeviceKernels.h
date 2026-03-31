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
 * Inline DFTEFE_DEVICE_FUNC definitions for Spline device evaluation.
 * Included by Spline.h (inside #ifdef DFTEFE_WITH_DEVICE) so that every
 * CUDA/HIP/SYCL translation unit that includes Spline.h gets its own
 * inline copy — no cross-TU __device__ linkage required.
 *
 * @author Avirup Sircar
 */

#ifndef dftefe_SplineDeviceKernels_h
#define dftefe_SplineDeviceKernels_h

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/Spline.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <cmath>

namespace dftefe
{
  namespace utils
  {
    namespace
    {
      //-----------------------------------------------------------------------
      // splineFindIdx — device index search: returns the knot index i such
      // that knotX[i] <= x < knotX[i+1] (or the boundary indices).
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC size_type
      splineFindIdx(const double       xi,
                    const double *     knotX,
                    const size_type    nKnots,
                    const bool         isSubdivGrid,
                    const double       a_param,
                    const double       r_param,
                    const unsigned int numSubDiv)
      {
        if (isSubdivGrid)
          {
            if (xi > knotX[nKnots - 1])
              return nKnots - 1;
            if (xi < knotX[0])
              return 0;
            double       rDiff = r_param - 1.0;
            unsigned int n_gp  = 0;
            unsigned int subId = 0;
            if (rDiff < 1e-6 && rDiff > -1e-6)
              {
                subId = static_cast<unsigned int>(xi / a_param);
              }
            else
              {
                n_gp = static_cast<unsigned int>(
                  log(xi * rDiff / a_param + 1.0) / log(r_param));
                double segStart =
                  a_param *
                  (pow(r_param, static_cast<double>(n_gp)) - 1.0) / rDiff;
                double segWidth =
                  a_param * pow(r_param, static_cast<double>(n_gp));
                subId = static_cast<unsigned int>(
                  static_cast<double>(numSubDiv) * (xi - segStart) / segWidth);
              }
            size_type idx =
              static_cast<size_type>(n_gp * numSubDiv + subId);
            return (idx >= nKnots) ? nKnots - 1 : idx;
          }
        else
          {
            if (xi <= knotX[0])
              return 0;
            if (xi >= knotX[nKnots - 1])
              return nKnots - 1;
            size_type lo = 0;
            size_type hi = nKnots - 1;
            while (hi - lo > 1)
              {
                size_type mid = (lo + hi) / 2;
                if (knotX[mid] <= xi)
                  lo = mid;
                else
                  hi = mid;
              }
            return lo;
          }
      }

      //-----------------------------------------------------------------------
      // Scalar device helper: evaluate spline at a single point x.
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      SplineEvalKernel(const double       x,
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
                       const unsigned int numSubDiv)
      {
        double          xi  = x;
        const size_type idx =
          splineFindIdx(xi, knotX, nKnots, isSubdivGrid, a_param, r_param, numSubDiv);
        double h = xi - knotX[idx];
        double interpol;
        if (xi < knotX[0])
          interpol = (c0 * h + coefB[0]) * h + knotY[0];
        else if (xi > knotX[nKnots - 1])
          interpol =
            (coefC[nKnots - 1] * h + coefB[nKnots - 1]) * h +
            knotY[nKnots - 1];
        else
          interpol =
            ((coefD[idx] * h + coefC[idx]) * h + coefB[idx]) * h +
            knotY[idx];
        return interpol;
      }

      //-----------------------------------------------------------------------
      // Scalar device helper: evaluate spline derivative at a single point x.
      //-----------------------------------------------------------------------
      DFTEFE_DEVICE_FUNC double
      SplineDerivKernel(const int          derivOrder,
                        const double       x,
                        const double *     knotX,
                        const double *     coefB,
                        const double *     coefC,
                        const double *     coefD,
                        const size_type    nKnots,
                        const double       c0,
                        const bool         isSubdivGrid,
                        const double       a_param,
                        const double       r_param,
                        const unsigned int numSubDiv)
      {
        double          xi  = x;
        const size_type idx =
          splineFindIdx(xi, knotX, nKnots, isSubdivGrid, a_param, r_param, numSubDiv);
        double h        = xi - knotX[idx];
        double interpol = 0.0;
        if (xi < knotX[0])
          {
            if (derivOrder == 1)
              interpol = 2.0 * c0 * h + coefB[0];
            else if (derivOrder == 2)
              interpol = 2.0 * c0;
          }
        else if (xi > knotX[nKnots - 1])
          {
            if (derivOrder == 1)
              interpol =
                2.0 * coefC[nKnots - 1] * h + coefB[nKnots - 1];
            else if (derivOrder == 2)
              interpol = 2.0 * coefC[nKnots - 1];
          }
        else
          {
            if (derivOrder == 1)
              interpol =
                (3.0 * coefD[idx] * h + 2.0 * coefC[idx]) * h +
                coefB[idx];
            else if (derivOrder == 2)
              interpol = 6.0 * coefD[idx] * h + 2.0 * coefC[idx];
            else if (derivOrder == 3)
              interpol = 6.0 * coefD[idx];
          }
        return interpol;
      }

    } // anonymous namespace

    //=========================================================================
    // Free device functions — take a SplineDeviceView by value (no object ptr).
    // Call these from kernels instead of spline->getValueDevice / getDerivDevice.
    //=========================================================================
    DFTEFE_DEVICE_FUNC double
    SplineEvalDevice(const SplineDeviceView v, double x)
    {
      return SplineEvalKernel(x,
                              v.x,
                              v.y,
                              v.b,
                              v.c,
                              v.d,
                              v.n,
                              v.c0,
                              v.isSubdivGrid,
                              v.a,
                              v.r,
                              v.numSubDiv);
    }

    DFTEFE_DEVICE_FUNC double
    SplineDerivDevice(const SplineDeviceView v, int order, double x)
    {
      return SplineDerivKernel(order,
                               x,
                               v.x,
                               v.b,
                               v.c,
                               v.d,
                               v.n,
                               v.c0,
                               v.isSubdivGrid,
                               v.a,
                               v.r,
                               v.numSubDiv);
    }

  } // namespace utils
} // namespace dftefe

#endif // DFTEFE_WITH_DEVICE
#endif // dftefe_SplineDeviceKernels_h
