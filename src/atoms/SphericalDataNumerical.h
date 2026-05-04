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

#ifndef dftefeSphericalDataNumerical_h
#define dftefeSphericalDataNumerical_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/DeviceTypeConfig.h>
#include <vector>
#include <utils/Point.h>
#include <atoms/SphericalData.h>
#include <utils/Spline.h>
#include <memory>
#include <utils/Point.h>
#include <atoms/Defaults.h>
#include <atoms/SphericalHarmonicFunctions.h>
#include <utils/DeviceKernelLauncherHelpers.h>

namespace dftefe
{
  namespace atoms
  {
    class SphericalDataNumerical : public SphericalData
    {
    public:
      // Lightweight functor holding all data needed for single-point
      // evaluation in a specific memory space.  Obtained on the host via
      // getFunc<MemorySpace>(), then passed by value into host or device
      // (CUDA/HIP/SYCL) kernels.
      // Func<HOST>   — host spline pointers, callable from host code.
      // Func<DEVICE> — device spline pointers, callable from device kernels.
      template <dftefe::utils::MemorySpace memorySpace>
      class Func
      {
      public:
        Func();

        Func(utils::Spline::Func<memorySpace> radialSpline,
             int    l,
             int    m,
             int    mEff,
             double constant,
             double cutoff,
             double smoothness,
             double polarAngleTolerance,
             double cutoffTolerance,
             double radiusTolerance);

        DFTEFE_HOST_DEVICE_FUNC double
        getValue(const double *point, const double *origin) const;

        DFTEFE_HOST_DEVICE_FUNC void
        getGradientValue(const double *point,
                         const double *origin,
                         double *      grad) const;

      private:
        utils::Spline::Func<memorySpace> d_radialSpline;
        int    d_l, d_m, d_mEff;
        double d_constant, d_cutoff, d_smoothness, d_polarAngleTolerance;
        double d_cutoffTolerance, d_radiusTolerance;
      };

      SphericalDataNumerical(
        const std::vector<int>            qNumbers,
        const std::vector<double>         radialPoints,
        const std::vector<double>         radialValues,
        const double                      cutoff,
        const double                      smoothness,
        const SphericalHarmonicFunctions &sphericalHarmonicFunc,
        const double polarAngleTolerance = SphericalDataDefaults::POL_ANG_TOL,
        const double cutoffTolerance     = SphericalDataDefaults::CUTOFF_TOL,
        const double radiusTolerance     = SphericalDataDefaults::RADIUS_TOL,
        const size_type dim              = SphericalDataDefaults::DEFAULT_DIM);

      ~SphericalDataNumerical() = default;

      void
      initSpline();

      std::vector<double>
      getValue(const std::vector<utils::Point> &point,
               const utils::Point &             origin) override;

      std::vector<double>
      getGradientValue(const std::vector<utils::Point> &point,
                       const utils::Point &             origin) override;

      std::vector<double>
      getHessianValue(const std::vector<utils::Point> &point,
                      const utils::Point &             origin) override;

      double
      getValue(const utils::Point &point, const utils::Point &origin) override;

      std::vector<double>
      getGradientValue(const utils::Point &point,
                       const utils::Point &origin) override;

      std::vector<double>
      getHessianValue(const utils::Point &point,
                      const utils::Point &origin) override;

#ifdef DFTEFE_WITH_DEVICE
      void
      getValueDevice(const size_type numPoints, 
              const double    *points, 
              const double    *origin,
              double *out,
              utils::deviceStream_t  streamId = utils::defaultStream) override;

      void
      getGradientValueDevice(const size_type numPoints, 
              const double    *points, 
              const double    *origin,
              double *out,
              utils::deviceStream_t  streamId = utils::defaultStream) override;

      void
      getHessianValueDevice(const size_type numPoints, 
              const double    *points, 
              const double    *origin,
              double *out,
              utils::deviceStream_t  streamId = utils::defaultStream) override;
#endif              

      void
      getValue(const size_type numPoints,
               const double *  points,
               const double *  origin,
               double *        out) override;

      void
      getGradientValue(const size_type numPoints,
                       const double *  points,
                       const double *  origin,
                       double *        out) override;

      void
      getHessianValue(const size_type numPoints,
                      const double *  points,
                      const double *  origin,
                      double *        out) override;

      std::vector<double>
      getRadialValue(const std::vector<double> &r) override;

      std::vector<double>
      getAngularValue(const std::vector<double> &r,
                      const std::vector<double> &theta,
                      const std::vector<double> &phi) override;

      std::vector<double>
      getRadialDerivative(const std::vector<double> &r) override;

      std::vector<std::vector<double>>
      getAngularDerivative(const std::vector<double> &r,
                           const std::vector<double> &theta,
                           const std::vector<double> &phi) override;

      std::vector<int>
      getQNumbers() const override;

      double
      getCutoff() const override;

      double
      getSmoothness() const override;

      // Returns a Func for the given memory space.
      // HOST:   fills from host std::vector spline data.
      // DEVICE: fills from device MemoryStorage spline data.
      // Both are host-callable only — call before launching a kernel.
      template <dftefe::utils::MemorySpace memorySpace>
      Func<memorySpace>
      getFunc() const;

    private:
      std::vector<int>                     d_qNumbers;
      std::vector<double>                  d_radialPoints;
      std::vector<double>                  d_radialValues;
      double                               d_cutoff;
      double                               d_smoothness;
      std::shared_ptr<const utils::Spline> d_spline;
      double                               d_polarAngleTolerance;
      double                               d_cutoffTolerance;
      double                               d_radiusTolerance;
      size_type                            d_dim;

      const SphericalHarmonicFunctions &d_sphericalHarmonicFunc;
    };

  } // end of namespace atoms
} // end of namespace dftefe

#include <atoms/SphericalDataNumericalKernels.h>

#endif // dftefeSphericalDataNumerical_h
