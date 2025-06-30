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

#ifndef dftefeSphericalDataAnalytical_h
#define dftefeSphericalDataAnalytical_h

#include <utils/TypeConfig.h>
#include <vector>
#include <utils/Point.h>
#include <atoms/SphericalData.h>
#include <utils/Spline.h>
#include <utils/ScalarSpatialFunction.h>
#include <memory>
#include <utils/Point.h>
#include <atoms/Defaults.h>
#include <atoms/SphericalHarmonicFunctions.h>

namespace dftefe
{
  namespace atoms
  {
    class SphericalDataAnalytical : public SphericalData
    {
    public:
    //Assumption ScalarSpatialFunctionReal is spherically symmetric . TODO : enforce this
      SphericalDataAnalytical(
        const std::vector<int>                  qNumbers,
        const utils::ScalarSpatialFunctionReal &function,
        const double                            cutoff,
        const double                            smoothness,
        const SphericalHarmonicFunctions &      sphericalHarmonicFunc,
        const double polarAngleTolerance = SphericalDataDefaults::POL_ANG_TOL,
        const size_type dim              = SphericalDataDefaults::DEFAULT_DIM);

      ~SphericalDataAnalytical() = default;

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

    private:
      std::vector<int>                     d_qNumbers;
      double                               d_polarAngleTolerance;
      double                               d_cutoffTolerance;
      double                               d_radiusTolerance;
      size_type                            d_dim;
      double                               d_cutoff;
      double                               d_smoothness;

      const SphericalHarmonicFunctions &      d_sphericalHarmonicFunc;
      const utils::ScalarSpatialFunctionReal &d_func;
    };

  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeSphericalDataAnalytical_h
