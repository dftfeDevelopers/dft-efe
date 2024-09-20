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
#include <vector>
#include <utils/Point.h>
#include <atoms/SphericalData.h>
#include <utils/Spline.h>
#include <memory>
#include <atoms/Defaults.h>

namespace dftefe
{
  namespace atoms
  {
    class SphericalDataNumerical : public SphericalData
    {
    public:
      SphericalDataNumerical(
        const std::vector<int>    qNumbers,
        const std::vector<double> radialPoints,
        const std::vector<double> radialValues,
        const double              cutoff,
        const double              smoothness,
        const double polarAngleTolerance = SphericalDataDefaults::POL_ANG_TOL,
        const double cutoffTolerance     = SphericalDataDefaults::CUTOFF_TOL,
        const double radiusTolerance     = SphericalDataDefaults::RADIUS_TOL,
        const size_type dim              = SphericalDataDefaults::DEFAULT_DIM);

      ~SphericalDataNumerical() = default;

      void
      initSpline();

      double
      getValue(const utils::Point &point, const utils::Point &origin) override;

      std::vector<double>
      getGradientValue(const utils::Point &point,
                       const utils::Point &origin) override;

      std::vector<double>
      getHessianValue(const utils::Point &point,
                      const utils::Point &origin) override;

      std::vector<int>
      getQNumbers() const override;

      double
      getCutoff() const override;

      double
      getSmoothness() const override;

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
    };

  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeSphericalDataNumerical_h
