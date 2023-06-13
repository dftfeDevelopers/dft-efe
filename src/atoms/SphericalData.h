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
 * @author Bikash Kanungo
 */

#ifndef dftefeSphericalData_h
#define dftefeSphericalData_h

#include <utils/TypeConfig.h>
#include <vector>
#include <utils/Point.h>
#include <utils/Spline.h>
#include <memory>

namespace dftefe
{
  namespace atoms
  {
    class SphericalData
    {
    public:
      std::vector<int>    qNumbers;
      std::vector<double> radialPoints;
      std::vector<double> radialValues;
      double              cutoff;
      double              smoothness;

      SphericalData(); // const double polarAngleTolerance =
                       // SphericalDataDefaults::POL_ANG_TOL

      ~SphericalData() = default;

      void
      initSpline();

      template <unsigned int dim>
      double
      getValue(const utils::Point &point,
               const utils::Point &origin,
               const double        polarAngleTolerance = 1e-6);

      template <unsigned int dim>
      std::vector<double>
      getGradientValue(const utils::Point &point,
                       const utils::Point &origin,
                       const double        polarAngleTolerance = 1e-6,
                       const double        cutoffTolerance     = 1e-6);

      template <unsigned int dim>
      std::vector<double>
      getHessianValue(const utils::Point &point,
                      const utils::Point &origin,
                      const double        polarAngleTolerance = 1e-6,
                      const double        cutoffTolerance     = 1e-6);

    private:
      std::shared_ptr<const utils::Spline> d_spline;
      double                               d_value;
      std::vector<double>                  d_gradient;
      std::vector<double>                  d_hessian;
    };

  } // end of namespace atoms
} // end of namespace dftefe
#include <atoms/SphericalData.t.cpp>
#endif // dftefeSphericalData_h
