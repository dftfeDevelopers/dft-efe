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
#include <utils/ScalarSpatialFunction.h>

namespace dftefe
{
  namespace atoms
  {
    class SphericalData
    {
    public:
      ~SphericalData() = default;

      virtual std::vector<double>
      getValue(const std::vector<utils::Point> &point,
               const utils::Point &             origin) = 0;

      virtual std::vector<double>
      getGradientValue(const std::vector<utils::Point> &point,
                       const utils::Point &             origin) = 0;

      virtual std::vector<double>
      getHessianValue(const std::vector<utils::Point> &point,
                      const utils::Point &             origin) = 0;

      virtual double
      getValue(const utils::Point &point, const utils::Point &origin) = 0;

      virtual std::vector<double>
      getGradientValue(const utils::Point &point,
                       const utils::Point &origin) = 0;

      virtual std::vector<double>
      getHessianValue(const utils::Point &point,
                      const utils::Point &origin) = 0;

      // gives f(r)
      virtual std::vector<double>
      getRadialValue(const std::vector<double> &r) = 0;

      // gives Y_lm(theta, phi)
      virtual std::vector<double>
      getAngularValue(const std::vector<double> &r,
                      const std::vector<double> &theta,
                      const std::vector<double> &phi) = 0;

      // gives f'(r)
      virtual std::vector<double>
      getRadialDerivative(const std::vector<double> &r) = 0;

      // gives Y'(theta, phi) = std::vector{(1/r)dY_lm/dtheta ,
      // (1/rsin(theta))dY_lm/dphi} return value has size 2 vector for theta hat
      // and phi hat components of Y_lm derivative
      virtual std::vector<std::vector<double>>
      getAngularDerivative(const std::vector<double> &r,
                           const std::vector<double> &theta,
                           const std::vector<double> &phi) = 0;

      virtual std::vector<int>
      getQNumbers() const = 0;

      virtual double
      getCutoff() const = 0;

      virtual double
      getSmoothness() const = 0;
    };

  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeSphericalData_h
