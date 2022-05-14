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

#include <quadrature/QuadratureRuleGLL.h>
#include <deal.II/base/quadrature_lib.h>
#include <utils/MathFunctions.h>
namespace dftefe
{
  namespace quadrature
  {
    QuadratureRuleGLL::QuadratureRuleGLL(const size_type dim,
                                         const size_type order1D)
    {
      d_dim = dim;

      d_num1DPoints = order1D;
      d_1DPoints.resize(order1D, utils::Point(1, 0.0));
      d_1DWeights.resize(order1D);
      d_isTensorStructured = true;

      d_numPoints = utils::mathFunctions::sizeTypePow(order1D, dim);
      d_points.resize(d_numPoints, utils::Point(d_dim, 0.0));
      d_weights.resize(d_numPoints);

      dealii::QGaussLobatto<1> qGLL1D(order1D);

      for (size_type iQuad = 0; iQuad < order1D; iQuad++)
        {
          d_1DWeights[iQuad]   = qGLL1D.weight(iQuad);
          d_1DPoints[iQuad][0] = qGLL1D.point(iQuad)[0];
        }
      if (d_dim == 1)
        {
          size_type quadIndex = 0;
          for (size_type iQuad = 0; iQuad < order1D; iQuad++)
            {
              d_points[quadIndex][0] = d_1DPoints[iQuad][0];

              d_weights[quadIndex] = d_1DWeights[iQuad];

              quadIndex++;
            }
        }
      else if (d_dim == 2)
        {
          size_type quadIndex = 0;
          for (size_type jQuad = 0; jQuad < order1D; jQuad++)
            {
              for (size_type iQuad = 0; iQuad < order1D; iQuad++)
                {
                  d_points[quadIndex][0] = d_1DPoints[iQuad][0];
                  d_points[quadIndex][1] = d_1DPoints[jQuad][0];

                  d_weights[quadIndex] =
                    d_1DWeights[iQuad] * d_1DWeights[jQuad];

                  quadIndex++;
                }
            }
        }
      else if (d_dim == 3)
        {
          size_type quadIndex = 0;
          for (size_type kQuad = 0; kQuad < order1D; kQuad++)
            {
              for (size_type jQuad = 0; jQuad < order1D; jQuad++)
                {
                  for (size_type iQuad = 0; iQuad < order1D; iQuad++)
                    {
                      d_points[quadIndex][0] = d_1DPoints[iQuad][0];
                      d_points[quadIndex][1] = d_1DPoints[jQuad][0];
                      d_points[quadIndex][2] = d_1DPoints[kQuad][0];

                      d_weights[quadIndex] = d_1DWeights[iQuad] *
                                             d_1DWeights[jQuad] *
                                             d_1DWeights[kQuad];

                      quadIndex++;
                    }
                }
            }
        }
      else
        {
          utils::throwException(d_dim > 3,
                                "dim passed to quadrature class is not valid");

          utils::throwException(d_dim < 1,
                                "dim passed to quadrature class is not valid");
        }
    }

  } // end of namespace quadrature
} // end of namespace dftefe
