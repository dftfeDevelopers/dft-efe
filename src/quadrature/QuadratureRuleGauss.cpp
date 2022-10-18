#include <quadrature/QuadratureRuleGauss.h>
#include <utils/MathFunctions.h>
#include <deal.II/base/quadrature_lib.h>

namespace dftefe
{
  namespace quadrature
  {
    QuadratureRuleGauss::QuadratureRuleGauss(const size_type dim,
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

      dealii::QGauss<1> qgauss1D(order1D);

      for (size_type iQuad = 0; iQuad < order1D; iQuad++)
        {
          d_1DWeights[iQuad]   = qgauss1D.weight(iQuad);
          d_1DPoints[iQuad][0] = qgauss1D.point(iQuad)[0];
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
