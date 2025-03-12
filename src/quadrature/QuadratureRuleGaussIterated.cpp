#include <quadrature/QuadratureRuleGaussIterated.h>
#include <deal.II/base/quadrature_lib.h>
#include <utils/MathFunctions.h>

namespace dftefe
{
  namespace quadrature
  {
    QuadratureRuleGaussIterated::QuadratureRuleGaussIterated(
      const size_type dim,
      const size_type order1D,
      const size_type copies)
      : d_numCopies(copies)
      , d_order1D(order1D)
    {
      d_dim = dim;

      dealii::QIterated<1> qIterated1D(dealii::QGauss<1>(order1D), copies);

      d_num1DPoints = qIterated1D.size();
      d_1DPoints.resize(d_num1DPoints, utils::Point(1, 0.0));
      d_1DWeights.resize(d_num1DPoints);
      d_isTensorStructured = true;

      d_numPoints = utils::mathFunctions::sizeTypePow(d_num1DPoints, dim);
      d_points.resize(d_numPoints, utils::Point(d_dim, 0.0));
      d_weights.resize(d_numPoints);

      for (size_type iQuad = 0; iQuad < d_num1DPoints; iQuad++)
        {
          d_1DWeights[iQuad]   = qIterated1D.weight(iQuad);
          d_1DPoints[iQuad][0] = qIterated1D.point(iQuad)[0];
        }
      if (d_dim == 1)
        {
          size_type quadIndex = 0;
          for (size_type iQuad = 0; iQuad < d_num1DPoints; iQuad++)
            {
              d_points[quadIndex][0] = d_1DPoints[iQuad][0];

              d_weights[quadIndex] = d_1DWeights[iQuad];

              quadIndex++;
            }
        }
      else if (d_dim == 2)
        {
          size_type quadIndex = 0;
          for (size_type jQuad = 0; jQuad < d_num1DPoints; jQuad++)
            {
              for (size_type iQuad = 0; iQuad < d_num1DPoints; iQuad++)
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
          for (size_type kQuad = 0; kQuad < d_num1DPoints; kQuad++)
            {
              for (size_type jQuad = 0; jQuad < d_num1DPoints; jQuad++)
                {
                  for (size_type iQuad = 0; iQuad < d_num1DPoints; iQuad++)
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

    size_type
    QuadratureRuleGaussIterated::numCopies() const
    {
      return d_numCopies;
    }

    size_type
    QuadratureRuleGaussIterated::order1D() const
    {
      return d_order1D;
    }

  } // end of namespace quadrature
} // end of namespace dftefe
