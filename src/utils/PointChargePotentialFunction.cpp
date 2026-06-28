#include <cmath>
#include "Exceptions.h"
#include <utils/PointChargePotentialFunction.h>

namespace dftefe
{
  namespace utils
  {
    PointChargePotentialFunction::PointChargePotentialFunction(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges)
      : d_atomCoordinates(atomCoordinates)
      , d_z(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_dim(atomCoordinates[0].size())
    {}

    PointChargePotentialFunction::PointChargePotentialFunction(
      const utils::Point &atomCoordinates,
      const double        atomCharges)
      : PointChargePotentialFunction(std::vector<utils::Point>{atomCoordinates},
                                     std::vector<double>{atomCharges})
    {}

    double
    PointChargePotentialFunction::operator()(const utils::Point &point) const
    {
      double ret = 0;
      for (size_type i = 0; i < d_atomCoordinates.size(); i++)
        {
          double r = 0;
          for (size_type j = 0; j < point.size(); j++)
            {
              r += std::pow((point[j] - d_atomCoordinates[i][j]), 2);
            }
          r = std::sqrt(r);
          DFTEFE_AssertWithMsg(std::abs(r) >= 1e-12,
                               "Value undefined at nucleus for 1/r potential");
          ret += 1 / r * d_z[i];
        }
      return ret;
    }

    std::vector<double>
    PointChargePotentialFunction::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (size_type i = 0; i < d_atomCoordinates.size(); i++)
        {
          for (size_type iPoint = 0; iPoint < N; ++iPoint)
            {
              double r = 0;
              for (size_type j = 0; j < points[iPoint].size(); j++)
                {
                  r +=
                    std::pow((points[iPoint][j] - d_atomCoordinates[i][j]), 2);
                }
              r = std::sqrt(r);
              DFTEFE_AssertWithMsg(
                std::abs(r) >= 1e-12,
                "Value undefined at nucleus for 1/r potential");
              returnValue[iPoint] += 1 / r * d_z[i];
            }
        }
      return returnValue;
    }

    void
    PointChargePotentialFunction::evalHost(size_type     numPoints,
                                           const double *t,
                                           double *      q) const
    {
      utils::Point              p(d_dim);
      std::vector<utils::Point> points(numPoints, p);
      for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
        for (size_type iDim = 0; iDim < d_dim; ++iDim)
          points[iPoint][iDim] = t[iPoint * d_dim + iDim];
      std::vector<double> retValue = (*this)(points);
      std::copy(retValue.begin(), retValue.end(), q);
    }

  } // namespace utils
} // namespace dftefe
