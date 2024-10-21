#include "PointChargePotentialFunction.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    PointChargePotentialFunction::PointChargePotentialFunction(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges)
      : d_atomCoordinates(atomCoordinates)
      , d_z(atomCharges)
    {}

    PointChargePotentialFunction::PointChargePotentialFunction(
      const utils::Point &atomCoordinates,
      const double        atomCharges)
      : d_atomCoordinates(std::vector<utils::Point>{atomCoordinates})
      , d_z(std::vector<double>{atomCharges})
    {}

    double
    PointChargePotentialFunction::operator()(const utils::Point &point) const
    {
      double ret = 0;
      for (unsigned int i = 0; i < d_atomCoordinates.size(); i++)
        {
          double r = 0;
          for (unsigned int j = 0; j < point.size(); j++)
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
      for (unsigned int iPoint = 0; iPoint < N; ++iPoint)
        {
          double ret = 0;
          for (unsigned int i = 0; i < d_atomCoordinates.size(); i++)
            {
              double r = 0;
              for (unsigned int j = 0; j < points[iPoint].size(); j++)
                {
                  r +=
                    std::pow((points[iPoint][j] - d_atomCoordinates[i][j]), 2);
                }
              r = std::sqrt(r);
              DFTEFE_AssertWithMsg(
                std::abs(r) >= 1e-12,
                "Value undefined at nucleus for 1/r potential");
              ret += 1 / r * d_z[i];
            }
          returnValue[iPoint] = ret;
        }
      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
