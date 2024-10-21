#include "SmearChargePotentialFunction.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    SmearChargePotentialFunction::SmearChargePotentialFunction(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges,
      const std::vector<double> &      smearedChargeRadius)
      : d_atomCoordinates(atomCoordinates)
      , d_z(atomCharges)
      , d_rc(smearedChargeRadius)
    {}

    SmearChargePotentialFunction::SmearChargePotentialFunction(
      const utils::Point &atomCoordinates,
      const double        atomCharges,
      const double        smearedChargeRadius)
      : d_atomCoordinates(std::vector<utils::Point>{atomCoordinates})
      , d_z(std::vector<double>{atomCharges})
      , d_rc(std::vector<double>{smearedChargeRadius})
    {}

    double
    SmearChargePotentialFunction::operator()(const utils::Point &point) const
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
          if (r > d_rc[i])
            ret += 1 / r * d_z[i];
          else
            ret += d_z[i] *
                   (9 * std::pow(r, 7) - 30 * std::pow(r, 6) * d_rc[i] +
                    28 * std::pow(r, 5) * std::pow(d_rc[i], 2) -
                    14 * std::pow(r, 2) * std::pow(d_rc[i], 5) +
                    12 * std::pow(d_rc[i], 7)) /
                   (5 * std::pow(d_rc[i], 8));
        }
      return ret;
    }

    std::vector<double>
    SmearChargePotentialFunction::operator()(
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
              if (r > d_rc[i])
                ret += 1 / r * d_z[i];
              else
                ret += d_z[i] *
                       (9 * std::pow(r, 7) - 30 * std::pow(r, 6) * d_rc[i] +
                        28 * std::pow(r, 5) * std::pow(d_rc[i], 2) -
                        14 * std::pow(r, 2) * std::pow(d_rc[i], 5) +
                        12 * std::pow(d_rc[i], 7)) /
                       (5 * std::pow(d_rc[i], 8));
            }
          returnValue[iPoint] = ret;
        }
      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
