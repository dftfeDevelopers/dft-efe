#include "SmearChargeDensityFunction.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    SmearChargeDensityFunction::SmearChargeDensityFunction(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges,
      const std::vector<double> &      smearedChargeRadius)
      : d_atomCoordinates(atomCoordinates)
      , d_z(atomCharges)
      , d_rc(smearedChargeRadius)
    {}

    SmearChargeDensityFunction::SmearChargeDensityFunction(
      const utils::Point &atomCoordinates,
      const double        atomCharges,
      const double        smearedChargeRadius)
      : d_atomCoordinates(std::vector<utils::Point>{atomCoordinates})
      , d_z(std::vector<double>{atomCharges})
      , d_rc(std::vector<double>{smearedChargeRadius})
    {}

    double
    SmearChargeDensityFunction::operator()(const utils::Point &point) const
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
            ret += 0;
          else
            ret += d_z[i] * -21 * std::pow((r - d_rc[i]), 3) *
                   (6 * r * r + 3 * r * d_rc[i] + d_rc[i] * d_rc[i]) /
                   (5 * M_PI * std::pow(d_rc[i], 8));
        }
      return ret;
    }

    std::vector<double>
    SmearChargeDensityFunction::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (unsigned int i = 0; i < d_atomCoordinates.size(); i++)
        {
          for (unsigned int iPoint = 0; iPoint < N; ++iPoint)
            {
              double r = 0;
              for (unsigned int j = 0; j < points[iPoint].size(); j++)
                {
                  r +=
                    std::pow((points[iPoint][j] - d_atomCoordinates[i][j]), 2);
                }
              r = std::sqrt(r);
              if (r < d_rc[i])
                returnValue[iPoint] +=
                  d_z[i] * -21 * std::pow((r - d_rc[i]), 3) *
                  (6 * r * r + 3 * r * d_rc[i] + d_rc[i] * d_rc[i]) /
                  (5 * M_PI * std::pow(d_rc[i], 8));
            }
        }
      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
