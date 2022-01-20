#include "LogModX.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    LogModX::LogModX(const unsigned int component,
                     const double       base /*= e (Euler's constant)*/)
      : d_logBase(std::log(base))
      , d_component(component)
    {}

    double
    LogModX::operator()(const utils::Point &point) const
    {
      return std::log(std::abs(point[d_component])) / d_logBase;
    }

    std::vector<double>
    LogModX::operator()(const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (unsigned int i = 0; i < N; ++i)
        returnValue[i] = std::log(std::abs(points[i][d_component])) / d_logBase;

      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
