#include "ExpModX.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    ExpModX::ExpModX(const size_type component,
                     const double       exponent /*= 1.0*/)
      : d_exponent(exponent)
      , d_component(component)
    {}

    double
    ExpModX::operator()(const utils::Point &point) const
    {
      return std::exp(d_exponent * std::abs(point[d_component]));
    }

    std::vector<double>
    ExpModX::operator()(const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (size_type i = 0; i < N; ++i)
        returnValue[i] =
          std::exp(d_exponent * std::abs(points[i][d_component]));

      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
