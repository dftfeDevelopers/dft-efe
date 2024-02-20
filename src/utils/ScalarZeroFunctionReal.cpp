#include "ScalarZeroFunctionReal.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    ScalarZeroFunctionReal::ScalarZeroFunctionReal()
    {}

    double
    ScalarZeroFunctionReal::operator()(const utils::Point &point) const
    {
      return 0.0;
    }

    std::vector<double>
    ScalarZeroFunctionReal::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (unsigned int i = 0; i < N; ++i)
        returnValue[i] = 0.0;

      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
