#include "ScalarSpatialFunctionZero.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    template <typename T>
    ScalarSpatialFunctionZero::ScalarSpatialFunctionZero()
    {}

    template <typename T>
    T
    ScalarSpatialFunctionZero::operator()(const utils::Point &point) const
    {
      return (T)0.0;
    }

    template <typename T>
    std::vector<T>
    ScalarSpatialFunctionZero::operator()(const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<T> returnValue(N, 0.0);
      for (unsigned int i = 0; i < N; ++i)
        returnValue[i] = (T)0.0;

      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
