#ifndef dftefeScalarSpatialFunctionZero_h
#define dftefeScalarSpatialFunctionZero_h

#include "ScalarSpatialFunction.h"
namespace dftefe
{
  namespace utils
  {
    template <typename T>
    class ScalarSpatialFunctionZero : public ScalarSpatialFunction<T>
    {
    public:
      ScalarSpatialFunctionZero();
      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;
    };

  } // namespace utils
} // namespace dftefe
#endif // dftefeScalarSpatialFunctionZero_h
