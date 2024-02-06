#ifndef dftefeScalarZeroFunctionReal_h
#define dftefeScalarZeroFunctionReal_h

#include "ScalarSpatialFunction.h"
#include "MathConstants.h"
namespace dftefe
{
  namespace utils
  {
    class ScalarZeroFunctionReal : public ScalarSpatialFunctionReal
    {
    public:
      ScalarZeroFunctionReal();
      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;
    };

  } // namespace utils
} // namespace dftefe
#endif // dftefeScalarZeroFunctionReal_h
