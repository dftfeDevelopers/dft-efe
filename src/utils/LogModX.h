#ifndef dftefeLogModX_h
#define dftefeLogModX_h

#include "ScalarSpatialFunction.h"
#include "MathConstants.h"
namespace dftefe
{
  namespace utils
  {
    class LogModX : public ScalarSpatialFunctionReal
    {
    public:
      LogModX(const unsigned int component,
              const double       base = mathConstants::e);
      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    private:
      unsigned int d_component;
      double       d_logBase;
    };

  } // namespace utils
} // namespace dftefe
#endif // dftefeLogModX_h
