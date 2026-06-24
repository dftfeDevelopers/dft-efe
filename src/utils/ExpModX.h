#ifndef dftefeExpModX_h
#define dftefeExpModX_h

#include "ScalarSpatialFunction.h"
#include "MathConstants.h"
namespace dftefe
{
  namespace utils
  {
    class ExpModX : public ScalarSpatialFunctionReal
    {
    public:
      ExpModX(const size_type component, const double exponent = 1.0);
      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    private:
      size_type d_component;
      double    d_exponent;
    };

  } // namespace utils
} // namespace dftefe
#endif // dftefeExpModX_h
