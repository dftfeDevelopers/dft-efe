#ifndef dftefePointChargePotentialFunction_h
#define dftefePointChargePotentialFunction_h

#include "ScalarSpatialFunction.h"
#include "MathConstants.h"
namespace dftefe
{
  namespace utils
  {
    class PointChargePotentialFunction : public ScalarSpatialFunctionReal
    {
    public:
      PointChargePotentialFunction(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges);

      PointChargePotentialFunction(const utils::Point &atomCoordinates,
                                   const double        atomCharges);

      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    private:
      std::vector<utils::Point> d_atomCoordinates;
      std::vector<double>       d_z;
    };

  } // namespace utils
} // namespace dftefe
#endif // dftefePointChargePotentialFunction_h
