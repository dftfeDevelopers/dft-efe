#ifndef dftefeSmearChargePotentialFunction_h
#define dftefeSmearChargePotentialFunction_h

#include "ScalarSpatialFunction.h"
#include "MathConstants.h"
namespace dftefe
{
  namespace utils
  {
    class SmearChargePotentialFunction : public ScalarSpatialFunctionReal
    {
    public:
      SmearChargePotentialFunction(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius);

      SmearChargePotentialFunction(const utils::Point &atomCoordinates,
                                   const double        atomCharges,
                                   const double        smearedChargeRadius);

      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    private:
      std::vector<utils::Point> d_atomCoordinates;
      std::vector<double>       d_rc;
      std::vector<double>       d_z;
    };

  } // namespace utils
} // namespace dftefe
#endif // dftefeSmearChargePotentialFunction_h
