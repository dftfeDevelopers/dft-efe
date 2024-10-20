#ifndef dftefeSmearChargeDensityFunction_h
#define dftefeSmearChargeDensityFunction_h

#include "ScalarSpatialFunction.h"
#include "MathConstants.h"
namespace dftefe
{
  namespace utils
  {
    class SmearChargeDensityFunction : public ScalarSpatialFunctionReal
    {
    public:
      SmearChargeDensityFunction(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius);

      SmearChargeDensityFunction(const utils::Point &atomCoordinates,
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
#endif // dftefeSmearChargeDensityFunction_h
