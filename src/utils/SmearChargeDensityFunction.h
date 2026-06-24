#ifndef dftefeSmearChargeDensityFunction_h
#define dftefeSmearChargeDensityFunction_h

#include "ScalarSpatialFunction.h"
#include "MemorySpaceType.h"
#include "MemoryStorage.h"
#include "TypeConfig.h"
#include <vector>

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

      SmearChargeDensityFunction(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const double &                   smearedChargeRadius);

      SmearChargeDensityFunction(const utils::Point &atomCoordinates,
                                 const double        atomCharges,
                                 const double        smearedChargeRadius);

      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    protected:
      void
      evalHost(size_type numPoints, const double *t, double *q) const override;

#ifdef DFTEFE_WITH_DEVICE
      void
      evalDevice(size_type     numPoints,
                 const double *t,
                 double *      q) const override;
#endif

    private:
      std::vector<utils::Point> d_atomCoordinates;
      std::vector<double>       d_rc;
      std::vector<double>       d_z;
      size_type                 d_numAtoms;
      size_type                 d_dim;

#ifdef DFTEFE_WITH_DEVICE
      MemoryStorage<double, MemorySpace::DEVICE> d_atomCoordsFlatDevice;
      MemoryStorage<double, MemorySpace::DEVICE> d_rcDevice;
      MemoryStorage<double, MemorySpace::DEVICE> d_zDevice;
#endif
    };

  } // namespace utils
} // namespace dftefe

#endif // dftefeSmearChargeDensityFunction_h
