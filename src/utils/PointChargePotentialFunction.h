#ifndef dftefePointChargePotentialFunction_h
#define dftefePointChargePotentialFunction_h

#include "ScalarSpatialFunction.h"
#include "MemorySpaceType.h"
#include "MemoryStorage.h"
#include <memory>
#include "TypeConfig.h"
#include <vector>

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
      std::vector<double>       d_z;
      size_type                 d_numAtoms;
      size_type                 d_dim;

#ifdef DFTEFE_WITH_DEVICE
      mutable std::unique_ptr<MemoryStorage<double, MemorySpace::DEVICE>>
        d_atomCoordsFlatDevice;
      mutable std::unique_ptr<MemoryStorage<double, MemorySpace::DEVICE>>
        d_zDevice;
#endif
    };

  } // namespace utils
} // namespace dftefe

#endif // dftefePointChargePotentialFunction_h
