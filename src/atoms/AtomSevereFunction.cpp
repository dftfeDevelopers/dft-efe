#include "AtomSevereFunction.h"
#include <utils/TypeConfig.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    AtomSevereFunction::AtomSevereFunction(
        std::shared_ptr<const basis::EnrichementIdsPartition> enrichmentIdsPartition,
        std::shared_ptr<const AtomSphericalDataContainer> atomSphericalDataContainer,
        const std::vector<std::string> &             atomSymbol,
        const std::vector<utils::Point> &            atomCoordinates,
        const std::string                            fieldName,
        const size_type                              DerivativeType)
      : d_logBase(std::log(base))
      , d_component(component)
    {}

    double
    LogModX::operator()(const utils::Point &point) const
    {
      return std::log(std::abs(point[d_component])) / d_logBase;
    }

    std::vector<double>
    LogModX::operator()(const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (unsigned int i = 0; i < N; ++i)
        returnValue[i] = std::log(std::abs(points[i][d_component])) / d_logBase;

      return returnValue;
    }
  } // namespace utils
} // namespace dftefe
