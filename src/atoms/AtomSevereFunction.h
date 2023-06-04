#ifndef dftefeAtomSevereFunction_h
#define dftefeAtomSevereFunction_h

#include <utils/ScalarSpatialFunction.h>
namespace dftefe
{
  namespace atoms
  {
    class AtomSevereFunction : public utils::ScalarSpatialFunctionReal
    {
    public:
      AtomSevereFunction(
        std::shared_ptr<const basis::EnrichementIdsPartition> enrichmentIdsPartition,
        std::shared_ptr<const AtomSphericalDataContainer> atomSphericalDataContainer,
        const std::vector<std::string> &             atomSymbol,
        const std::vector<utils::Point> &            atomCoordinates,
        const std::string                            fieldName,
        const size_type                              DerivativeType);// give arguments here
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
#endif // dftefeAtomSevereFunction_h
