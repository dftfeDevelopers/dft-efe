#ifndef dftefeQuadratureRuleAdaptive_h
#define dftefeQuadratureRuleAdaptive_h

#include "QuadratureRule.h"
#include <utils/TypeConfig.h>
#include <utils/ScalarSpatialFunction.h>
#include <basis/CellMappingBase.h>
#include <basis/TriangulationCellBase.h>
#include <basis/ParentToChildCellsManagerBase.h>
#include <vector>
#include <memory>
#include <quadrature/Defaults.h>
#include <map>
#include <chrono>

namespace dftefe
{
  namespace quadrature
  {
    class QuadratureRuleAdaptive : public QuadratureRule
    {
    public:
      QuadratureRuleAdaptive(
        const basis::TriangulationCellBase &  cell,
        const QuadratureRule &                baseQuadratureRule,
        const basis::CellMappingBase &        cellMapping,
        basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                       functions,
        const std::vector<double> &    absoluteTolerances,
        const std::vector<double> &    relativeTolerances,
        const std::vector<double> &    integralThresholds,
        std::map<std::string, double> &timer,
        const double                   smallestCellVolume =
          QuadratureRuleAdaptiveDefaults::SMALLEST_CELL_VOLUME,
        const unsigned int maxRecursion =
          QuadratureRuleAdaptiveDefaults::MAX_RECURSION);

    private:
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRuleAdaptive_h
