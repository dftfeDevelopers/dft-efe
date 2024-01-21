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
        const std::vector<double> &absoluteTolerances,
        const std::vector<double> &relativeTolerances,
        const std::vector<double> &integralThresholds,
        const double               smallestCellVolume = 1e-12,
        const unsigned int         maxRecursion       = 100);

    private:
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRuleAdaptive_h
