#ifndef dftefeQuadratureRuleAdaptive_h
#define dftefeQuadratureRuleAdaptive_h

#include <utils/TypeConfig.h>
#include <utils/ScalarFunction.h>
#include <basis/QuadratureRule.h>
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
        std::vector<std::shared_ptr<const ScalarFunction>> functions,
        const std::vector<double> &                        tolerances,
        const std::vector<double> &                        integralThresholds,
        const double       smallestCellVolume = 1e-12,
        const unsigned int maxRecursion       = 100);

      ~QuadratureRuleAdaptive();


    private:
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRuleAdaptive_h
