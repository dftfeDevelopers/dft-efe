#ifndef dftefeCellQuadratureContainer_h
#define dftefeCellQuadratureContainer_h

#include "QuadratureRule.h"
#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/TriangulationBase.h>
#include <basis/CellMappingBase.h>
#include <memory>

namespace dftefe
{
  namespace quadrature
  {
    class CellQuadratureContainer
    {
    public:
      CellQuadratureContainer(
        std::shared_ptr<const QuadratureRule>           quadratureRule,
        std::shared_ptr<const basis::TriangulationBase> triangulation,
        const basis::CellMappingBase &                  cellMapping);

      CellQuadratureContainer(
        std::vector<std::shared_ptr<const QuadratureRule>> quadratureRule,
        std::shared_ptr<const basis::TriangulationBase>    triangulation,
        const basis::CellMappingBase &                     cellMapping);

      const std::vector<dftefe::utils::Point> &
      getRealPoints() const;

      std::vector<dftefe::utils::Point>
      getCellRealPoints(const unsigned int cellId) const;

      const std::vector<dftefe::utils::Point> &
      getCellParametricPoints(const unsigned int cellId) const;

      const std::vector<double> &
      getCellQuadratureWeights(const unsigned int cellId) const;

      const std::vector<double> &
      getJxW() const;

      std::vector<double>
      getCellJxW(const unsigned int cellId) const;

      const QuadratureRule &
      getQuadratureRule(const unsigned int cellId) const;

      size_type
      nQuadraturePoints() const;

      size_type
      nCellQuadraturePoints(const unsigned int cellId) const;


    private:
      std::vector<std::shared_ptr<const QuadratureRule>> d_quadratureRuleVec;
      std::vector<size_type>                             d_numCellQuadPoints;
      std::vector<size_type>                             d_cellQuadStartIds;
      std::vector<dftefe::utils::Point>                  d_realPoints;
      std::vector<double>                                d_JxW;
      unsigned int                                       d_dim;
      size_type                                          d_numQuadPoints;
    };
  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeCellQuadratureContainer_h
