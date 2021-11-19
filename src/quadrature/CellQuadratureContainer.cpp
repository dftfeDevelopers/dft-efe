#include "CellQuadratureContainer.h"
namespace dftefe
{
  namespace quadrature
  {
    CellQuadratureContainer::CellQuadratureContainer(
      std::shared_ptr<const QuadratureRule>           quadratureRule,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping)
    {
      size_type numCells = triangulation->nLocalCells();
      // d_quadratureRuleVec.resize(numCells, std::shared_ptr<QuadratureRule>
      // (new QuadratureRule()));
      d_quadratureRuleVec.resize(0);
      unsigned int numQuadPoints = 0;
      for (unsigned int iCell = 0; iCell < numCells; ++iCell)
        {
          d_quadratureRuleVec.push_back(
            static_cast<const std::shared_ptr<QuadratureRule>>(quadratureRule));
          numQuadPoints += d_quadratureRuleVec[iCell]->nPoints();
        }

      d_realPoints.resize(numQuadPoints,
                          dftefe::utils::Point(triangulation->getDim(), 0.0));
      for (unsigned int iCell = 0; iCell < numCells; ++iCell)
        {
          const std::vector<dftefe::utils::Point> parametricPoints =
            d_quadratureRuleVec[iCell]->getPoints();
          std::vector<dftefe::utils::Point> realPoints(0);
        }
    }

  } // end of namespace quadrature
} // end of namespace dftefe
