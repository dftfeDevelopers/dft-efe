#include "CellQuadratureContainer.h"
#include <utils/Exceptions.h>

namespace dftefe
{
  namespace quadrature
  {
    CellQuadratureContainer::CellQuadratureContainer(
      std::shared_ptr<const QuadratureRule>           quadratureRule,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping)
      : d_dim(triangulation->getDim())
    {
      utils::throwException(
        d_dim == quadratureRule[0]->getDim(),
        "Mismatch of dimension of the quadrature points and the triangulation.");
      size_type numCells = triangulation->nLocalCells();
      std::vector<std::shared_ptr<const QuadratureRule>> quadratureRuleVec(
        numCells, quadratureRule);
      CellQuadratureContainer(quadratureRuleVec, triangulation, cellMapping);
    }

    CellQuadratureContainer::CellQuadratureContainer(
      std::vector<std::shared_ptr<const QuadratureRule>> quadratureRuleVec,
      std::shared_ptr<const basis::TriangulationBase>    triangulation,
      const basis::CellMappingBase &                     cellMapping)
      : d_dim(triangulation->getDim())
      , d_quadratureRuleVec(quadratureRuleVec)
    {
      utils::throwException(
        d_dim == d_quadratureRuleVec[0]->getDim(),
        "Mismatch of dimension of the quadrature points and the triangulation.");

      size_type numCells = triangulation->nLocalCells();

      utils::throwException(
        numCells == d_quadratureRuleVec.size(),
        "Mismatch of number of cells in the quadratureRuleVec and the"
        "number of cells in the triangulation.");

      d_numCellQuadPoints.resize(numCells, 0);
      d_cellQuadStartIds.resize(numCells, 0);
      d_numQuadPoints = 0;
      for (unsigned int iCell = 0; iCell < numCells; ++iCell)
        {
          const size_type numCellQuadPoints =
            d_quadratureRuleVec[iCell]->nPoints();
          d_numCellQuadPoints[iCell] = numCellQuadPoints;
          d_cellQuadStartIds[iCell]  = d_numQuadPoints;
          d_numQuadPoints += numCellQuadPoints;
        }

      d_realPoints.resize(d_numQuadPoints, utils::Point(d_dim, 0.0));
      d_JxW.resize(d_numQuadPoints, 0.0);
      basis::TriangulationBase::const_cellIterator cellIter =
        triangulationBase->beginLocal();
      unsigned int iCell = 0;
      for (; cellIter != triangulationBase->endLocal(); ++celIter)
        {
          const size_type numCellQuadPoints = d_numCellQuadPoints[iCell];
          const std::vector<utils::Point> &parametricPoints =
            d_quadratureRuleVec[iCell]->getPoints();
          std::vector<utils::Point> cellRealPoints(numCellQuadPoints,
                                                   utils::Point(d_dim, 0.0));
          cellMapping.getRealPoints(parametericPoints,
                                     *(*cellIter),
                                     cellRealPoints);
          const std::vector<double> &weights =
            d_quadratureRuleVec[iCell]->getWeights();
          std::vector<double> cellJxW(numCellQuadPoints, 0.0);
          cellMapping.getJxW(*(*celIter), parametricPoints, weights, cellJxW);
          const size_type cellQuadStartId = d_cellQuadStartIds[iCell];
          std::copy(cellRealPoints.begin(),
                    cellRealPoints.end(),
                    d_realPoints.begin() + cellQuadStartId);
          std::copy(cellJxW.begin(),
                    cellJxW.end(),
                    d_JxW.begin() + cellQuadStartId);
          iCell++;
        }
    }
    const std::vector<utils::Point> &
    CellQuadratureContainer::getRealPoints() const
    {
      return d_realPoints;
    }

    std::vector<utils::Point>
    CellQuadratureContainer::getCellRealPoints(const unsigned int cellId) const
    {
      const size_type numCellQuadPoints = d_numCellQuadPoints[cellId];
      const size_type cellQuadStartId   = d_cellQuadStartIds[cellId];
      const size_type cellQuadEndId     = cellQuadStartId + numCellQuadPoints;

      std::vector<utils::Point> cellRealPoints(numCellQuadPoints,
                                               utils::Point(d_dim, 0.0));

      std::copy(d_realPoints.begin() + cellQuadStartId,
                d_realPoints.begin() + cellQuadEndId,
                cellRealPoints.begin());

      return cellRealPoints;
    }

    const std::vector<utils::Point> &
    CellQuadratureContainer::getCellParametricPoints(
      const unsigned int cellId) const
    {
      return d_quadratureRuleVec[cellId]->getPoints();
    }

    const std::vector<double> &
    CellQuadratureContainer::getCellQuadratureWeights(
      const unsigned int cellId) const
    {
      return d_quadratureRuleVec[cellId]->getWeights();
    }

    const std::vector<double> &
    CellQuadratureContainer::getJxW() const
    {
      return d_JxW;
    }

    std::vector<double>
    CellQuadratureContainer::getCellJxW(const unsigned int cellId) const
    {
      const size_type numCellQuadPoints = d_numCellQuadPoints[cellId];
      const size_type cellQuadStartId   = d_cellQuadStartIds[cellId];
      const size_type cellQuadEndId     = cellQuadStartId + numCellQuadPoints;

      std::vector<double> cellJxW(numCellQuadPoints, 0.0);
      std::copy(d_JxW.begin() + cellQuadStartid,
                d_JxW.begin() + cellQuadEndId,
                cellJxW.begin());
      return cellJxW;
    }

    const QuadratureRule &
    CellQuadratureContainer::getQuadratureRule(const unsigned int cellId) const
    {
      return *d_quadratureRuleVec[cellId];
    }

    size_type
    CellQuadratureContainer::nQuadraturePoints() const
    {
      return d_numQuadPoints;
    }

    size_type
    CellQuadratureContainer::nCellQuadraturePoints(
      const unsigned int cellId) const
    {
      return d_numCellQuadPoints[cellId];
    }
  } // end of namespace quadrature
} // end of namespace dftefe
