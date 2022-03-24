#include "CellQuadratureContainer.h"
#include "QuadratureRuleAdaptive.h"
#include <utils/Exceptions.h>

namespace dftefe
{
  namespace quadrature
  {
    namespace
    {
      void
      initialize(
        std::vector<std::shared_ptr<const QuadratureRule>> quadratureRuleVec,
        std::shared_ptr<const basis::TriangulationBase>    triangulation,
        const basis::CellMappingBase &                     cellMapping,
        std::vector<size_type> &                           numCellQuadPoints,
        std::vector<size_type> &                           cellQuadStartIds,
        std::vector<utils::Point> &                        realPoints,
        std::vector<double> &                              JxW,
        size_type &                                        numQuadPoints)
      {
        const unsigned int dim      = triangulation->getDim();
        const size_type    numCells = triangulation->nLocalCells();
        numCellQuadPoints.resize(numCells, 0);
        cellQuadStartIds.resize(numCells, 0);
        numQuadPoints = 0;
        for (unsigned int iCell = 0; iCell < numCells; ++iCell)
          {
            const size_type numQuadPointsInCell =
              quadratureRuleVec[iCell]->nPoints();
            numCellQuadPoints[iCell] = numQuadPointsInCell;
            cellQuadStartIds[iCell]  = numQuadPoints;
            numQuadPoints += numQuadPointsInCell;
          }

        realPoints.resize(numQuadPoints, dftefe::utils::Point(dim, 0.0));
        JxW.resize(numQuadPoints, 0.0);
        basis::TriangulationBase::const_TriangulationCellIterator cellIter =
          triangulation->beginLocal();
        unsigned int iCell = 0;
        for (; cellIter != triangulation->endLocal(); ++cellIter)
          {
            const size_type numQuadPointsInCell = numCellQuadPoints[iCell];
            const std::vector<dftefe::utils::Point> &parametricPoints =
              quadratureRuleVec[iCell]->getPoints();
            std::vector<dftefe::utils::Point> cellRealPoints(
              numQuadPointsInCell, utils::Point(dim, 0.0));
            cellMapping.getRealPoints(parametricPoints,
                                      *(*cellIter),
                                      cellRealPoints);
            const std::vector<double> &weights =
              quadratureRuleVec[iCell]->getWeights();
            std::vector<double> cellJxW(numQuadPointsInCell, 0.0);
            cellMapping.getJxW(*(*cellIter),
                               parametricPoints,
                               weights,
                               cellJxW);
            const size_type cellQuadStartId = cellQuadStartIds[iCell];
            std::copy(cellRealPoints.begin(),
                      cellRealPoints.end(),
                      realPoints.begin() + cellQuadStartId);
            std::copy(cellJxW.begin(),
                      cellJxW.end(),
                      JxW.begin() + cellQuadStartId);
            iCell++;
          }
      }

    } // namespace


    CellQuadratureContainer::CellQuadratureContainer(
      std::shared_ptr<const QuadratureRule>           quadratureRule,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping)
      : d_dim(triangulation->getDim())
      , d_numCellQuadPoints(0)
      , d_cellQuadStartIds(0)
      , d_realPoints(0, utils::Point(triangulation->getDim(), 0))
      , d_JxW(0)
      , d_numQuadPoints(0)
    {
      utils::throwException(
        d_dim == quadratureRule->getDim(),
        "Mismatch of dimension of the quadrature points and the triangulation.");
      d_numCells = triangulation->nLocalCells();
      d_quadratureRuleVec =
        std::vector<std::shared_ptr<const QuadratureRule>>(d_numCells,
                                                           quadratureRule);
      initialize(d_quadratureRuleVec,
                 triangulation,
                 cellMapping,
                 d_numCellQuadPoints,
                 d_cellQuadStartIds,
                 d_realPoints,
                 d_JxW,
                 d_numQuadPoints);
    }

    CellQuadratureContainer::CellQuadratureContainer(
      std::vector<std::shared_ptr<const QuadratureRule>> quadratureRuleVec,
      std::shared_ptr<const basis::TriangulationBase>    triangulation,
      const basis::CellMappingBase &                     cellMapping)
      : d_dim(triangulation->getDim())
      , d_quadratureRuleVec(quadratureRuleVec)
      , d_numCellQuadPoints(0)
      , d_cellQuadStartIds(0)
      , d_realPoints(0, utils::Point(triangulation->getDim(), 0))
      , d_JxW(0)
      , d_numQuadPoints(0)
    {
      d_numCells = triangulation->nLocalCells();
      utils::throwException(
        d_numCells == d_quadratureRuleVec.size(),
        "Mismatch of number of cells in the quadratureRuleVec and the"
        "number of cells in the triangulation.");
      for (unsigned int iCell = 0; iCell < d_numCells; ++iCell)
        {
          utils::throwException(
            d_dim == d_quadratureRuleVec[iCell]->getDim(),
            "Mismatch of dimension of the quadrature points and the triangulation.");
        }

      initialize(d_quadratureRuleVec,
                 triangulation,
                 cellMapping,
                 d_numCellQuadPoints,
                 d_cellQuadStartIds,
                 d_realPoints,
                 d_JxW,
                 d_numQuadPoints);
    }


    CellQuadratureContainer::CellQuadratureContainer(
      std::shared_ptr<const QuadratureRule>           baseQuadratureRule,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping,
      basis::ParentToChildCellsManagerBase &          parentToChildCellsManager,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                 functions,
      const std::vector<double> &tolerances,
      const std::vector<double> &integralThresholds,
      const double               smallestCellVolume /*= 1e-12*/,
      const unsigned int         maxRecursion /*= 100*/)
      : d_dim(triangulation->getDim())
    {
      utils::throwException(
        d_dim == baseQuadratureRule->getDim(),
        "Mismatch of dimension of the quadrature points and the triangulation.");

      d_numCells = triangulation->nLocalCells();
      d_quadratureRuleVec.resize(d_numCells);
      d_numCellQuadPoints.resize(d_numCells, 0);
      d_cellQuadStartIds.resize(d_numCells, 0);
      d_numQuadPoints                                    = 0;
      unsigned int                                 iCell = 0;
      basis::TriangulationBase::const_TriangulationCellIterator cellIter =
        triangulation->beginLocal();
      for (; cellIter != triangulation->endLocal(); ++cellIter)
        {
          QuadratureRuleAdaptive adaptiveQuadratureRule(
            *(*cellIter),
            *baseQuadratureRule,
            cellMapping,
            parentToChildCellsManager,
            functions,
            tolerances,
            integralThresholds,
            smallestCellVolume,
            maxRecursion);
          d_quadratureRuleVec[iCell] =
            std::make_shared<QuadratureRule>(adaptiveQuadratureRule);

          const size_type numCellQuadPoints =
            d_quadratureRuleVec[iCell]->nPoints();
          d_numCellQuadPoints[iCell] = numCellQuadPoints;
          d_cellQuadStartIds[iCell]  = d_numQuadPoints;
          d_numQuadPoints += numCellQuadPoints;
          iCell++;
        }

      d_realPoints.resize(d_numQuadPoints, dftefe::utils::Point(d_dim, 0.0));
      d_JxW.resize(d_numQuadPoints, 0.0);
      iCell = 0;
      for (auto cellIter = triangulation->beginLocal();
           cellIter != triangulation->endLocal();
           ++cellIter)
        {
          const size_type numCellQuadPoints = d_numCellQuadPoints[iCell];
          const std::vector<dftefe::utils::Point> &parametricPoints =
            d_quadratureRuleVec[iCell]->getPoints();
          std::vector<dftefe::utils::Point> cellRealPoints(
            numCellQuadPoints, dftefe::utils::Point(d_dim, 0.0));
          cellMapping.getRealPoints(parametricPoints,
                                    *(*cellIter),
                                    cellRealPoints);
          const std::vector<double> &weights =
            d_quadratureRuleVec[iCell]->getWeights();
          std::vector<double> cellJxW(numCellQuadPoints, 0.0);
          cellMapping.getJxW(*(*cellIter), parametricPoints, weights, cellJxW);
          const size_type cellQuadStartId = d_cellQuadStartIds[iCell];
          std::copy(cellRealPoints.begin(),
                    cellRealPoints.end(),
                    d_realPoints.begin() + cellQuadStartId);
          std::copy(cellJxW.begin(),
                    cellJxW.end(),
                    d_JxW.begin() + cellQuadStartId);
          // double cellVolume  = 0.0;
          // std::cout << "\niCell JxW: " << iCell << std::endl;
          // for(unsigned int i = 0; i < numCellQuadPoints; ++i)
          //{
          //  std::cout << cellJxW[i] << std::endl;
          //  cellVolume += cellJxW[i];
          //}

          // std::cout << "iCell volume: " << cellVolume << std::endl;
          iCell++;
        }
    }


    size_type
    CellQuadratureContainer::nCells() const
    {
      return d_numCells;
    }

    const std::vector<dftefe::utils::Point> &
    CellQuadratureContainer::getRealPoints() const
    {
      return d_realPoints;
    }

    std::vector<dftefe::utils::Point>
    CellQuadratureContainer::getCellRealPoints(const unsigned int cellId) const
    {
      const size_type numCellQuadPoints = d_numCellQuadPoints[cellId];
      const size_type cellQuadStartId   = d_cellQuadStartIds[cellId];
      const size_type cellQuadEndId     = cellQuadStartId + numCellQuadPoints;

      std::vector<dftefe::utils::Point> cellRealPoints(
        numCellQuadPoints, dftefe::utils::Point(d_dim, 0.0));

      std::copy(d_realPoints.begin() + cellQuadStartId,
                d_realPoints.begin() + cellQuadEndId,
                cellRealPoints.begin());

      return cellRealPoints;
    }

    const std::vector<dftefe::utils::Point> &
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
      std::copy(d_JxW.begin() + cellQuadStartId,
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
