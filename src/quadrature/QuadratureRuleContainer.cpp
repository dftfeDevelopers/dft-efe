#include <quadrature/QuadratureRuleContainer.h>
#include <quadrature/QuadratureRuleAdaptive.h>
#include <quadrature/QuadratureRuleGaussIterated.h>
#include <utils/Exceptions.h>
#include <numeric>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>

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
        const size_type    numCells = triangulation->nLocallyOwnedCells();
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


    QuadratureRuleContainer::QuadratureRuleContainer(
      const QuadratureRuleAttributes &                quadratureRuleAttributes,
      std::shared_ptr<const QuadratureRule>           quadratureRule,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping)
      : d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_dim(triangulation->getDim())
      , d_numCellQuadPoints(0)
      , d_cellQuadStartIds(0)
      , d_realPoints(0, utils::Point(triangulation->getDim(), 0))
      , d_JxW(0)
      , d_numQuadPoints(0)
      , d_triangulation(triangulation)
      , d_cellMapping(cellMapping)
    {
      utils::throwException(
        d_dim == quadratureRule->getDim(),
        "Mismatch of dimension of the quadrature points and the triangulation.");

      const QuadratureFamily quadFamily =
        d_quadratureRuleAttributes.getQuadratureFamily();
      if (!(quadFamily == QuadratureFamily::GAUSS ||
            quadFamily == QuadratureFamily::GLL))
        utils::throwException<utils::LogicError>(
          false,
          "The constructor "
          "for QuadratureRuleContainer with a single input "
          " QuadratureRule is only valid for QuadratureRuleAttributes "
          "built with QuadratureFamily GAUSS or GLL.");

      d_numCells = triangulation->nLocallyOwnedCells();
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

    QuadratureRuleContainer::QuadratureRuleContainer(
      const QuadratureRuleAttributes &quadratureRuleAttributes,
      std::vector<std::shared_ptr<const QuadratureRule>> quadratureRuleVec,
      std::shared_ptr<const basis::TriangulationBase>    triangulation,
      const basis::CellMappingBase &                     cellMapping)
      : d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_dim(triangulation->getDim())
      , d_quadratureRuleVec(quadratureRuleVec)
      , d_numCellQuadPoints(0)
      , d_cellQuadStartIds(0)
      , d_realPoints(0, utils::Point(triangulation->getDim(), 0))
      , d_JxW(0)
      , d_numQuadPoints(0)
      , d_triangulation(triangulation)
      , d_cellMapping(cellMapping)
    {
      d_numCells = triangulation->nLocallyOwnedCells();
      utils::throwException<utils::LengthError>(
        d_numCells == d_quadratureRuleVec.size(),
        "Mismatch of number of cells in the quadratureRuleVec and the"
        "number of cells in the triangulation.");

      const QuadratureFamily quadFamily =
        d_quadratureRuleAttributes.getQuadratureFamily();
      if (!(quadFamily == QuadratureFamily::GAUSS_VARIABLE ||
            quadFamily == QuadratureFamily::GLL_VARIABLE))
        utils::throwException<utils::LogicError>(
          false,
          "The constructor "
          "for QuadratureRuleContainer with a vector of "
          " QuadratureRule is only valid for QuadratureRuleAttributes "
          "built with QuadratureFamily GAUSS_VARIABLE or GLL_VARIABLE");

      for (unsigned int iCell = 0; iCell < d_numCells; ++iCell)
        {
          utils::throwException<utils::LengthError>(
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


    QuadratureRuleContainer::QuadratureRuleContainer(
      const QuadratureRuleAttributes &                quadratureRuleAttributes,
      std::shared_ptr<const QuadratureRule>           baseQuadratureRule,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping,
      basis::ParentToChildCellsManagerBase &          parentToChildCellsManager,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                 functions,
      const std::vector<double> &absoluteTolerances,
      const std::vector<double> &relativeTolerances,
      const std::vector<double> &integralThresholds,
      const double               smallestCellVolume /*= 1e-12*/,
      const unsigned int         maxRecursion /*= 100*/)
      : d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_dim(triangulation->getDim())
      , d_triangulation(triangulation)
      , d_cellMapping(cellMapping)
    {
      utils::throwException(
        d_dim == baseQuadratureRule->getDim(),
        "Mismatch of dimension of the quadrature points and the triangulation.");

      const QuadratureFamily quadFamily =
        d_quadratureRuleAttributes.getQuadratureFamily();
      if (!(quadFamily == QuadratureFamily::ADAPTIVE))
        utils::throwException<utils::LogicError>(
          false,
          "The constructor "
          "for QuadratureRuleContainer with a base QuadratureRule, "
          "input ScalarSpatialFunctionReal and various tolerances "
          "is only valid for QuadratureRuleAttributes "
          "built with QuadratureFamily ADAPTIVe.");

      d_numCells = triangulation->nLocallyOwnedCells();
      d_quadratureRuleVec.resize(d_numCells);
      d_numCellQuadPoints.resize(d_numCells, 0);
      d_cellQuadStartIds.resize(d_numCells, 0);
      d_numQuadPoints                                                 = 0;
      unsigned int                                              iCell = 0;
      basis::TriangulationBase::const_TriangulationCellIterator cellIter =
        triangulation->beginLocal();

      std::map<std::string, double> timer;
      timer["Function Eval"]       = 0;
      timer["Child Cell Creation"] = 0;
      timer["Cell Mapping"]        = 0;
      for (; cellIter != triangulation->endLocal(); ++cellIter)
        {
          QuadratureRuleAdaptive adaptiveQuadratureRule(
            *(*cellIter),
            *baseQuadratureRule,
            cellMapping,
            parentToChildCellsManager,
            functions,
            absoluteTolerances,
            relativeTolerances,
            integralThresholds,
            timer,
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

      // std::cout << "Function Eval: " << timer["Function Eval"]/1e6 << "\n" <<
      // std::flush; std::cout << "Child Cell Creation: " << timer["Child Cell
      // Creation"]/1e6 << "\n" << std::flush; std::cout << "Cell Mapping: " <<
      // timer["Cell Mapping"]/1e6 << "\n" << std::flush;

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

    QuadratureRuleContainer::QuadratureRuleContainer(
      const QuadratureRuleAttributes &                quadratureRuleAttributes,
      const size_type                                 order1DMin,
      const size_type                                 order1DMax,
      const size_type                                 copies1DMax,
      std::shared_ptr<const basis::TriangulationBase> triangulation,
      const basis::CellMappingBase &                  cellMapping,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                 functions,
      const std::vector<double> &absoluteTolerances,
      const std::vector<double> &relativeTolerances,
      const quadrature::QuadratureRuleContainer
        &                        quadratureRuleContainerReference,
      const utils::mpi::MPIComm &comm)
      : d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_dim(triangulation->getDim())
      , d_numCellQuadPoints(0)
      , d_cellQuadStartIds(0)
      , d_realPoints(0, utils::Point(triangulation->getDim(), 0))
      , d_JxW(0)
      , d_numQuadPoints(0)
      , d_triangulation(triangulation)
      , d_cellMapping(cellMapping)
    {
      const QuadratureFamily quadFamily =
        d_quadratureRuleAttributes.getQuadratureFamily();
      if (!(quadFamily == QuadratureFamily::GAUSS_SUBDIVIDED))
        utils::throwException<utils::LogicError>(
          false,
          "The constructor "
          "for QuadratureRuleContainer with a base QuadratureRule, "
          "input ScalarSpatialFunctionReal, 1D orders and various tolerances "
          "is only valid for QuadratureRuleAttributes "
          "built with QuadratureFamily GAUSS_SUBDIVIDED.");

      utils::throwException(
        triangulation == quadratureRuleContainerReference.getTriangulation() &&
          &(cellMapping) ==
            &(quadratureRuleContainerReference.getCellMapping()),
        "The triangulation or FEcellMapping passed to the reference Quadrature"
        " does not match the input triangulation or FEcellMapping.");

      const size_type        numCells = triangulation->nLocallyOwnedCells();
      std::vector<size_type> cellIdsWithMaxRefQuadPoints(0);
      if (quadratureRuleContainerReference.getQuadratureRuleAttributes()
            .getQuadratureFamily() == QuadratureFamily::ADAPTIVE)
        {
          const size_type numCellsInSet =
            QuadratureRuleGaussSubdividedDefaults::
              NUM_CELLS_FOR_ADAPTIVE_REFERENCE;
          std::vector<double>    referenceQuadDensityInCellsVec(0);
          std::vector<size_type> cellIndex(numCells, 0);

          for (unsigned int iCell = 0; iCell < numCells; ++iCell)
            {
              std::vector<double> cellJxW =
                quadratureRuleContainerReference.getCellJxW(iCell);
              double cellVolume =
                std::accumulate(cellJxW.begin(), cellJxW.end(), 0.0);
              referenceQuadDensityInCellsVec.push_back(
                quadratureRuleContainerReference.nCellQuadraturePoints(iCell) *
                1.0 / cellVolume);
              cellIndex[iCell] = iCell;
            }

          std::sort(cellIndex.begin(),
                    cellIndex.end(),
                    [&](size_type A, size_type B) -> bool {
                      return referenceQuadDensityInCellsVec[A] >
                             referenceQuadDensityInCellsVec[B];
                    });

          cellIdsWithMaxRefQuadPoints.resize(std::min(numCells, numCellsInSet),
                                             0);
          for (size_type i = 0; i < cellIdsWithMaxRefQuadPoints.size(); i++)
            cellIdsWithMaxRefQuadPoints[i] = cellIndex[i];
        }
      else
        {
          cellIdsWithMaxRefQuadPoints.resize(numCells, 0);
          for (unsigned int iCell = 0; iCell < numCells; ++iCell)
            cellIdsWithMaxRefQuadPoints[iCell] = iCell;
        }

      std::vector<size_type> orderVec(0);
      std::vector<size_type> iterVec(0);
      std::vector<size_type> quadPointsVec(0);

      for (size_type i = order1DMin; i <= order1DMax; i++)
        {
          for (size_type j = 1; j <= copies1DMax; j++)
            {
              QuadratureRuleGaussIterated iteratedGaussQuadratureRule(d_dim,
                                                                      i,
                                                                      j);
              bool                        allFunctionsConvergeInAllCells = true;

              for (auto cellId : cellIdsWithMaxRefQuadPoints)
                {
                  std::shared_ptr<basis::TriangulationCellBase> cell =
                    *(triangulation->beginLocal() + cellId);

                  bool allFunctionsConverge = true;

                  const std::vector<utils::Point>
                    &gaussIteratedParametricPoints =
                      iteratedGaussQuadratureRule.getPoints();
                  std::vector<utils::Point> realQuadPointsGaussIterated(
                    iteratedGaussQuadratureRule.nPoints(),
                    utils::Point(d_dim, 0.0));
                  cellMapping.getRealPoints(gaussIteratedParametricPoints,
                                            *cell,
                                            realQuadPointsGaussIterated);

                  std::vector<double> cellJxWGaussIterated(
                    iteratedGaussQuadratureRule.nPoints(), 0.0);
                  cellMapping.getJxW(*cell,
                                     gaussIteratedParametricPoints,
                                     iteratedGaussQuadratureRule.getWeights(),
                                     cellJxWGaussIterated);

                  const std::vector<utils::Point>
                    &referenceQuadParametricPoints =
                      quadratureRuleContainerReference.getCellParametricPoints(
                        cellId);
                  std::vector<utils::Point> realQuadPointsReference(
                    quadratureRuleContainerReference.nCellQuadraturePoints(
                      cellId),
                    utils::Point(d_dim, 0.0));
                  cellMapping.getRealPoints(referenceQuadParametricPoints,
                                            *cell,
                                            realQuadPointsReference);

                  std::vector<double> cellJxWReference(
                    quadratureRuleContainerReference.nCellQuadraturePoints(
                      cellId),
                    0.0);
                  cellMapping.getJxW(*cell,
                                     referenceQuadParametricPoints,
                                     quadratureRuleContainerReference
                                       .getCellQuadratureWeights(cellId),
                                     cellJxWReference);

                  for (size_type iFunction = 0; iFunction < functions.size();
                       ++iFunction)
                    {
                      double gaussIteratedIntegralValue = 0;
                      double referenceIntegralValue     = 0;
                      std::shared_ptr<const utils::ScalarSpatialFunctionReal>
                                          function = functions[iFunction];
                      std::vector<double> functionValuesGaussIterated =
                        (*function)(realQuadPointsGaussIterated);
                      std::vector<double> functionValuesReference =
                        (*function)(realQuadPointsReference);

                      gaussIteratedIntegralValue =
                        std::inner_product(functionValuesGaussIterated.begin(),
                                           functionValuesGaussIterated.end(),
                                           cellJxWGaussIterated.begin(),
                                           0.0);
                      referenceIntegralValue =
                        std::inner_product(functionValuesReference.begin(),
                                           functionValuesReference.end(),
                                           cellJxWReference.begin(),
                                           0.0);

                      const double diff = fabs(gaussIteratedIntegralValue -
                                               referenceIntegralValue);
                      if (diff > std::max(absoluteTolerances[iFunction],
                                          fabs(referenceIntegralValue) *
                                            relativeTolerances[iFunction]))
                        {
                          allFunctionsConverge = false;
                          break;
                        }
                    }
                  if (!allFunctionsConverge)
                    {
                      allFunctionsConvergeInAllCells = false;
                      break;
                    }
                }

              if (allFunctionsConvergeInAllCells)
                {
                  orderVec.push_back(i);
                  iterVec.push_back(j);
                  quadPointsVec.push_back(
                    iteratedGaussQuadratureRule.nPoints());
                }
            }
        }
      utils::throwException(
        orderVec.size() != 0 && iterVec.size() != 0 &&
          quadPointsVec.size() != 0,
        "No eligible pairs found that converges to the same accuracy as the "
        "input reference quadrature grid. Try"
        " again using a higher order1DMax or higher copies1DMax or relaxing the tolerances.");

      size_type smallestNQuadPointInProcIndex =
        std::distance(std::begin(quadPointsVec),
                      std::min_element(std::begin(quadPointsVec),
                                       std::end(quadPointsVec)));

      int                          nProcs;
      int                          err = utils::mpi::MPICommSize(comm, &nProcs);
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      int rank;
      err                = utils::mpi::MPICommRank(comm, &rank);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      std::vector<size_type> optimumOrderInAllProcs(nProcs, 0),
        optimumOrderInAllProcsTmp(nProcs, 0);
      std::vector<size_type> optimumIterInAllProcs(nProcs, 0),
        optimumIterInAllProcsTmp(nProcs, 0);
      std::vector<size_type> optimumQuadPointsInAllProcs(nProcs, 0),
        optimumQuadPointsInAllProcsTmp(nProcs, 0);

      optimumOrderInAllProcsTmp[rank] = orderVec[smallestNQuadPointInProcIndex];
      optimumIterInAllProcsTmp[rank]  = iterVec[smallestNQuadPointInProcIndex];
      optimumQuadPointsInAllProcsTmp[rank] =
        quadPointsVec[smallestNQuadPointInProcIndex];

      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        optimumOrderInAllProcsTmp.data(),
        optimumOrderInAllProcs.data(),
        optimumOrderInAllProcsTmp.size(),
        utils::mpi::MPIUnsigned,
        utils::mpi::MPISum,
        comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        optimumIterInAllProcsTmp.data(),
        optimumIterInAllProcs.data(),
        optimumIterInAllProcsTmp.size(),
        utils::mpi::MPIUnsigned,
        utils::mpi::MPISum,
        comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        optimumQuadPointsInAllProcsTmp.data(),
        optimumQuadPointsInAllProcs.data(),
        optimumQuadPointsInAllProcsTmp.size(),
        utils::mpi::MPIUnsigned,
        utils::mpi::MPISum,
        comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      size_type largestNQuadPointInAllProcsIndex =
        std::distance(std::begin(optimumQuadPointsInAllProcs),
                      std::max_element(std::begin(optimumQuadPointsInAllProcs),
                                       std::end(optimumQuadPointsInAllProcs)));

      size_type order =
        optimumOrderInAllProcs[largestNQuadPointInAllProcsIndex];
      size_type copies =
        optimumIterInAllProcs[largestNQuadPointInAllProcsIndex];

      d_numCells = numCells;
      std::shared_ptr<const QuadratureRule> quadratureRule =
        std::make_shared<const QuadratureRuleGaussIterated>(d_dim,
                                                            order,
                                                            copies);

      // std::cout << "Chosen pairs are: "<< order << "," << copies << " Num
      // Quad Pts: " <<
      // optimumQuadPointsInAllProcs[largestNQuadPointInAllProcsIndex] <<
      // std::endl;

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


    const QuadratureRuleAttributes &
    QuadratureRuleContainer::getQuadratureRuleAttributes() const
    {
      return d_quadratureRuleAttributes;
    }

    size_type
    QuadratureRuleContainer::nCells() const
    {
      return d_numCells;
    }

    const std::vector<dftefe::utils::Point> &
    QuadratureRuleContainer::getRealPoints() const
    {
      return d_realPoints;
    }

    std::vector<dftefe::utils::Point>
    QuadratureRuleContainer::getCellRealPoints(const unsigned int cellId) const
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
    QuadratureRuleContainer::getCellParametricPoints(
      const unsigned int cellId) const
    {
      return d_quadratureRuleVec[cellId]->getPoints();
    }

    const std::vector<double> &
    QuadratureRuleContainer::getCellQuadratureWeights(
      const unsigned int cellId) const
    {
      return d_quadratureRuleVec[cellId]->getWeights();
    }

    const std::vector<double> &
    QuadratureRuleContainer::getJxW() const
    {
      return d_JxW;
    }

    std::vector<double>
    QuadratureRuleContainer::getCellJxW(const unsigned int cellId) const
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
    QuadratureRuleContainer::getQuadratureRule(const unsigned int cellId) const
    {
      return *d_quadratureRuleVec[cellId];
    }

    size_type
    QuadratureRuleContainer::nQuadraturePoints() const
    {
      return d_numQuadPoints;
    }

    size_type
    QuadratureRuleContainer::nCellQuadraturePoints(
      const unsigned int cellId) const
    {
      return d_numCellQuadPoints[cellId];
    }

    const std::vector<size_type> &
    QuadratureRuleContainer::getCellQuadStartIds() const
    {
      return d_cellQuadStartIds;
    }

    size_type
    QuadratureRuleContainer::getCellQuadStartId(const size_type cellId) const
    {
      return d_cellQuadStartIds[cellId];
    }

    std::shared_ptr<const basis::TriangulationBase>
    QuadratureRuleContainer::getTriangulation() const
    {
      return d_triangulation;
    }

    const basis::CellMappingBase &
    QuadratureRuleContainer::getCellMapping() const
    {
      return d_cellMapping;
    }

  } // end of namespace quadrature
} // end of namespace dftefe
