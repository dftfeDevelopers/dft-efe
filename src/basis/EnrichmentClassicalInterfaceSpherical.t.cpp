/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Avirup Sircar
 */
#include <utils/PointImpl.h>
#include <basis/L2ProjectionLinearSolverFunction.h>
#include <atoms/SphericalDataNumerical.h>
#include <utils/MemoryStorage.h>
#include <utils/MemoryTransfer.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/CFEOverlapInverseOpContextGLL.h>
#include <utils/ScalarSpatialFunction.h>
#include <linearAlgebra/Defaults.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <algorithm>
#include <basis/EnrichmentDataEvalKernels.h>
#include <set>
#include <string>

namespace dftefe
{
  namespace basis
  {
    namespace
    {
      void
      checkEnrichmentsSpillToBoundary(
        std::shared_ptr<const TriangulationBase> triangulation,
        std::vector<std::vector<global_size_type>>
          overlappingEnrichmentIdsInCells)
      {
        auto cell   = triangulation->beginLocal();
        auto endc   = triangulation->endLocal();
        int  cellId = 0;
        for (; cell != endc; cell++)
          {
            if (overlappingEnrichmentIdsInCells[cellId].size() > 0)
              {
                for (int iFace = 0; iFace < 2 * (*cell)->getDim(); iFace++)
                  {
                    if ((*cell)->isAtBoundary(iFace))
                      {
                        if (!(*cell)->hasPeriodicNeighbor(iFace))
                          {
                            dftefe::utils::Point centerPoint(
                              std::vector<double>(3, 0));
                            (*cell)->center(centerPoint);
                            utils::throwException(
                              false,
                              "Enrichments cannot spill to boundary cells in "
                              "EnrichmentClassicalInterfaceSpherical for non periodic boundary. The cell with center (" +
                                std::to_string(centerPoint[0]) + " , " +
                                std::to_string(centerPoint[1]) + " , " +
                                std::to_string(centerPoint[2]) +
                                ") has enrichment ids which is not possible.");
                          }
                      }
                  }
              }
            cellId++;
          }
      }
    } // namespace
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      EnrichmentClassicalInterfaceSpherical(
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          cfeBasisDataStorageOverlapMatrix,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          cfeBasisDataStorageRhs,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const double                     atomPartitionTolerance,
        const std::vector<std::string> & atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinatesVec,
        const std::string                fieldName,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm,
        const size_type            enrichmentBatchSize)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_enrichmentIdsPartition(nullptr)
      , d_atomIdsPartition(nullptr)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_overlappingEnrichmentIdsInCells(0)
      , d_linAlgOpContext(linAlgOpContext)
      , d_comm(comm)
      , d_enrichBatchSize(enrichmentBatchSize)
    {
      d_isOrthogonalized = true;

      int rank;
      utils::mpi::MPICommRank(comm, &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      int numProcs;
      utils::mpi::MPICommSize(comm, &numProcs);

      if (dim != 3)
        utils::throwException(
          false, "Dimension should be 3 for Spherical Enrichment Dofs.");

      utils::throwException(
        ((cfeBasisDataStorageRhs->getBasisDofHandler()).get() ==
         (cfeBasisDataStorageOverlapMatrix->getBasisDofHandler()).get()),
        "The BasisDofHandler of the dataStorage and basisOverlapOperator should be same in EnrichmentClassicalInterfaceSpherical ");

      d_cfeBasisDofHandler = std::dynamic_pointer_cast<
        const FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>>(
        cfeBasisDataStorageRhs->getBasisDofHandler());
      utils::throwException(
        d_cfeBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in EnrichmentClassicalInterfaceSpherical");

      d_triangulation = d_cfeBasisDofHandler->getTriangulation();

      // no p refinement assumed
      d_feOrder = d_cfeBasisDofHandler->getFEOrder(0);

      // Partition the enriched dofs based on the BCs and Orthogonalized EFE

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto cell = d_triangulation->beginLocal();
      auto endc = d_triangulation->endLocal();

      size_type cellIndex                        = 0;
      int       locallyOwnedCellsInTriangulation = 0;

      for (; cell != endc; cell++)
        {
          (*cell)->getVertices(cellVertices);
          cellVerticesVector.push_back(cellVertices);
          locallyOwnedCellsInTriangulation++;
        }

      utils::throwException(
        d_cfeBasisDofHandler->nLocallyOwnedCells() ==
          locallyOwnedCellsInTriangulation,
        "locallyOwnedCellsInTriangulation does not match to that in dofhandler in EnrichmentClassicalInterface()");

      std::vector<double> minbound;
      std::vector<double> maxbound;
      maxbound.resize(dim, 0);
      minbound.resize(dim, 0);

      for (unsigned int k = 0; k < dim; k++)
        {
          double maxtmp = -DBL_MAX, mintmp = DBL_MAX;
          auto   cellIter = cellVerticesVector.begin();
          for (; cellIter != cellVerticesVector.end(); ++cellIter)
            {
              auto cellVertices = cellIter->begin();
              for (; cellVertices != cellIter->end(); ++cellVertices)
                {
                  if (maxtmp <= *(cellVertices->begin() + k))
                    maxtmp = *(cellVertices->begin() + k);
                  if (mintmp >= *(cellVertices->begin() + k))
                    mintmp = *(cellVertices->begin() + k);
                }
            }
          maxbound[k] = maxtmp;
          minbound[k] = mintmp;
        }

      // Create atomIdsPartition Object.
      d_atomIdsPartition =
        std::make_shared<const AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                      minbound,
                                                      maxbound,
                                                      cellVerticesVector,
                                                      atomPartitionTolerance,
                                                      comm);

      // Create enrichmentIdsPartition Object.
      d_enrichmentIdsPartition = std::make_shared<EnrichmentIdsPartition<dim>>(
        d_atomSphericalDataContainer,
        d_atomIdsPartition,
        atomSymbolVec,
        atomCoordinatesVec,
        fieldName,
        minbound,
        maxbound,
        d_triangulation->maxElementLength(),
        d_triangulation->getDomainVectors(),
        d_triangulation->getPeriodicFlags(),
        cellVerticesVector,
        comm);

      d_overlappingEnrichmentIdsInCells =
        d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      checkEnrichmentsSpillToBoundary(d_triangulation,
                                      d_overlappingEnrichmentIdsInCells);

      // For Non-Periodic BC, a sparse vector d_i with hanging with homogenous
      // BC will be formed which will be solved by Md =
      // integrateWithBasisValues( homogeneous BC). Form quadRuleContainer for
      // Pristine enrichment. Form OperatorContext object for OverlapMatrix.
      // Form L2ProjectionLinearSolverContext.
      // Get the multiVector for basisInterfaceCoeff.

      // Form the quadValuesContainer for pristine enrichment N_A
      // quadValuesEnrichmentFunction

      // Find the total number of local and ghost enrichment ids = num
      // Componebts of the quadValuesContainer

      // Create a feBasisManagerObject

      std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
        zeroFunction =
          std::make_shared<dftefe::utils::ScalarZeroFunctionReal>();

      // // Set up BasisManager
      d_cfeBasisManager =
        std::make_shared<dftefe::basis::FEBasisManager<ValueTypeBasisData,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>>(
          d_cfeBasisDofHandler, zeroFunction);

      global_size_type nTotalEnrichmentIds =
        d_enrichmentIdsPartition->nTotalEnrichmentIds();

      d_enrichmentIdToClassicalLocalIdMap.clear();
      d_enrichmentIdToInterfaceCoeffMap.clear();

      std::vector<double> quadValuesInAllCellsEnrichment, quadGradientsInAllCellsEnrichment;
      getEnrichmentDataInAllCellsAtQuadPts(
        true,
        false,
        *cfeBasisDataStorageRhs->getQuadratureRuleContainer(),
        quadValuesInAllCellsEnrichment,
        quadGradientsInAllCellsEnrichment,
        *linAlgOpContext);

      for (global_size_type enrichStartId = 0;
           enrichStartId < nTotalEnrichmentIds;
           enrichStartId += d_enrichBatchSize)
        {
          const size_type enrichEndId =
            std::min(enrichStartId + d_enrichBatchSize, nTotalEnrichmentIds);
          const size_type numEnrichInBatch = enrichEndId - enrichStartId;

          std::shared_ptr<
            linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>
            basisInterfaceCoeff = std::make_shared<
              linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>(
              d_cfeBasisManager->getMPIPatternP2P(),
              linAlgOpContext,
              numEnrichInBatch,
              ValueTypeBasisData());

          quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace>
            quadValuesEnrichmentFunction(
              cfeBasisDataStorageRhs->getQuadratureRuleContainer(),
              numEnrichInBatch,
              (ValueTypeBasisData)0.0);

          quadrature::QuadratureValuesContainer<ValueTypeBasisData,
                                                utils::MemorySpace::HOST>
            quadValuesEnrichmentFunctionHost(
              cfeBasisDataStorageRhs->getQuadratureRuleContainer(),
              numEnrichInBatch,
              (ValueTypeBasisData)0.0);

          const size_type numLocallyOwnedCells =
            d_cfeBasisDofHandler->nLocallyOwnedCells();
          std::vector<size_type> nQuadPointsInCell(0);
          nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
          cellIndex = 0;
          auto locallyOwnedCellIter =
            d_cfeBasisDofHandler->beginLocallyOwnedCells();
          ValueTypeBasisData *quadValuesEnrichmentFunctionPtr =
            quadValuesEnrichmentFunctionHost.begin();
          size_type cumulativeQuadEnrichInCell = 0;
          size_type cumulativeQuadPerEnrichPerCell = 0;
          const double *quadValuesInAllCellsEnrichmentPtr = quadValuesInAllCellsEnrichment.data();

          for (; locallyOwnedCellIter !=
                 d_cfeBasisDofHandler->endLocallyOwnedCells();
               ++locallyOwnedCellIter)
            {
              size_type nQuadPointInCell =
                cfeBasisDataStorageRhs->getQuadratureRuleContainer()
                  ->nCellQuadraturePoints(cellIndex);
              // std::vector<utils::Point> quadRealPointsVec =
              //   cfeBasisDataStorageRhs->getQuadratureRuleContainer()
              //     ->getCellRealPoints(cellIndex);
              const auto &enrichInCell = d_overlappingEnrichmentIdsInCells[cellIndex];
              const size_type numEnrichInCell = enrichInCell.size();
              if (numEnrichInCell > 0)
                {
                  for (int iEnrichInCell = 0 ; 
                        iEnrichInCell < numEnrichInCell ; iEnrichInCell++)
                    {
                      if (enrichInCell[iEnrichInCell] >= enrichStartId &&
                          enrichInCell[iEnrichInCell] < enrichEndId)
                        {
                          // basis::EnrichmentIdAttribute eIdAttr =
                          //   d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                          //     enrichmentId);
                          // utils::Point origin(
                          //   d_atomCoordinatesVec[eIdAttr.atomId]);
                          // auto sphericalData =
                          //   d_atomSphericalDataContainer->getSphericalData(
                          //     d_atomSymbolVec[eIdAttr.atomId],
                          //     d_fieldName)[eIdAttr.localIdInAtom];

                          for (unsigned int qPoint = 0;
                               qPoint < nQuadPointInCell;
                               qPoint++)
                            {
                              quadValuesEnrichmentFunctionPtr
                                [cumulativeQuadEnrichInCell +
                                 numEnrichInBatch * qPoint + enrichInCell[iEnrichInCell] -
                                 enrichStartId] = 
                                  quadValuesInAllCellsEnrichmentPtr[cumulativeQuadPerEnrichPerCell + nQuadPointInCell * iEnrichInCell + qPoint];
                                  // sphericalData->getValue(
                                  //   quadRealPointsVec[qPoint], origin);
                            }
                        }
                    }
                    cumulativeQuadPerEnrichPerCell += nQuadPointInCell * numEnrichInCell;
                }
              cumulativeQuadEnrichInCell += nQuadPointInCell * numEnrichInBatch;
              cellIndex = cellIndex + 1;
            }

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;
          memoryTransfer.copy(quadValuesEnrichmentFunction.nEntries(),
                              quadValuesEnrichmentFunction.begin(),
                              quadValuesEnrichmentFunctionHost.begin());

          // Create OperatorContext for CFEBasisoverlap
          std::shared_ptr<
            const dftefe::basis::CFEOverlapOperatorContext<ValueTypeBasisData,
                                                           ValueTypeBasisData,
                                                           memorySpace,
                                                           dim>>
            cfeBasisOverlapOperator = std::make_shared<
              dftefe::basis::CFEOverlapOperatorContext<ValueTypeBasisData,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>>(
              *d_cfeBasisManager,
              *cfeBasisDataStorageOverlapMatrix,
              L2ProjectionDefaults::CELL_BATCH_SIZE,
              numEnrichInBatch,
              linAlgOpContext);

          std::shared_ptr<
            linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                ValueTypeBasisData,
                                                memorySpace>>
            linearSolverFunction = std::make_shared<
              L2ProjectionLinearSolverFunction<ValueTypeBasisData,
                                               ValueTypeBasisData,
                                               memorySpace,
                                               dim>>(
              d_cfeBasisManager,
              cfeBasisOverlapOperator,
              cfeBasisDataStorageRhs,
              quadValuesEnrichmentFunction,
              L2ProjectionDefaults::PC_TYPE,
              linAlgOpContext,
              L2ProjectionDefaults::CELL_BATCH_SIZE,
              numEnrichInBatch);

          linearAlgebra::LinearAlgebraProfiler profiler;

          std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                          ValueTypeBasisData,
                                                          memorySpace>>
            CGSolve =
              std::make_shared<linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                             ValueTypeBasisData,
                                                             memorySpace>>(
                L2ProjectionDefaults::MAX_ITER,
                L2ProjectionDefaults::ABSOLUTE_TOL,
                L2ProjectionDefaults::RELATIVE_TOL,
                L2ProjectionDefaults::DIVERGENCE_TOL,
                profiler);

          linearAlgebra::LinearSolverError errLS;
          errLS = CGSolve->solve(*linearSolverFunction);
          linearSolverFunction->getSolution(*basisInterfaceCoeff);

          if (errLS.err != linearAlgebra::LinearSolverErrorCode::SUCCESS)
          {
            rootCout << errLS.msg << std::endl << std::flush;
            utils::throwException(
              false, "CG solve for orthogonalization was not successful.");
          }

          /**
          // Can also do via the M^(-1) route withot solving CG.

          linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace> d(
                d_cfeBasisManager->getMPIPatternP2P(),
                linAlgOpContext,
                nTotalEnrichmentIds);
          d.setValue(0.0);

          FEBasisOperations<ValueTypeBasisData, ValueTypeBasisData, memorySpace,
          dim> cfeBasisOperations(cfeBasisDataStorageRhs,
          L2ProjectionDefaults::CELL_BATCH_SIZE, nTotalEnrichmentIds; rootCout
          << "Begin creating integrateWithBasisValues\n";
          // Integrate this with different quarature rule. (i.e. adaptive for
          the
          // enrichment functions) , inp will be in adaptive grid
          cfeBasisOperations.integrateWithBasisValues(quadValuesEnrichmentFunction,
                                                     *d_cfeBasisManager,
                                                     d);
          rootCout << "End creating integrateWithBasisValues\n";
          utils::throwException(
            cfeBasisDataStorageOverlapMatrix->getQuadratureRuleContainer()
                ->getQuadratureRuleAttributes()
                .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
            "The quadrature rule for integration of Classical FE dofs has to be
          GLL if Mc = d" "is not solved via a poisson solve. Contact developers
          if extra options are needed.");

          std::shared_ptr<dftefe::linearAlgebra::OperatorContext<ValueTypeBasisData,
                                                       ValueTypeBasisData,
                                                       memorySpace>> MInvContext
                                                       =
          std::make_shared<dftefe::basis::CFEOverlapInverseOpContextGLL<ValueTypeBasisData,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>>
                                                       (*d_cfeBasisManager,
                                                        *cfeBasisDataStorageOverlapMatrix,
                                                        linAlgOpContext);

          MInvContext->apply(d,*basisInterfaceCoeff, true, true);
          **/


          // populate an unordered_map<id, <vec1, vec2>>  i.e. map from
          // enrichedId
          // -> pair(localId, coeff)

          std::vector<ValueTypeBasisData> basisInterfaceCoeffSTL(
            numEnrichInBatch * d_cfeBasisManager->nLocal(),
            ValueTypeBasisData());

          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            numEnrichInBatch * d_cfeBasisManager->nLocal(),
            basisInterfaceCoeffSTL.data(),
            basisInterfaceCoeff->data());

          std::unordered_map<global_size_type, std::set<size_type>>
            enrichmentIdToClassicalLocalIdMapSet;
          enrichmentIdToClassicalLocalIdMapSet.clear();

          for (size_type i = 0; i < d_cfeBasisManager->nLocal(); i++)
            {
              for (global_size_type j = 0; j < numEnrichInBatch; j++)
                {
                  if (std::abs(*(basisInterfaceCoeffSTL.data() +
                                 i * numEnrichInBatch + j)) > ECIDefaults::ENRICHMENT_ORTHO_COEFF_TOL)
                    {
                      enrichmentIdToClassicalLocalIdMapSet[j + enrichStartId]
                        .insert(i);
                      d_enrichmentIdToInterfaceCoeffMap[j + enrichStartId]
                        .push_back(*(basisInterfaceCoeffSTL.data() +
                                     i * numEnrichInBatch + j));
                    }
                }
            }

          for (auto i = enrichmentIdToClassicalLocalIdMapSet.begin();
               i != enrichmentIdToClassicalLocalIdMapSet.end();
               i++)
            {
              d_enrichmentIdToClassicalLocalIdMap[i->first] =
                utils::OptimizedIndexSet<size_type>(i->second);
            }

          rootCout << "Orthogonalized Enrichment Ids : " << enrichStartId
                   << " to " << enrichEndId - 1 << "\n";
        }

      //// ------------optimization----------
      //// Another approach is to fix it in enrichmentIdsPartition using
      //// dealii::neighbour()
      //// get cellId of the localId and create
      //// overlappingEnrichmentIdsInCells
      //// change the ghostids based on those enrichment ids i.e. the
      //// partitioning

      std::vector<std::vector<global_size_type>>
        overlappingEnrichmentIdsInCells(
          d_overlappingEnrichmentIdsInCells.size(),
          std::vector<global_size_type>(0));

      cellIndex = 0;
      std::vector<size_type> vecLocalNodeId(0);

      cell = d_triangulation->beginLocal();
      endc = d_triangulation->endLocal();

      for (; cell != endc; cell++)
        {
          d_cfeBasisManager->getCellDofsLocalIds(cellIndex, vecLocalNodeId);
          for (auto &pair : d_enrichmentIdToClassicalLocalIdMap)
            {
              for (auto &iCellLocalId : vecLocalNodeId)
                {
                  size_type pos   = 0;
                  bool      found = false;
                  pair.second.getPosition(iCellLocalId, pos, found);
                  if (found /*pair.second.find(iCellLocalId) != pair.second.end()*/)
                    {
                      overlappingEnrichmentIdsInCells[cellIndex].push_back(
                        pair.first);
                      DFTEFE_AssertWithMsg(
                        std::find(
                          d_overlappingEnrichmentIdsInCells[cellIndex].begin(),
                          d_overlappingEnrichmentIdsInCells[cellIndex].end(),
                          pair.first) !=
                          d_overlappingEnrichmentIdsInCells[cellIndex].end(),
                        "The enrichment ids were not there in "
                        " overlapping enrichmentIds in cells with larger ball"
                        "radius.");
                      break;
                    }
                }
            }
          cellIndex++;
        }
      d_enrichmentIdsPartition->modifyNumCellsOverlapWithEnrichments(
        overlappingEnrichmentIdsInCells);

      d_overlappingEnrichmentIdsInCells.resize(0);
      d_overlappingEnrichmentIdsInCells =
        d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      global_size_type maxEnrich = 0;
      global_size_type minEnrich = 0;
      global_size_type avgEnrich = 0;
      global_size_type maxTotalEnrichInProc =
        d_enrichmentIdsPartition->nLocalEnrichmentIds();
      cell      = d_triangulation->beginLocal();
      endc      = d_triangulation->endLocal();
      cellIndex = 0;
      for (; cell != endc; cell++)
        {
          global_size_type numEnrichInCell =
            d_overlappingEnrichmentIdsInCells[cellIndex].size();
          if (maxEnrich < numEnrichInCell)
            maxEnrich = numEnrichInCell;
          if (minEnrich > numEnrichInCell)
            minEnrich = numEnrichInCell;
          avgEnrich += numEnrichInCell;
          cellIndex++;
        }

      avgEnrich /= d_cfeBasisDofHandler->nLocallyOwnedCells();

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &maxEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &minEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMin,
        comm);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &avgEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &maxTotalEnrichInProc,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      rootCout << "Maximum " << fieldName
               << " Enrichment Ids In a Cell in a Processor: " << maxEnrich
               << "\n";
      rootCout << "Minimum " << fieldName
               << " Enrichment Ids In a Cell in a Processor: " << minEnrich
               << "\n";
      rootCout << "Max Average " << fieldName
               << " Enrichment Ids In a Cell in a Processor: " << avgEnrich
               << "\n";
      rootCout << "Maximum " << fieldName
               << " Total Enrichment Ids In a Processor: "
               << maxTotalEnrichInProc << "\n";

      rootCout
        << "Completed creating Orthogonalized EnrichmentClassicalInterfaceSpherical for "
        << d_enrichmentIdsPartition->nTotalEnrichmentIds() << " " << fieldName
        << " enrichments." << std::endl;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      EnrichmentClassicalInterfaceSpherical(
        std::shared_ptr<const TriangulationBase> triangulation,
        size_type                                feOrder,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const double                     atomPartitionTolerance,
        const std::vector<std::string> & atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinatesVec,
        const std::string                fieldName,
        const utils::mpi::MPIComm &      comm)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_enrichmentIdsPartition(nullptr)
      , d_atomIdsPartition(nullptr)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_triangulation(triangulation)
      , d_overlappingEnrichmentIdsInCells(0)
      , d_linAlgOpContext(nullptr)
      , d_feOrder(feOrder)
      , d_comm(comm)
    {
      d_isOrthogonalized = false;

      int rank;
      utils::mpi::MPICommRank(comm, &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      int numProcs;
      utils::mpi::MPICommSize(comm, &numProcs);

      if (dim != 3)
        utils::throwException(
          false, "Dimension should be 3 for Spherical Enrichment Dofs.");

      // Partition the enriched dofs with pristine enrichment

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto                                   cell = triangulation->beginLocal();
      auto                                   endc = triangulation->endLocal();

      int locallyOwnedCellsInTriangulation = 0;

      for (; cell != endc; cell++)
        {
          (*cell)->getVertices(cellVertices);
          cellVerticesVector.push_back(cellVertices);
          locallyOwnedCellsInTriangulation++;
        }

      std::vector<double> minbound;
      std::vector<double> maxbound;
      maxbound.resize(dim, 0);
      minbound.resize(dim, 0);

      for (unsigned int k = 0; k < dim; k++)
        {
          double maxtmp = -DBL_MAX, mintmp = DBL_MAX;
          auto   cellIter = cellVerticesVector.begin();
          for (; cellIter != cellVerticesVector.end(); ++cellIter)
            {
              auto cellVertices = cellIter->begin();
              for (; cellVertices != cellIter->end(); ++cellVertices)
                {
                  if (maxtmp <= *(cellVertices->begin() + k))
                    maxtmp = *(cellVertices->begin() + k);
                  if (mintmp >= *(cellVertices->begin() + k))
                    mintmp = *(cellVertices->begin() + k);
                }
            }
          maxbound[k] = maxtmp;
          minbound[k] = mintmp;
        }

      // Create atomIdsPartition Object.
      d_atomIdsPartition =
        std::make_shared<const AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                      minbound,
                                                      maxbound,
                                                      cellVerticesVector,
                                                      atomPartitionTolerance,
                                                      comm);

      // Create enrichmentIdsPartition Object.
      d_enrichmentIdsPartition = std::make_shared<EnrichmentIdsPartition<dim>>(
        d_atomSphericalDataContainer,
        d_atomIdsPartition,
        atomSymbolVec,
        atomCoordinatesVec,
        fieldName,
        minbound,
        maxbound,
        0,
        triangulation->getDomainVectors(),
        d_triangulation->getPeriodicFlags(),
        cellVerticesVector,
        comm);

      d_overlappingEnrichmentIdsInCells =
        d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      global_size_type maxEnrich = 0;
      global_size_type minEnrich = 0;
      global_size_type avgEnrich = 0;
      global_size_type maxTotalEnrichInProc =
        d_enrichmentIdsPartition->nLocalEnrichmentIds();
      size_type cellIndex = 0;
      cell                = d_triangulation->beginLocal();
      for (; cell != endc; cell++)
        {
          global_size_type numEnrichInCell =
            d_overlappingEnrichmentIdsInCells[cellIndex].size();
          if (maxEnrich < numEnrichInCell)
            maxEnrich = numEnrichInCell;
          if (minEnrich > numEnrichInCell)
            minEnrich = numEnrichInCell;
          avgEnrich += numEnrichInCell;
          cellIndex++;
        }

      avgEnrich /= locallyOwnedCellsInTriangulation;

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &maxEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &minEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMin,
        comm);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &avgEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPISum,
        comm);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &maxTotalEnrichInProc,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      avgEnrich /= numProcs;

      rootCout << "Maximum " << fieldName
               << " Enrichment Ids In a Cell in Processor: " << maxEnrich
               << "\n";
      rootCout << "Minimum " << fieldName
               << " Enrichment Ids In a Cell in Processor: " << minEnrich
               << "\n";
      rootCout << "Average " << fieldName
               << " Enrichment Ids In a Cell Processor: " << avgEnrich << "\n";
      rootCout << "Maximum " << fieldName
               << " Total Enrichment Ids In a Processor: "
               << maxTotalEnrichInProc << "\n";

      rootCout
        << "Completed creating Pristine EnrichmentClassicalInterfaceSpherical for "
        << d_enrichmentIdsPartition->nTotalEnrichmentIds() << " " << fieldName
        << " enrichments." << std::endl;

      checkEnrichmentsSpillToBoundary(d_triangulation,
                                      d_overlappingEnrichmentIdsInCells);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const atoms::AtomSphericalDataContainer>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getAtomSphericalDataContainer()
      const
    {
      return d_atomSphericalDataContainer;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const EnrichmentIdsPartition<dim>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getEnrichmentIdsPartition()
      const
    {
      return d_enrichmentIdsPartition;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const AtomIdsPartition<dim>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getAtomIdsPartition() const
    {
      return d_atomIdsPartition;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const BasisManager<ValueTypeBasisData, memorySpace>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getCFEBasisManager() const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getCFEBasisManager() for no orthogonalization of EFE mesh.");

      return d_cfeBasisManager;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const BasisDofHandler>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getCFEBasisDofHandler() const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getCFEBasisDofHandler() for no orthogonalization of EFE mesh.");

      return d_cfeBasisDofHandler;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const std::unordered_map<global_size_type,
                             utils::OptimizedIndexSet<size_type>> &
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getClassicalComponentLocalIdsMap() const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getEnrichmentIdToClassicalLocalIdMap() for no orthogonalization of EFE mesh.");

      return d_enrichmentIdToClassicalLocalIdMap;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const std::unordered_map<global_size_type,
                             std::vector<ValueTypeBasisData>> &
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getClassicalComponentCoeffMap()
      const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getEnrichmentIdToClassicalLocalIdCoeffMap() for no orthogonalization of EFE mesh.");

      return d_enrichmentIdToInterfaceCoeffMap;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<ValueTypeBasisData>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getClassicalComponentCoeffsInCellOEFE(const size_type    
                                             cellIndex) const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getEnrichmentIdToClassicalLocalIdCoeffMap() for no orthogonalization of EFE mesh.");

        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;

        cfeBasisManager =
          std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                         ValueTypeBasisData,
                                                         memorySpace,
                                                         dim>>(
            getCFEBasisManager());

        enrichmentIdToClassicalLocalIdMap = &(getClassicalComponentLocalIdsMap());

        enrichmentIdToInterfaceCoeffMap = &(getClassicalComponentCoeffMap());

        std::vector<size_type> vecClassicalLocalNodeId(0);

        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((d_feOrder + 1),
                                            dim);

          const auto &enrichInCellVec = d_overlappingEnrichmentIdsInCells[cellIndex];

        size_type numEnrichmentIdsInCell = enrichInCellVec.size();                                            
        std::vector<ValueTypeBasisData> coeffsInCell(classicalDofsPerCell *
                                                       numEnrichmentIdsInCell,
                                                     0);

          cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                              vecClassicalLocalNodeId);

          for (size_type cellEnrichId = 0; cellEnrichId < numEnrichmentIdsInCell;
              cellEnrichId++)
            {
              // get the enrichmentIds
              global_size_type enrichmentId = enrichInCellVec[cellEnrichId];

            // get the vectors of non-zero localIds and coeffs
            auto iter = enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
            auto it   = enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
            if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                it != enrichmentIdToClassicalLocalIdMap->end())
              {
                const std::vector<ValueTypeBasisData> &coeffsInLocalIdsMap =
                  iter->second;

                for (size_type i = 0; i < classicalDofsPerCell; i++)
                  {
                    size_type pos   = 0;
                    bool      found = false;
                    it->second.getPosition(vecClassicalLocalNodeId[i],
                                           pos,
                                           found);
                    if (found)
                      {
                        coeffsInCell[numEnrichmentIdsInCell * i +
                                     cellEnrichId] = coeffsInLocalIdsMap[pos];
                      }
                  }
              }
          }
        return coeffsInCell;
      }


    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<ValueTypeBasisData>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getClassicalComponentCoeffsInAllCellsOEFE() const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getEnrichmentIdToClassicalLocalIdCoeffMap() for no orthogonalization of EFE mesh.");

        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;

        cfeBasisManager =
          std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                         ValueTypeBasisData,
                                                         memorySpace,
                                                         dim>>(
            getCFEBasisManager());

        enrichmentIdToClassicalLocalIdMap =
          &(getClassicalComponentLocalIdsMap());

        enrichmentIdToInterfaceCoeffMap =
          &(getClassicalComponentCoeffMap());

        std::vector<size_type> vecClassicalLocalNodeId(0);
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((d_feOrder + 1),
                                            dim);
                
        size_type coeffInAllCellsVecSize = 0 , cumulativeCoeffsInCell = 0;                                    
        for(int cellIndex = 0 ; cellIndex < d_overlappingEnrichmentIdsInCells.size() ; cellIndex++)
           coeffInAllCellsVecSize += d_overlappingEnrichmentIdsInCells[cellIndex].size();
        coeffInAllCellsVecSize *= classicalDofsPerCell;
          
        std::vector<ValueTypeBasisData> coeffsInAllCells(coeffInAllCellsVecSize,
                                                     0);

        for(int cellIndex = 0 ; cellIndex < d_overlappingEnrichmentIdsInCells.size() ; cellIndex++)
        {
          const auto &enrichInCellVec = d_overlappingEnrichmentIdsInCells[cellIndex];
          cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                              vecClassicalLocalNodeId);

          size_type numEnrichmentIdsInCell = enrichInCellVec.size();

          for (size_type cellEnrichId = 0; cellEnrichId < numEnrichmentIdsInCell;
              cellEnrichId++)
            {
              // get the enrichmentIds
              global_size_type enrichmentId = enrichInCellVec[cellEnrichId];

              // get the vectors of non-zero localIds and coeffs
              auto iter = enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
              auto it   = enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
              if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                  it != enrichmentIdToClassicalLocalIdMap->end())
                {
                  const std::vector<ValueTypeBasisData> &coeffsInLocalIdsMap =
                    iter->second;

                  for (size_type i = 0; i < classicalDofsPerCell; i++)
                    {
                      size_type pos   = 0;
                      bool      found = false;
                      it->second.getPosition(vecClassicalLocalNodeId[i],
                                            pos,
                                            found);
                      if (found)
                        {
                          coeffsInAllCells[cumulativeCoeffsInCell + numEnrichmentIdsInCell * i +
                                      cellEnrichId] = coeffsInLocalIdsMap[pos];
                        }
                    }
                }
            }
            cumulativeCoeffsInCell += numEnrichmentIdsInCell * classicalDofsPerCell;
          }
        return coeffsInAllCells;
      }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getLinAlgOpContext() const
    {
      if (!d_isOrthogonalized)
        utils::throwException(
          false,
          "Cannot call getLinAlgOpContext() for no orthogonalization of EFE mesh.");

      return d_linAlgOpContext;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::isOrthogonalized() const
    {
      return d_isOrthogonalized;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<std::string>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getAtomSymbolVec() const
    {
      return d_atomSymbolVec;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<utils::Point>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getAtomCoordinatesVec() const
    {
      return d_atomCoordinatesVec;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::string
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getFieldName() const
    {
      return d_fieldName;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const TriangulationBase>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getTriangulation() const
    {
      return d_triangulation;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    size_type
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::getFEOrder() const
    {
      return d_feOrder;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    global_size_type
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentId(size_type cellId,
                            size_type enrichmentCellLocalId) const
    {
      global_size_type enrichmentId = UINT_MAX;
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              enrichmentCellLocalId)
            {
              enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [enrichmentCellLocalId];
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cell does not have any enrichment ids overlapping with it.");
        }
      return enrichmentId;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    size_type
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentLocalId(size_type cellId,
                                 size_type enrichmentCellLocalId) const
    {
      global_size_type enrichmentId      = UINT_MAX;
      size_type        enrichmentLocalId = UINT_MAX;
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              enrichmentCellLocalId)
            {
              enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [enrichmentCellLocalId];
              enrichmentLocalId = getEnrichmentLocalId(enrichmentId);
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cell does not have any enrichment ids overlapping with it.");
        }
      return enrichmentLocalId;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    size_type
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentLocalId(global_size_type enrichmentId) const
    {
      size_type        enrichmentLocalId = UINT_MAX;
      global_size_type locallyOwnedIdsBegin =
        d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().first;
      global_size_type locallyOwnedIdsEnd =
        d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().second;

      if (enrichmentId < locallyOwnedIdsEnd &&
          enrichmentId >= locallyOwnedIdsBegin)
        enrichmentLocalId = enrichmentId - locallyOwnedIdsBegin;

      else
        {
          int c = 0;
          for (auto it : d_enrichmentIdsPartition->ghostEnrichmentIds())
            {
              if (it == enrichmentId)
                {
                  return c;
                  break;
                }
              else
                c += 1;
            }
          // auto it =
          // std::find(d_enrichmentIdsPartition->ghostEnrichmentIds().begin(),
          //   d_enrichmentIdsPartition->ghostEnrichmentIds().end(),
          //   enrichmentId);
          // if(it != d_enrichmentIdsPartition->ghostEnrichmentIds().end())
          //   enrichmentLocalId = locallyOwnedIdsEnd + it -
          //   d_enrichmentIdsPartition->ghostEnrichmentIds().begin();
          if (c == d_enrichmentIdsPartition->ghostEnrichmentIds().size())
            {
              utils::throwException(
                false,
                "The requested enrichmentId is not found in the locally owned or ghost set." +
                  std::to_string(enrichmentId) + " " +
                  std::to_string(locallyOwnedIdsBegin) + " " +
                  std::to_string(locallyOwnedIdsEnd) + " " +
                  std::to_string(
                    d_enrichmentIdsPartition->ghostEnrichmentIds()[0]));
            }
        }
      return enrichmentLocalId;
    }

    // Enrichment functions with dealii mesh. The enrichedid is the cell local
    // id.
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<double>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      getEnrichmentValue(const size_type                          cellId,
                         const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<global_size_type> enrichIdVec =
        d_overlappingEnrichmentIdsInCells[cellId];
      unsigned int        numEnrichIdsInCell = enrichIdVec.size();
      unsigned int        numPoints          = points.size();
      std::vector<double> retValue(numPoints * numEnrichIdsInCell, 0),
        rVec(numPoints, 0), thetaVec(numPoints, 0), phiVec(numPoints, 0);
      std::vector<dftefe::utils::Point> x(numPoints, utils::Point(dim));
      DFTEFE_AssertWithMsg(
        !enrichIdVec.empty(),
        "The requested cell does not have any enrichment ids.");
      // unsigned int numEnrichedIdsSkipped = 0;
      // unsigned int l                     = 0;

      size_type  atomIdPrev = std::numeric_limits<size_type>::max();
      for (int iEnrich = 0; iEnrich < numEnrichIdsInCell;
           iEnrich += 1 /*numEnrichedIdsSkipped*/)
        {
          basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(
              enrichIdVec[iEnrich]);

          size_type atomId  = eIdAttr.atomId;
          size_type localId = eIdAttr.localIdInAtom;

          if(atomIdPrev != atomId)
          {   
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::transform(points.begin(),
                         points.end(),
                         x.begin(),
                         [origin](utils::Point p) { return p - origin; });

          for(int iPts = 0 ; iPts < points.size() ; iPts++)
            atoms::convertCartesianToSpherical(
              x[iPts],
              rVec[iPts],
              thetaVec[iPts],
              phiVec[iPts],
              atoms::SphericalDataDefaults::POL_ANG_TOL);
          }

          auto sphericalDataVec =
            d_atomSphericalDataContainer->getSphericalData(
              d_atomSymbolVec[atomId], d_fieldName);

          // auto quantumNoVec =
          //   d_atomSphericalDataContainer->getQNumbers(d_atomSymbolVec[atomId],
          //                                             d_fieldName);

          // l = quantumNoVec[localId][1];

          auto radialValue = sphericalDataVec[localId]->getRadialValue(rVec);

          // for (int mCount = 0; mCount < 2 * l + 1; mCount++)
          //   {
          auto angularValue = (sphericalDataVec[localId /*+mCount*/])
                                ->getAngularValue(rVec, thetaVec, phiVec);

          linearAlgebra::blasLapack::hadamardProduct<ValueTypeBasisData,
                                                     ValueTypeBasisData,
                                                     utils::MemorySpace::HOST>(
            numPoints,
            radialValue.data(),
            angularValue.data(),
            retValue.data() + (iEnrich /*+mCount*/) * numPoints,
            *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);
          //   }
          // numEnrichedIdsSkipped = (2 * l + 1);
          atomIdPrev = atomId;
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<double>
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentDerivative(const size_type cellId,
                                    const std::vector<dftefe::utils::Point>
                                      &points) const
    {
      std::vector<global_size_type> enrichIdVec =
        d_overlappingEnrichmentIdsInCells[cellId];
      unsigned int        numEnrichIdsInCell = enrichIdVec.size();
      unsigned int        numPoints          = points.size();
      std::vector<double> retValue(dim * numPoints * numEnrichIdsInCell, 0),
        rVec(numPoints, 0), thetaVec(numPoints, 0), phiVec(numPoints, 0);
      std::vector<dftefe::utils::Point> x(numPoints, utils::Point(dim));
      DFTEFE_AssertWithMsg(
        !enrichIdVec.empty(),
        "The requested cell does not have any enrichment ids.");
      // unsigned int numEnrichedIdsSkipped = 0;
      unsigned int l = 0;

      size_type  atomIdPrev = std::numeric_limits<size_type>::max();

      for (int iEnrich = 0; iEnrich < numEnrichIdsInCell;
           iEnrich += 1 /*numEnrichedIdsSkipped*/)
        {
          basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(
              enrichIdVec[iEnrich]);

          size_type atomId  = eIdAttr.atomId;
          size_type localId = eIdAttr.localIdInAtom;

          if(atomIdPrev != atomId)
          {    
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::transform(points.begin(),
                         points.end(),
                         x.begin(),
                         [origin](utils::Point p) { return p - origin; });

          for(int iPts = 0 ; iPts < points.size() ; iPts++)
            atoms::convertCartesianToSpherical(
              x[iPts],
              rVec[iPts],
              thetaVec[iPts],
              phiVec[iPts],
              atoms::SphericalDataDefaults::POL_ANG_TOL);
          }

          auto sphericalDataVec =
            d_atomSphericalDataContainer->getSphericalData(
              d_atomSymbolVec[atomId], d_fieldName);

          auto quantumNoVec =
            d_atomSphericalDataContainer->getQNumbers(d_atomSymbolVec[atomId],
                                                      d_fieldName);

          l = quantumNoVec[localId][1];

          auto radialValue = sphericalDataVec[localId]->getRadialValue(rVec);
          auto radialDerivative =
            sphericalDataVec[localId]->getRadialDerivative(rVec);

          // for (int mCount = 0; mCount < 2 * l + 1; mCount++)
          //   {
          auto angularValue = (sphericalDataVec[localId /*+mCount*/])
                                ->getAngularValue(rVec, thetaVec, phiVec);
          auto angularDerivative =
            (sphericalDataVec[localId /*+mCount*/])
              ->getAngularDerivative(rVec, thetaVec, phiVec);

          for (int i = 0; i < numPoints; i++)
            {
              double dValueDR        = radialDerivative[i] * angularValue[i];
              double dValueDThetaByr = 0.;
              dValueDThetaByr        = radialValue[i] * angularDerivative[0][i];
              double dValueDPhiByrsinTheta = 0.;
              dValueDPhiByrsinTheta = radialValue[i] * angularDerivative[1][i];
              if ((rVec[i] < 1e-4 && l > 0))
                {
                  dValueDThetaByr =
                    radialDerivative[i] * angularDerivative[0][i] * rVec[i];
                  dValueDPhiByrsinTheta =
                    radialDerivative[i] * angularDerivative[1][i] * rVec[i];
                }
              double theta = thetaVec[i], phi = phiVec[i];

              retValue[(iEnrich /*+mCount*/) * numPoints * dim + i * dim + 0] =
                dValueDR * (sin(theta) * cos(phi)) +
                dValueDThetaByr * (cos(theta) * cos(phi)) -
                sin(phi) * dValueDPhiByrsinTheta;
              retValue[(iEnrich /*+mCount*/) * numPoints * dim + i * dim + 1] =
                dValueDR * (sin(theta) * sin(phi)) +
                dValueDThetaByr * (cos(theta) * sin(phi)) +
                cos(phi) * dValueDPhiByrsinTheta;
              retValue[(iEnrich /*+mCount*/) * numPoints * dim + i * dim + 2] =
                dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));
            }
          //   }
          // numEnrichedIdsSkipped = (2 * l + 1);
          atomIdPrev = atomId;
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<double>
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentHessian(const size_type cellId,
                                 const std::vector<dftefe::utils::Point>
                                   &points) const
    {
      utils::throwException(
        false,
        "getEnrichmentHessian() in EFEBasisDofHandlerDealii is not yet implemented.");
      return std::vector<double>(0);
    }

    // gpu/cpu kernel for calculating the enrichment id values at 
    // all cells in all quad points 
    // for variable quad points also
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      getEnrichmentDataInAllCellsAtQuadPts(bool storeValues, 
                                          bool storeGradients, 
                                          const quadrature::QuadratureRuleContainer &quadRuleContainer, 
                                          std::vector<double> &quadValuesInAllCellsEnrichment,
                                          std::vector<double> &quadGradientsInAllCellsEnrichment,
                                          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext) const
    {
      if(!storeValues && !storeGradients)
      {
        utils::throwException(false, "One or both values and gradient should be true.");
      }
      const auto &localToCellLocalEIds = d_enrichmentIdsPartition->localToCellLocalEIdsVec();
      const auto &cellsInLocalEId = d_enrichmentIdsPartition->cellsInLocalEIdVec();
      size_type numLocalEnrichIds = d_enrichmentIdsPartition->nLocalEnrichmentIds();
      const auto &localToGlobalEnrichmentIds = d_enrichmentIdsPartition->localToGlobalEnrichmentIds();
      const auto &overlappingCellsWithLocalEnrichmentIds = d_enrichmentIdsPartition->overlappingCellsWithLocalEnrichmentIds();
      std::vector<size_type> quadInLocalEId(numLocalEnrichIds, 0);
      std::vector<std::shared_ptr<atoms::SphericalData>> sphericalDataVec(0);

      const size_type numLocallyOwnedCells = d_overlappingEnrichmentIdsInCells.size();

      std::vector<size_type> quadxEnrichPerCell(numLocallyOwnedCells);
      for(int iCell = 0 ; iCell < numLocallyOwnedCells ; iCell++)
      {
        size_type numEnrichInCell =  d_overlappingEnrichmentIdsInCells[iCell].size();
        size_type numQuadInCell = quadRuleContainer.nCellQuadraturePoints(iCell);
        quadxEnrichPerCell[iCell] += numQuadInCell * numEnrichInCell;
      }

      const size_type cumulativeNumPointsInAllEnrich = 
          std::accumulate(quadxEnrichPerCell.begin(), quadxEnrichPerCell.end(), (size_type)0);

      std::vector<double> quadValuesInOverlappingCellsWithLocalEIds(0), quadGradientsInOverlappingCellsWithLocalEIds(0);
      utils::MemoryStorage<double, memorySpace> quadPtsInOverlappingCellsWithLocalEIdsMemSpace(cumulativeNumPointsInAllEnrich * dim),
                  quadValuesInOverlappingCellsWithLocalEIdsMemSpace(0), quadGradientsInOverlappingCellsWithLocalEIdsMemSpace(0);

      if(storeValues)
      {       
        quadValuesInOverlappingCellsWithLocalEIds.resize(cumulativeNumPointsInAllEnrich);
        quadValuesInOverlappingCellsWithLocalEIdsMemSpace.resize(cumulativeNumPointsInAllEnrich);
        quadValuesInAllCellsEnrichment.clear();
        quadValuesInAllCellsEnrichment.resize(cumulativeNumPointsInAllEnrich);
      }
      if(storeGradients)
      {
        quadGradientsInOverlappingCellsWithLocalEIds.resize(cumulativeNumPointsInAllEnrich * dim);
        quadGradientsInOverlappingCellsWithLocalEIdsMemSpace.resize(cumulativeNumPointsInAllEnrich * dim);
        quadGradientsInAllCellsEnrichment.clear();
        quadGradientsInAllCellsEnrichment.resize(cumulativeNumPointsInAllEnrich * dim);
      }

      std::vector<double> origin(numLocalEnrichIds*dim, 0);
      utils::MemoryStorage<double, memorySpace> originMemspace(numLocalEnrichIds*dim, 0);

      const size_type *overlappingCellsWithLocalEnrichmentIdsPtr = overlappingCellsWithLocalEnrichmentIds.data();
      size_type cumulativeQuadInCellPerLocalEnrich = 0;
      for (int iLocalEId = 0 ; iLocalEId < numLocalEnrichIds ; iLocalEId++)
      {
         basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(localToGlobalEnrichmentIds[iLocalEId]);

          size_type atomId  = eIdAttr.atomId;
          size_type localId = eIdAttr.localIdInAtom;
          
          std::memcpy(origin.data() + iLocalEId*dim, d_atomCoordinatesVec[atomId].data(), dim * sizeof(double));

          sphericalDataVec.push_back(d_atomSphericalDataContainer->getSphericalData(
              d_atomSymbolVec[atomId], d_fieldName)[localId]);

        quadInLocalEId[iLocalEId] = 0;
        const size_type numCellInocalEId = cellsInLocalEId[iLocalEId];
        for (int iCell = 0 ; iCell < numCellInocalEId ; iCell++)
        {
          size_type cellId = *(overlappingCellsWithLocalEnrichmentIdsPtr + iCell);
          size_type nCellQuadraturePoints = quadRuleContainer.nCellQuadraturePoints(cellId);
          quadInLocalEId[iLocalEId] += nCellQuadraturePoints;
          linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
                nCellQuadraturePoints * dim,
                quadRuleContainer.template getRealPointsPtr<memorySpace>() +
                  (quadRuleContainer.getCellQuadStartId(cellId) * dim),
                quadPtsInOverlappingCellsWithLocalEIdsMemSpace.data() + cumulativeQuadInCellPerLocalEnrich,
                linAlgOpContext);
          cumulativeQuadInCellPerLocalEnrich += nCellQuadraturePoints * dim;
        }
        overlappingCellsWithLocalEnrichmentIdsPtr += numCellInocalEId;
      }
      
      originMemspace.template copyFrom<utils::MemorySpace::HOST>(origin.data());

      if(storeValues)
      {
        // computation  of the enrichment ids in all points local enrichment wise.
        EnrichmentDataEvalKernels<memorySpace>::getEnrichmentValues(
          numLocalEnrichIds,
          quadInLocalEId,
          sphericalDataVec,
          quadPtsInOverlappingCellsWithLocalEIdsMemSpace.data(),
          originMemspace.data(),
          quadValuesInOverlappingCellsWithLocalEIdsMemSpace.data(),
          linAlgOpContext);

        // transfer values to host 
        quadValuesInOverlappingCellsWithLocalEIdsMemSpace.
          template copyTo<utils::MemorySpace::HOST>(quadValuesInOverlappingCellsWithLocalEIds.data());
      }
      if(storeGradients)
      {
        // computation  of the enrichment ids in all points local enrichment wise.
        EnrichmentDataEvalKernels<memorySpace>::getEnrichmentGradients(
          numLocalEnrichIds,
          quadInLocalEId,
          sphericalDataVec,
          quadPtsInOverlappingCellsWithLocalEIdsMemSpace.data(),
          originMemspace.data(),
          quadGradientsInOverlappingCellsWithLocalEIdsMemSpace.data(),
          linAlgOpContext);

        // transfer values to host 
        quadGradientsInOverlappingCellsWithLocalEIdsMemSpace.
          template copyTo<utils::MemorySpace::HOST>(quadGradientsInOverlappingCellsWithLocalEIds.data());
      }

      if(storeValues)
      {
        // transpose from enrich->cell->quad with overlapping cells to cell->enrich->quad with all cells 
        int cumuLativeQuadPerCellInLocalEnrich = 0; 
        overlappingCellsWithLocalEnrichmentIdsPtr = overlappingCellsWithLocalEnrichmentIds.data();
        const size_type *localToCellLocalEIdsPtr = localToCellLocalEIds.data();
        for (int iLocalEId = 0 ; iLocalEId < numLocalEnrichIds ; iLocalEId++)
        {
          const size_type numCellInocalEId = cellsInLocalEId[iLocalEId];
          for (int iCell = 0 ; iCell < numCellInocalEId ; iCell++)
          {
            size_type cellId = *(overlappingCellsWithLocalEnrichmentIdsPtr + iCell);
            size_type numQuadInCell = quadRuleContainer.nCellQuadraturePoints(cellId);
            size_type cumulativeQuadxEnrichInCell = 
              std::accumulate(quadxEnrichPerCell.begin(), quadxEnrichPerCell.begin()+cellId, (size_type)0);
              std::memcpy(quadValuesInAllCellsEnrichment.data() + cumulativeQuadxEnrichInCell + 
                          *(localToCellLocalEIdsPtr + iCell) * numQuadInCell, // dst 
                          quadValuesInOverlappingCellsWithLocalEIds.data() + cumuLativeQuadPerCellInLocalEnrich, // src 
                          numQuadInCell * sizeof(double));
            cumuLativeQuadPerCellInLocalEnrich += numQuadInCell;
          }
          overlappingCellsWithLocalEnrichmentIdsPtr += numCellInocalEId;
          localToCellLocalEIdsPtr += numCellInocalEId;
        }
      }

      if(storeGradients)
      {
        // transpose from enrich->cell->quad with overlapping cells to cell->enrich->quad with all cells 
        int cumuLativeQuadPerCellInLocalEnrich = 0; 
        overlappingCellsWithLocalEnrichmentIdsPtr = overlappingCellsWithLocalEnrichmentIds.data();
        const size_type *localToCellLocalEIdsPtr = localToCellLocalEIds.data();
        for (int iLocalEId = 0 ; iLocalEId < numLocalEnrichIds ; iLocalEId++)
        {
          const size_type numCellInocalEId = cellsInLocalEId[iLocalEId];
          for (int iCell = 0 ; iCell < numCellInocalEId ; iCell++)
          {
            size_type cellId = *(overlappingCellsWithLocalEnrichmentIdsPtr + iCell);
            size_type numQuadInCell = quadRuleContainer.nCellQuadraturePoints(cellId);
            size_type cumulativeQuadxEnrichInCell = 
              std::accumulate(quadxEnrichPerCell.begin(), quadxEnrichPerCell.begin()+cellId, (size_type)0);
              std::memcpy(quadGradientsInAllCellsEnrichment.data() + cumulativeQuadxEnrichInCell * dim + 
                          *(localToCellLocalEIdsPtr + iCell) * numQuadInCell * dim, // dst 
                          quadGradientsInOverlappingCellsWithLocalEIds.data() + cumuLativeQuadPerCellInLocalEnrich * dim, // src 
                          numQuadInCell * dim * sizeof(double));
            cumuLativeQuadPerCellInLocalEnrich += numQuadInCell;
          }
          overlappingCellsWithLocalEnrichmentIdsPtr += numCellInocalEId;
          localToCellLocalEIdsPtr += numCellInocalEId;
        }
      }
    }
    
  } // namespace basis
} // namespace dftefe
