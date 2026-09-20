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
#include <utils/Profiler.h>

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
        auto      cell   = triangulation->beginLocal();
        auto      endc   = triangulation->endLocal();
        size_type cellId = 0;
        for (; cell != endc; cell++)
          {
            if (overlappingEnrichmentIdsInCells[cellId].size() > 0)
              {
                for (size_type iFace = 0; iFace < 2 * (*cell)->getDim();
                     iFace++)
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
        const size_type            enrichmentBatchSize,
        const size_type            cellBlockSize,
        std::shared_ptr<const PeriodicImageAtomGenerator> imageAtomGenerator)
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
      , d_cellBlockSize(cellBlockSize)
      , d_sphericalDataNumericalFuncPtrVec(nullptr)
    {
      d_isOrthogonalized = true;

      int rank;
      utils::mpi::MPICommRank(comm, &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      int numProcs;
      utils::mpi::MPICommSize(comm, &numProcs);

      utils::Profiler<memorySpace> profiler(
        comm, "EnrichmentClassicalInterfaceSpherical");

      profiler.registerStart("Pratition and Ortho Init");

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
      size_type locallyOwnedCellsInTriangulation = 0;

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

      for (size_type k = 0; k < dim; k++)
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
        comm,
        imageAtomGenerator);

      d_overlappingEnrichmentIdsInCells =
        d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      checkEnrichmentsSpillToBoundary(d_triangulation,
                                      d_overlappingEnrichmentIdsInCells);

      getOverlappingEnrichmentInCellsAdditionalData();

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

      const size_type numLocallyOwnedCells =
        d_cfeBasisDofHandler->nLocallyOwnedCells();
      const auto &quadRuleContainerRef =
        *cfeBasisDataStorageRhs->getQuadratureRuleContainer();

      std::vector<size_type> nQuadPerCell(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
        nQuadPerCell[iCell] = quadRuleContainerRef.nCellQuadraturePoints(iCell);

      size_type maxScratchSize = 0;
      for (size_type cbStart = 0; cbStart < numLocallyOwnedCells;
           cbStart += d_cellBlockSize)
        {
          const size_type cbEnd =
            std::min(cbStart + d_cellBlockSize, numLocallyOwnedCells);
          size_type blockSz = 0;
          for (size_type iCell = cbStart; iCell < cbEnd; iCell++)
            blockSz += d_overlappingEnrichmentIdsInCells[iCell].size() *
                       nQuadPerCell[iCell];
          maxScratchSize = std::max(maxScratchSize, blockSz);
        }

      const utils::mpi::MPIComm &mpiComm =
        d_cfeBasisManager->getMPIPatternP2P()->mpiCommunicator();

      double scratchGB =
        static_cast<double>((maxScratchSize > 0 ? maxScratchSize : 1)) *
        sizeof(double) / (1024.0 * 1024.0 * 1024.0);
      rootCout << "MaxScratchSize = "
               << (maxScratchSize > 0 ? maxScratchSize : 1) << " elements ("
               << scratchGB << " GB)\n";

      utils::MemoryStorage<double, memorySpace> scratch(
        maxScratchSize > 0 ? maxScratchSize : 1);
      utils::printCurrentMemoryUsage<memorySpace>(
        mpiComm, "ECI : After orthogonalization scratch alloc");

      rootCout << "Using enrichBatchSize = " << d_enrichBatchSize << "\n";

      std::shared_ptr<
        linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>
        basisInterfaceCoeff = std::make_shared<
          linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>(
          d_cfeBasisManager->getMPIPatternP2P(),
          linAlgOpContext,
          d_enrichBatchSize,
          ValueTypeBasisData());
      utils::printCurrentMemoryUsage<memorySpace>(
        mpiComm, "ECI : After basisInterfaceCoeff alloc");

      quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace>
        quadValuesEnrichmentFunction(
          cfeBasisDataStorageRhs->getQuadratureRuleContainer(),
          d_enrichBatchSize,
          (ValueTypeBasisData)0.0);
      utils::printCurrentMemoryUsage<memorySpace>(
        mpiComm, "ECI : After quadValuesEnrichmentFunction alloc");

      profiler.registerEnd("Pratition and Ortho Init");

      for (global_size_type enrichStartId = 0;
           enrichStartId < nTotalEnrichmentIds;
           enrichStartId += d_enrichBatchSize)
        {
          const size_type enrichEndId =
            std::min(enrichStartId + d_enrichBatchSize, nTotalEnrichmentIds);
          const size_type numEnrichInBatch = enrichEndId - enrichStartId;

          profiler.registerStart("cellLoop");
          quadValuesEnrichmentFunction.setValue((ValueTypeBasisData)0.0);

          for (size_type cbStart = 0; cbStart < numLocallyOwnedCells;
               cbStart += d_cellBlockSize)
            {
              const size_type cbEnd =
                std::min(cbStart + d_cellBlockSize, numLocallyOwnedCells);
              const std::pair<size_type, size_type> cellRange(cbStart, cbEnd);

              getEnrichmentValuesInCellRangeAtQuadPts(quadRuleContainerRef,
                                                      scratch.data(),
                                                      *linAlgOpContext,
                                                      cellRange);

              size_type scratchOffset = 0;
              for (size_type iCell = cbStart; iCell < cbEnd; iCell++)
                {
                  const auto &enrichInCell =
                    d_overlappingEnrichmentIdsInCells[iCell];
                  const size_type numEnrichInCell = enrichInCell.size();
                  const size_type nQuadInCell     = nQuadPerCell[iCell];
                  for (size_type iEnrich = 0; iEnrich < numEnrichInCell;
                       iEnrich++)
                    {
                      if (enrichInCell[iEnrich] >= enrichStartId &&
                          enrichInCell[iEnrich] < enrichEndId)
                        {
                          const size_type batchId =
                            enrichInCell[iEnrich] - enrichStartId;
                          linearAlgebra::blasLapack::stridedBlockCopy(
                            nQuadInCell,
                            1,
                            numEnrichInCell,
                            iEnrich,
                            d_enrichBatchSize,
                            batchId,
                            scratch.data() + scratchOffset,
                            quadValuesEnrichmentFunction.begin(iCell),
                            *linAlgOpContext);
                        }
                    }
                  scratchOffset += numEnrichInCell * nQuadInCell;
                }
            }

          profiler.registerEnd("cellLoop");
          profiler.registerStart("Class Init");

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
              L2ProjectionDefaults<memorySpace>::CELL_BATCH_SIZE,
              d_enrichBatchSize,
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
              L2ProjectionDefaults<memorySpace>::PC_TYPE,
              linAlgOpContext,
              L2ProjectionDefaults<memorySpace>::CELL_BATCH_SIZE,
              d_enrichBatchSize);

          linearAlgebra::LinearAlgebraProfiler profiler1;

          std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                          ValueTypeBasisData,
                                                          memorySpace>>
            CGSolve =
              std::make_shared<linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                             ValueTypeBasisData,
                                                             memorySpace>>(
                L2ProjectionDefaults<memorySpace>::MAX_ITER,
                L2ProjectionDefaults<memorySpace>::ABSOLUTE_TOL,
                L2ProjectionDefaults<memorySpace>::RELATIVE_TOL,
                L2ProjectionDefaults<memorySpace>::DIVERGENCE_TOL,
                profiler1);

          profiler.registerEnd("Class Init");
          profiler.registerStart("Solve");

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
          L2ProjectionDefaults<memorySpace>::CELL_BATCH_SIZE,
          nTotalEnrichmentIds; rootCout
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

          profiler.registerEnd("Solve");
          profiler.registerStart("Copy D2H and Store maps");

          std::vector<ValueTypeBasisData> basisInterfaceCoeffSTL(
            d_enrichBatchSize * d_cfeBasisManager->nLocal(),
            ValueTypeBasisData());

          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            d_enrichBatchSize * d_cfeBasisManager->nLocal(),
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
                                 i * d_enrichBatchSize + j)) >
                      ECIDefaults::ENRICHMENT_ORTHO_COEFF_TOL)
                    {
                      enrichmentIdToClassicalLocalIdMapSet[j + enrichStartId]
                        .insert(i);
                      d_enrichmentIdToInterfaceCoeffMap[j + enrichStartId]
                        .push_back(*(basisInterfaceCoeffSTL.data() +
                                     i * d_enrichBatchSize + j));
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
          profiler.registerEnd("Copy D2H and Store maps");
        }

      //// ------------optimization----------
      //// Another approach is to fix it in enrichmentIdsPartition using
      //// dealii::neighbour()
      //// get cellId of the localId and create
      //// overlappingEnrichmentIdsInCells
      //// change the ghostids based on those enrichment ids i.e. the
      //// partitioning

      profiler.registerStart("Repartition");
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

      getOverlappingEnrichmentInCellsAdditionalData();

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

      profiler.registerEnd("Repartition");
      profiler.print();
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
        const utils::mpi::MPIComm &      comm,
        std::shared_ptr<const PeriodicImageAtomGenerator> imageAtomGenerator)
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
      , d_sphericalDataNumericalFuncPtrVec(nullptr)
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

      size_type locallyOwnedCellsInTriangulation = 0;

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

      for (size_type k = 0; k < dim; k++)
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
        comm,
        imageAtomGenerator);

      d_overlappingEnrichmentIdsInCells =
        d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      getOverlappingEnrichmentInCellsAdditionalData();

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
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::~EnrichmentClassicalInterfaceSpherical()
    {
      if (d_sphericalDataNumericalFuncPtrVec != nullptr)
        {
          utils::MemoryManager<
            atoms::SphericalDataNumerical::Func<memorySpace>,
            memorySpace>::deallocate(d_sphericalDataNumericalFuncPtrVec);
          d_sphericalDataNumericalFuncPtrVec = nullptr;
        }
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
                                          dim>::
      getClassicalComponentCoeffsInCellOEFE(const size_type cellIndex) const
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
        utils::mathFunctions::sizeTypePow((d_feOrder + 1), dim);

      const auto &enrichInCellVec =
        d_overlappingEnrichmentIdsInCells[cellIndex];

      size_type numEnrichmentIdsInCell = enrichInCellVec.size();
      std::vector<ValueTypeBasisData> coeffsInCell(classicalDofsPerCell *
                                                     numEnrichmentIdsInCell,
                                                   0);

      cfeBasisManager->getCellDofsLocalIds(cellIndex, vecClassicalLocalNodeId);

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
                      coeffsInCell[numEnrichmentIdsInCell * i + cellEnrichId] =
                        coeffsInLocalIdsMap[pos];
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
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
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

      enrichmentIdToClassicalLocalIdMap = &(getClassicalComponentLocalIdsMap());

      enrichmentIdToInterfaceCoeffMap = &(getClassicalComponentCoeffMap());

      std::vector<size_type> vecClassicalLocalNodeId(0);
      size_type              classicalDofsPerCell =
        utils::mathFunctions::sizeTypePow((d_feOrder + 1), dim);

      size_type coeffInAllCellsVecSize = 0, cumulativeCoeffsInCell = 0;
      for (size_type cellIndex = 0;
           cellIndex < d_overlappingEnrichmentIdsInCells.size();
           cellIndex++)
        coeffInAllCellsVecSize +=
          d_overlappingEnrichmentIdsInCells[cellIndex].size();
      coeffInAllCellsVecSize *= classicalDofsPerCell;

      std::vector<ValueTypeBasisData> coeffsInAllCells(coeffInAllCellsVecSize,
                                                       0);

      for (size_type cellIndex = 0;
           cellIndex < d_overlappingEnrichmentIdsInCells.size();
           cellIndex++)
        {
          const auto &enrichInCellVec =
            d_overlappingEnrichmentIdsInCells[cellIndex];
          cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                               vecClassicalLocalNodeId);

          size_type numEnrichmentIdsInCell = enrichInCellVec.size();

          for (size_type cellEnrichId = 0;
               cellEnrichId < numEnrichmentIdsInCell;
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
                          coeffsInAllCells[cumulativeCoeffsInCell +
                                           numEnrichmentIdsInCell * i +
                                           cellEnrichId] =
                            coeffsInLocalIdsMap[pos];
                        }
                    }
                }
            }
          cumulativeCoeffsInCell +=
            numEnrichmentIdsInCell * classicalDofsPerCell;
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
      global_size_type enrichmentId =
        basis::MaxSizeDefaults::GLOBAL_SIZE_TYPE_MAX;
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
      global_size_type enrichmentId =
        basis::MaxSizeDefaults::GLOBAL_SIZE_TYPE_MAX;
      size_type enrichmentLocalId = basis::MaxSizeDefaults::SIZE_TYPE_MAX;
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
      size_type enrichmentLocalId = basis::MaxSizeDefaults::SIZE_TYPE_MAX;
      global_size_type locallyOwnedIdsBegin =
        d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().first;
      global_size_type locallyOwnedIdsEnd =
        d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().second;

      if (enrichmentId < locallyOwnedIdsEnd &&
          enrichmentId >= locallyOwnedIdsBegin)
        enrichmentLocalId = enrichmentId - locallyOwnedIdsBegin;

      else
        {
          size_type c = 0;
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
      size_type           numEnrichIdsInCell = enrichIdVec.size();
      size_type           numPoints          = points.size();
      std::vector<double> retValue(numPoints * numEnrichIdsInCell, 0),
        rVec(numPoints, 0), thetaVec(numPoints, 0), phiVec(numPoints, 0);
      std::vector<dftefe::utils::Point> x(numPoints, utils::Point(dim));
      DFTEFE_AssertWithMsg(
        !enrichIdVec.empty(),
        "The requested cell does not have any enrichment ids.");
      // size_type numEnrichedIdsSkipped = 0;
      // size_type l                     = 0;

      // Sum over each enrichment's origins, swept level by level rather than
      // enrichment by enrichment so that at level 0 consecutive enrichments of
      // one atom share an origin and the transform cache still hits.
      const std::vector<size_type> &extendedAtomIds =
        d_enrichmentIdsPartition->getExtendedAtomIdsForAllEnrichInCell(cellId);
      const std::vector<size_type> &extendedAtomIdOffsets =
        d_enrichmentIdsPartition->getExtendedAtomIdOffsetsForAllEnrichInCell(cellId);

      // The spherical data depends on the species and the quantum numbers, not
      // on where the function is centred, so it is looked up once per
      // enrichment here instead of once per (enrichment, origin) in the sweep.
      std::vector<std::shared_ptr<atoms::SphericalData>>
                sphericalDataPerEnrich(numEnrichIdsInCell);
      size_type maxOriginsInCell = 0;
      for (size_type iEnrich = 0; iEnrich < numEnrichIdsInCell; iEnrich++)
        {
          basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(
              enrichIdVec[iEnrich]);
          sphericalDataPerEnrich[iEnrich] =
            d_atomSphericalDataContainer->getSphericalData(
              d_atomSymbolVec[eIdAttr.atomId],
              d_fieldName)[eIdAttr.localIdInAtom];
          maxOriginsInCell =
            std::max(maxOriginsInCell,
                     extendedAtomIdOffsets[iEnrich + 1] -
                       extendedAtomIdOffsets[iEnrich]);
        }

      size_type extendedAtomIdPrev = std::numeric_limits<size_type>::max();
      for (size_type level = 0; level < maxOriginsInCell; level++)
        {
          for (size_type iEnrich = 0; iEnrich < numEnrichIdsInCell;
               iEnrich += 1 /*numEnrichedIdsSkipped*/)
            {
              // Ragged: this enrichment may have fewer origins than the cell's
              // deepest one.
              if (extendedAtomIdOffsets[iEnrich] + level >=
                  extendedAtomIdOffsets[iEnrich + 1])
                continue;
              const size_type extendedAtomId =
                extendedAtomIds[extendedAtomIdOffsets[iEnrich] + level];

              if (extendedAtomIdPrev != extendedAtomId)
                {
                  utils::Point origin(
                    d_enrichmentIdsPartition->getPositionOfExtendedAtomId(
                      extendedAtomId));
                  std::transform(points.begin(),
                                 points.end(),
                                 x.begin(),
                                 [origin](utils::Point p) {
                                   return p - origin;
                                 });

                  for (size_type iPts = 0; iPts < points.size(); iPts++)
                    atoms::convertCartesianToSpherical(
                      x[iPts],
                      rVec[iPts],
                      thetaVec[iPts],
                      phiVec[iPts],
                      atoms::SphericalDataDefaults::POL_ANG_TOL);

                  extendedAtomIdPrev = extendedAtomId;
                }

              // auto quantumNoVec =
              //   d_atomSphericalDataContainer->getQNumbers(d_atomSymbolVec[atomId],
              //                                             d_fieldName);

              // l = quantumNoVec[localId][1];

              auto radialValue =
                sphericalDataPerEnrich[iEnrich]->getRadialValue(rVec);

              // for (size_type mCount = 0; mCount < 2 * l + 1; mCount++)
              //   {
              auto angularValue =
                sphericalDataPerEnrich[iEnrich /*+mCount*/]->getAngularValue(
                  rVec, thetaVec, phiVec);

              double *out = retValue.data() + (iEnrich /*+mCount*/) * numPoints;
              for (size_type iPts = 0; iPts < numPoints; iPts++)
                out[iPts] += radialValue[iPts] * angularValue[iPts];
              //   }
              // numEnrichedIdsSkipped = (2 * l + 1);
            }
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
      size_type           numEnrichIdsInCell = enrichIdVec.size();
      size_type           numPoints          = points.size();
      std::vector<double> retValue(dim * numPoints * numEnrichIdsInCell, 0),
        rVec(numPoints, 0), thetaVec(numPoints, 0), phiVec(numPoints, 0);
      std::vector<dftefe::utils::Point> x(numPoints, utils::Point(dim));
      DFTEFE_AssertWithMsg(
        !enrichIdVec.empty(),
        "The requested cell does not have any enrichment ids.");
      // size_type numEnrichedIdsSkipped = 0;
      size_type l = 0;

      // An enrichment is the sum over its origins; see getEnrichmentValue
      // above for why the origins are swept level by level rather than
      // enrichment by enrichment. Without periodicity only level 0 exists.
      const std::vector<size_type> &extendedAtomIds =
        d_enrichmentIdsPartition->getExtendedAtomIdsForAllEnrichInCell(cellId);
      const std::vector<size_type> &extendedAtomIdOffsets =
        d_enrichmentIdsPartition->getExtendedAtomIdOffsetsForAllEnrichInCell(cellId);

      // Species and quantum numbers do not vary with the origin, so these are
      // looked up once per enrichment rather than once per (enrichment,
      // origin) in the sweep.
      std::vector<std::shared_ptr<atoms::SphericalData>>
                             sphericalDataPerEnrich(numEnrichIdsInCell);
      std::vector<size_type> lPerEnrich(numEnrichIdsInCell, 0);
      size_type              maxOriginsInCell = 0;
      for (size_type iEnrich = 0; iEnrich < numEnrichIdsInCell; iEnrich++)
        {
          basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(
              enrichIdVec[iEnrich]);
          sphericalDataPerEnrich[iEnrich] =
            d_atomSphericalDataContainer->getSphericalData(
              d_atomSymbolVec[eIdAttr.atomId],
              d_fieldName)[eIdAttr.localIdInAtom];
          lPerEnrich[iEnrich] =
            d_atomSphericalDataContainer->getQNumbers(
              d_atomSymbolVec[eIdAttr.atomId],
              d_fieldName)[eIdAttr.localIdInAtom][1];
          maxOriginsInCell =
            std::max(maxOriginsInCell,
                     extendedAtomIdOffsets[iEnrich + 1] -
                       extendedAtomIdOffsets[iEnrich]);
        }

      size_type extendedAtomIdPrev = std::numeric_limits<size_type>::max();

      for (size_type level = 0; level < maxOriginsInCell; level++)
        {
        for (size_type iEnrich = 0; iEnrich < numEnrichIdsInCell;
             iEnrich += 1 /*numEnrichedIdsSkipped*/)
        {
          // Ragged: this enrichment may have fewer origins than the cell's
          // deepest one.
          if (extendedAtomIdOffsets[iEnrich] + level >=
              extendedAtomIdOffsets[iEnrich + 1])
            continue;
          const size_type extendedAtomId =
            extendedAtomIds[extendedAtomIdOffsets[iEnrich] + level];

          if (extendedAtomIdPrev != extendedAtomId)
            {
              utils::Point origin(
                d_enrichmentIdsPartition->getPositionOfExtendedAtomId(extendedAtomId));
              std::transform(points.begin(),
                             points.end(),
                             x.begin(),
                             [origin](utils::Point p) { return p - origin; });

              for (size_type iPts = 0; iPts < points.size(); iPts++)
                atoms::convertCartesianToSpherical(
                  x[iPts],
                  rVec[iPts],
                  thetaVec[iPts],
                  phiVec[iPts],
                  atoms::SphericalDataDefaults::POL_ANG_TOL);

              extendedAtomIdPrev = extendedAtomId;
            }

          l = lPerEnrich[iEnrich];

          auto radialValue =
            sphericalDataPerEnrich[iEnrich]->getRadialValue(rVec);
          auto radialDerivative =
            sphericalDataPerEnrich[iEnrich]->getRadialDerivative(rVec);

          // for (size_type mCount = 0; mCount < 2 * l + 1; mCount++)
          //   {
          auto angularValue =
            sphericalDataPerEnrich[iEnrich /*+mCount*/]->getAngularValue(
              rVec, thetaVec, phiVec);
          auto angularDerivative =
            sphericalDataPerEnrich[iEnrich /*+mCount*/]->getAngularDerivative(
              rVec, thetaVec, phiVec);

          for (size_type i = 0; i < numPoints; i++)
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

              // retValue is zero initialised, so each origin simply adds its
              // own contribution -- no per-level branch or scratch needed.
              retValue[(iEnrich /*+mCount*/) * numPoints * dim + i * dim + 0] +=
                dValueDR * (sin(theta) * cos(phi)) +
                dValueDThetaByr * (cos(theta) * cos(phi)) -
                sin(phi) * dValueDPhiByrsinTheta;
              retValue[(iEnrich /*+mCount*/) * numPoints * dim + i * dim + 1] +=
                dValueDR * (sin(theta) * sin(phi)) +
                dValueDThetaByr * (cos(theta) * sin(phi)) +
                cos(phi) * dValueDPhiByrsinTheta;
              retValue[(iEnrich /*+mCount*/) * numPoints * dim + i * dim + 2] +=
                dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));
            }
          //   }
          // numEnrichedIdsSkipped = (2 * l + 1);
        }
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
    // for variable quad points
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      getEnrichmentDataInAllCellsAtQuadPts(
        bool                                       storeValues,
        bool                                       storeGradients,
        const quadrature::QuadratureRuleContainer &quadRuleContainer,
        double *basisEnrichQuadStorageStartPtr,
        double *basisGradientEnrichQuadStorageStartPtr,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const size_type                              enrichBlock) const
    {
      if (!storeValues && !storeGradients)
        {
          utils::throwException(
            false, "One or both values and gradient should be true.");
        }

      // utils::Profiler<memorySpace> profiler(d_comm,
      //                                       "getEnrichmentDataInAllCellsAtQuadPts");

      // profiler.registerStart("setup");
      const auto &localToCellLocalEIds =
        d_enrichmentIdsPartition->localToCellLocalEIdsVec();
      const auto &cellsInLocalEId =
        d_enrichmentIdsPartition->cellsInLocalEIdVec();
      size_type numLocalEnrichIds =
        d_enrichmentIdsPartition->nLocalEnrichmentIds();
      const auto &localToGlobalEnrichmentIds =
        d_enrichmentIdsPartition->localToGlobalEnrichmentIds();
      const auto &overlappingCellsWithLocalEnrichmentIds =
        d_enrichmentIdsPartition->overlappingCellsWithLocalEnrichmentIds();

      const size_type numLocallyOwnedCells =
        d_overlappingEnrichmentIdsInCells.size();

      // quadxEnrichPerCell drives the output storage offsets — computed once
      // for all blocks
      std::vector<size_type> quadxEnrichPerCell(numLocallyOwnedCells);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
        {
          quadxEnrichPerCell[iCell] =
            quadRuleContainer.nCellQuadraturePoints(iCell) *
            d_overlappingEnrichmentIdsInCells[iCell].size();
        }

      // cumulative offset of each local enrich id into the flat
      // overlappingCells / localToCellLocalEIds arrays
      std::vector<size_type> enrichCellCumOffset(numLocalEnrichIds + 1, 0);
      for (size_type iLocalEId = 0; iLocalEId < numLocalEnrichIds; iLocalEId++)
        enrichCellCumOffset[iLocalEId + 1] =
          enrichCellCumOffset[iLocalEId] + cellsInLocalEId[iLocalEId];

      // --- sizing pass: find max scratch sizes across all blocks ---
      //
      // The unit handed to the kernel is an ENRICHMENT-ORIGIN PAIR, covering
      // every cell that origin reaches -- the pair's OVERLAP CELLS. Without
      // periodicity there is one pair per enrichment and nothing changes.
      size_type maxTotalQuadPtsForBlock = 0;
      size_type maxEnrichOriginPairsInBatch        = 0;
      for (size_type enrichBlockStart = 0; enrichBlockStart < numLocalEnrichIds;
           enrichBlockStart += enrichBlock)
        {
          const size_type enrichBlockEnd =
            std::min(enrichBlockStart + enrichBlock, numLocalEnrichIds);
          size_type totalQuadPtsThisBlock = 0;
          size_type totalEnrichOriginPairsThisBlock  = 0;
          for (size_type iLocalEId = enrichBlockStart;
               iLocalEId < enrichBlockEnd;
               iLocalEId++)
            {
              const size_type *cellsPtr =
                overlappingCellsWithLocalEnrichmentIds.data() +
                enrichCellCumOffset[iLocalEId];
              const size_type *localToCellPtr =
                localToCellLocalEIds.data() + enrichCellCumOffset[iLocalEId];
              for (size_type iCell = 0; iCell < cellsInLocalEId[iLocalEId];
                   iCell++)
                {
                  const size_type cellId       = cellsPtr[iCell];
                  const size_type enrichInCell = localToCellPtr[iCell];

                  // localToCellLocalEIds and overlappingEnrichmentIdsInCells
                  // are two views of one relation; a drift between them would
                  // silently pair an enrichment with another one's origins.
                  DFTEFE_AssertWithMsg(
                    d_overlappingEnrichmentIdsInCells[cellId][enrichInCell] ==
                      localToGlobalEnrichmentIds[iLocalEId],
                    "localToCellLocalEIds does not index this enrichment "
                    "within the cell's overlap list.");

                  const std::vector<size_type> &originOffsets =
                    d_enrichmentIdsPartition
                      ->getExtendedAtomIdOffsetsForAllEnrichInCell(cellId);
                  const size_type nOrigins = originOffsets[enrichInCell + 1] -
                                             originOffsets[enrichInCell];

                  totalEnrichOriginPairsThisBlock += nOrigins;
                  totalQuadPtsThisBlock +=
                    nOrigins * quadRuleContainer.nCellQuadraturePoints(cellId);
                }
            }
          maxTotalQuadPtsForBlock =
            std::max(maxTotalQuadPtsForBlock, totalQuadPtsThisBlock);
          maxEnrichOriginPairsInBatch = std::max(maxEnrichOriginPairsInBatch, totalEnrichOriginPairsThisBlock);
        }

      // Output offset of each cell, precomputed. The transpose below needs it
      // once per overlap cell, so the running std::accumulate it used to do
      // would now be repeated more often than before.
      std::vector<size_type> quadxEnrichCumulative(numLocallyOwnedCells + 1, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
        quadxEnrichCumulative[iCell + 1] =
          quadxEnrichCumulative[iCell] + quadxEnrichPerCell[iCell];

      // A cell reached by several origins appears as an overlap cell once per
      // origin, so the transpose accumulates instead of assigning and the
      // output has to start from zero.
      if (storeValues)
        std::fill(basisEnrichQuadStorageStartPtr,
                  basisEnrichQuadStorageStartPtr +
                    quadxEnrichCumulative[numLocallyOwnedCells],
                  0.0);
      if (storeGradients)
        std::fill(basisGradientEnrichQuadStorageStartPtr,
                  basisGradientEnrichQuadStorageStartPtr +
                    quadxEnrichCumulative[numLocallyOwnedCells] * dim,
                  0.0);
      // profiler.registerEnd("setup");

      // profiler.registerStart("scratchAlloc");
      // --- allocate scratch buffers once at max size ---
      std::vector<std::shared_ptr<atoms::SphericalData>> sphericalDataVecBlock(
        maxEnrichOriginPairsInBatch);
      std::vector<size_type> quadInEnrichOriginPairBlock(maxEnrichOriginPairsInBatch, 0);
      std::vector<double>    originBlock(maxEnrichOriginPairsInBatch * dim, 0);

      // The distinct origins of one enrichment, and the (cell, position within
      // that cell) of every overlap cell in the order the quadrature points
      // are laid out -- which is the order the transposes walk them back in.
      std::vector<size_type> distinctOrigins(maxEnrichOriginPairsInBatch, 0);
      std::vector<size_type> overlapCellId(maxEnrichOriginPairsInBatch, 0);
      std::vector<size_type> overlapCellEnrichPos(maxEnrichOriginPairsInBatch, 0);

      utils::MemoryStorage<double, memorySpace> quadPtsBlockMemSpace(
        maxTotalQuadPtsForBlock * dim);
      utils::MemoryStorage<double, memorySpace> originMemspaceBlock(
        maxEnrichOriginPairsInBatch * dim);

      utils::MemoryStorage<double, memorySpace> valuesBlockMemSpace(0);
      std::vector<double>                       valuesBlockHost(0);
      utils::MemoryStorage<double, memorySpace> gradientsBlockMemSpace(0);
      std::vector<double>                       gradientsBlockHost(0);

      if (storeValues)
        {
          valuesBlockMemSpace.resize(maxTotalQuadPtsForBlock);
          valuesBlockHost.resize(maxTotalQuadPtsForBlock);
        }
      if (storeGradients)
        {
          gradientsBlockMemSpace.resize(maxTotalQuadPtsForBlock * dim);
          gradientsBlockHost.resize(maxTotalQuadPtsForBlock * dim);
        }
      // profiler.registerEnd("scratchAlloc");

      // block loop over local enrichment ids
      for (size_type enrichBlockStart = 0; enrichBlockStart < numLocalEnrichIds;
           enrichBlockStart += enrichBlock)
        {
          const size_type enrichBlockEnd =
            std::min(enrichBlockStart + enrichBlock, numLocalEnrichIds);

          // profiler.registerStart("fillQuadPtsAndOrigins");
          // Build this block's enrichment-origin pairs and lay out their
          // quadrature points.
          size_type numEnrichOriginPairsInBatch                  = 0;
          size_type numOverlapCellsInBatch                 = 0;
          size_type totalQuadPtsForBlock              = 0;
          size_type cumulativeQuadInCellPerBlockEnrich = 0;
          for (size_type iLocalEId = enrichBlockStart;
               iLocalEId < enrichBlockEnd;
               iLocalEId++)
            {
              basis::EnrichmentIdAttribute eIdAttr =
                d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                  localToGlobalEnrichmentIds[iLocalEId]);
              const size_type atomId  = eIdAttr.atomId;
              const size_type localId = eIdAttr.localIdInAtom;

              const size_type *cellsPtr =
                overlappingCellsWithLocalEnrichmentIds.data() +
                enrichCellCumOffset[iLocalEId];
              const size_type *localToCellPtr =
                localToCellLocalEIds.data() + enrichCellCumOffset[iLocalEId];
              const size_type nCellsInEId = cellsInLocalEId[iLocalEId];

              // The origins that reach this enrichment anywhere, deduplicated.
              // Counts are tiny -- exactly one without periodicity -- so a
              // linear scan beats any ordered container.
              size_type numDistinctOrigins = 0;
              for (size_type iCell = 0; iCell < nCellsInEId; iCell++)
                {
                  const std::vector<size_type> &ids =
                    d_enrichmentIdsPartition->getExtendedAtomIdsForAllEnrichInCell(
                      cellsPtr[iCell]);
                  const std::vector<size_type> &offsets =
                    d_enrichmentIdsPartition
                      ->getExtendedAtomIdOffsetsForAllEnrichInCell(cellsPtr[iCell]);
                  for (size_type o = offsets[localToCellPtr[iCell]];
                       o < offsets[localToCellPtr[iCell] + 1];
                       o++)
                    {
                      bool alreadySeen = false;
                      for (size_type k = 0; k < numDistinctOrigins; k++)
                        if (distinctOrigins[k] == ids[o])
                          {
                            alreadySeen = true;
                            break;
                          }
                      if (!alreadySeen)
                        distinctOrigins[numDistinctOrigins++] = ids[o];
                    }
                }

              // One pair per distinct origin, holding every cell that origin
              // reaches, their quadrature points laid out back to back.
              for (size_type iDistinct = 0; iDistinct < numDistinctOrigins;
                   iDistinct++)
                {
                  const size_type    extendedAtomId = distinctOrigins[iDistinct];
                  const utils::Point origin =
                    d_enrichmentIdsPartition->getPositionOfExtendedAtomId(
                      extendedAtomId);
                  std::memcpy(originBlock.data() + numEnrichOriginPairsInBatch * dim,
                              origin.data(),
                              dim * sizeof(double));
                  sphericalDataVecBlock[numEnrichOriginPairsInBatch] =
                    d_atomSphericalDataContainer->getSphericalData(
                      d_atomSymbolVec[atomId], d_fieldName)[localId];

                  size_type quadPtsInEnrichOriginPair = 0;
                  for (size_type iCell = 0; iCell < nCellsInEId; iCell++)
                    {
                      const size_type cellId       = cellsPtr[iCell];
                      const size_type enrichInCell = localToCellPtr[iCell];
                      const std::vector<size_type> &ids =
                        d_enrichmentIdsPartition->getExtendedAtomIdsForAllEnrichInCell(
                          cellId);
                      const std::vector<size_type> &offsets =
                        d_enrichmentIdsPartition
                          ->getExtendedAtomIdOffsetsForAllEnrichInCell(cellId);

                      bool reachesThisCell = false;
                      for (size_type o = offsets[enrichInCell];
                           o < offsets[enrichInCell + 1];
                           o++)
                        if (ids[o] == extendedAtomId)
                          {
                            reachesThisCell = true;
                            break;
                          }
                      if (!reachesThisCell)
                        continue;

                      const size_type nCellQuadPoints =
                        quadRuleContainer.nCellQuadraturePoints(cellId);
                      overlapCellId[numOverlapCellsInBatch]       = cellId;
                      overlapCellEnrichPos[numOverlapCellsInBatch] = enrichInCell;
                      numOverlapCellsInBatch++;

                      linearAlgebra::blasLapack::
                        copyValueType1ArrToValueType2Arr(
                          nCellQuadPoints * dim,
                          quadRuleContainer
                              .template getRealPointsPtr<memorySpace>() +
                            quadRuleContainer.getCellQuadStartId(cellId) * dim,
                          quadPtsBlockMemSpace.data() +
                            cumulativeQuadInCellPerBlockEnrich,
                          linAlgOpContext);
                      cumulativeQuadInCellPerBlockEnrich +=
                        nCellQuadPoints * dim;
                      quadPtsInEnrichOriginPair += nCellQuadPoints;
                    }

                  quadInEnrichOriginPairBlock[numEnrichOriginPairsInBatch] = quadPtsInEnrichOriginPair;
                  totalQuadPtsForBlock += quadPtsInEnrichOriginPair;
                  numEnrichOriginPairsInBatch++;
                }
            }
          // profiler.registerEnd("fillQuadPtsAndOrigins");

          // profiler.registerStart("originH2D");
          // copy only the used portion of origins to device
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numEnrichOriginPairsInBatch * dim,
            originMemspaceBlock.data(),
            originBlock.data());
          // profiler.registerEnd("originH2D");

          if (storeValues)
            {
              // profiler.registerStart("getEnrichmentValues");
              EnrichmentDataEvalKernels<memorySpace>::getEnrichmentValues(
                numEnrichOriginPairsInBatch,
                quadInEnrichOriginPairBlock,
                sphericalDataVecBlock,
                quadPtsBlockMemSpace.data(),
                originMemspaceBlock.data(),
                valuesBlockMemSpace.data(),
                linAlgOpContext);
              // profiler.registerEnd("getEnrichmentValues");

              // profiler.registerStart("valuesD2H");
              // copy only the used portion back to host
              utils::MemoryTransfer<utils::MemorySpace::HOST,
                                    memorySpace>::copy(totalQuadPtsForBlock,
                                                       valuesBlockHost.data(),
                                                       valuesBlockMemSpace
                                                         .data());
              // profiler.registerEnd("valuesD2H");

              // profiler.registerStart("transposeValues");
              // transpose: enrichOriginPair->cell->quad (block-local) →
              // cell->quad->enrich (output), overlap cells in layout order.
              size_type cumuLativeQuadPerCellInLocalEnrich = 0;
              for (size_type iOverlapCell = 0; iOverlapCell < numOverlapCellsInBatch;
                   iOverlapCell++)
                {
                  const size_type cellId = overlapCellId[iOverlapCell];
                  const size_type numQuadInCell =
                    quadRuleContainer.nCellQuadraturePoints(cellId);
                  const size_type numEnrichInCell =
                    d_overlappingEnrichmentIdsInCells[cellId].size();
                  const size_type cumulativeQuadxEnrichInCell =
                    quadxEnrichCumulative[cellId];
                  const size_type enrichIndexInCell =
                    overlapCellEnrichPos[iOverlapCell];
                  for (size_type qPoint = 0; qPoint < numQuadInCell; qPoint++)
                    {
                      *(basisEnrichQuadStorageStartPtr +
                        cumulativeQuadxEnrichInCell +
                        qPoint * numEnrichInCell + enrichIndexInCell) +=
                        *(valuesBlockHost.data() +
                          cumuLativeQuadPerCellInLocalEnrich + qPoint);
                    }
                  cumuLativeQuadPerCellInLocalEnrich += numQuadInCell;
                }
              // profiler.registerEnd("transposeValues");
            }

          if (storeGradients)
            {
              // profiler.registerStart("getEnrichmentGradients");
              EnrichmentDataEvalKernels<memorySpace>::getEnrichmentGradients(
                numEnrichOriginPairsInBatch,
                quadInEnrichOriginPairBlock,
                sphericalDataVecBlock,
                quadPtsBlockMemSpace.data(),
                originMemspaceBlock.data(),
                gradientsBlockMemSpace.data(),
                linAlgOpContext);
              // profiler.registerEnd("getEnrichmentGradients");

              // profiler.registerStart("gradientsD2H");
              utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::
                copy(totalQuadPtsForBlock * dim,
                     gradientsBlockHost.data(),
                     gradientsBlockMemSpace.data());
              // profiler.registerEnd("gradientsD2H");

              // profiler.registerStart("transposeGradients");
              // transpose: enrichOriginPair->cell->quad (block-local) →
              // cell->quad->dim->enrich (output)
              size_type cumuLativeQuadPerCellInLocalEnrich = 0;
              for (size_type iOverlapCell = 0; iOverlapCell < numOverlapCellsInBatch;
                   iOverlapCell++)
                {
                  const size_type cellId = overlapCellId[iOverlapCell];
                  const size_type numQuadInCell =
                    quadRuleContainer.nCellQuadraturePoints(cellId);
                  const size_type numEnrichInCell =
                    d_overlappingEnrichmentIdsInCells[cellId].size();
                  const size_type cumulativeQuadxEnrichInCell =
                    quadxEnrichCumulative[cellId];
                  const size_type enrichIndexInCell =
                    overlapCellEnrichPos[iOverlapCell];
                  for (size_type qPoint = 0; qPoint < numQuadInCell; qPoint++)
                    {
                      for (size_type iDim = 0; iDim < dim; iDim++)
                        {
                          *(basisGradientEnrichQuadStorageStartPtr +
                            cumulativeQuadxEnrichInCell * dim +
                            qPoint * numEnrichInCell * dim +
                            iDim * numEnrichInCell + enrichIndexInCell) +=
                            *(gradientsBlockHost.data() +
                              (cumuLativeQuadPerCellInLocalEnrich + qPoint) *
                                dim +
                              iDim);
                        }
                    }
                  cumuLativeQuadPerCellInLocalEnrich += numQuadInCell;
                }
              // profiler.registerEnd("transposeGradients");
            }
        } // end enrichBlock loop

      // profiler.print();
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EnrichmentClassicalInterfaceSpherical<
      ValueTypeBasisData,
      memorySpace,
      dim>::getOverlappingEnrichmentInCellsAdditionalData()
    {
      if (d_sphericalDataNumericalFuncPtrVec != nullptr)
        {
          utils::MemoryManager<
            atoms::SphericalDataNumerical::Func<memorySpace>,
            memorySpace>::deallocate(d_sphericalDataNumericalFuncPtrVec);
          d_sphericalDataNumericalFuncPtrVec = nullptr;
        }
      const size_type numCells = d_overlappingEnrichmentIdsInCells.size();
      d_numEnrichInAllCells.resize(numCells);
      size_type totalEnrich = 0;
      for (size_type iCell = 0; iCell < numCells; iCell++)
        {
          d_numEnrichInAllCells[iCell] =
            d_overlappingEnrichmentIdsInCells[iCell].size();
          totalEnrich += d_numEnrichInAllCells[iCell];
        }

      // Each enrichment is a sum over its origins -- the master atom plus
      // every image whose cutoff reaches this cell -- so count them before
      // sizing the flat origin array. Without periodicity the count is one.
      std::vector<size_type> originOffsetHost(totalEnrich + 1, 0);
      size_type              enrichCount = 0;
      for (size_type iCell = 0; iCell < numCells; iCell++)
        for (size_type iEnrich = 0;
             iEnrich < d_overlappingEnrichmentIdsInCells[iCell].size();
             iEnrich++)
          {
            originOffsetHost[enrichCount + 1] =
              originOffsetHost[enrichCount] +
              d_enrichmentIdsPartition->getExtendedAtomIdsForCellEnrich(iCell, iEnrich)
                .size();
            enrichCount++;
          }
      const size_type totalOrigins = originOffsetHost[totalEnrich];

      std::vector<atoms::SphericalDataNumerical::Func<memorySpace>> funcVecHost(
        totalEnrich);
      std::vector<double> originHost(totalOrigins * dim);

      size_type enrichOffset = 0;
      for (size_type iCell = 0; iCell < numCells; iCell++)
        {
          const auto &enrichIds = d_overlappingEnrichmentIdsInCells[iCell];
          for (size_type iEnrich = 0; iEnrich < enrichIds.size(); iEnrich++)
            {
              const global_size_type globalEnrichId = enrichIds[iEnrich];
              EnrichmentIdAttribute  eIdAttr =
                d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                  globalEnrichId);
              const size_type atomId  = eIdAttr.atomId;
              const size_type localId = eIdAttr.localIdInAtom;

              auto spData = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName)[localId];
              auto *numericalData =
                dynamic_cast<atoms::SphericalDataNumerical *>(spData.get());
              utils::throwException(
                numericalData != nullptr,
                " For getEnrichmentDataInCellRangeAtQuadPts: enrichment data must be SphericalDataNumerical.");

              // One Func per (cell, enrichment): it depends on the species and
              // the quantum numbers, not on where the function is centred, so
              // the images reuse it and only add origins.
              funcVecHost[enrichOffset + iEnrich] =
                numericalData->getFunc<memorySpace>();

              // getPositionOfExtendedAtomId resolves the master/image encoding here,
              // on the host; the device only ever sees plain coordinates.
              size_type iOrigin = originOffsetHost[enrichOffset + iEnrich];
              for (auto extendedAtomId :
                   d_enrichmentIdsPartition->getExtendedAtomIdsForCellEnrich(iCell,
                                                                     iEnrich))
                {
                  const utils::Point origin =
                    d_enrichmentIdsPartition->getPositionOfExtendedAtomId(
                      extendedAtomId);
                  std::memcpy(originHost.data() + iOrigin * dim,
                              origin.data(),
                              dim * sizeof(double));
                  iOrigin++;
                }
            }
          enrichOffset += enrichIds.size();
        }

      utils::MemoryManager<
        atoms::SphericalDataNumerical::Func<memorySpace>,
        memorySpace>::allocate(totalEnrich,
                               &d_sphericalDataNumericalFuncPtrVec);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        totalEnrich, d_sphericalDataNumericalFuncPtrVec, funcVecHost.data());

      d_originMemSpace.resize(totalOrigins * dim);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        totalOrigins * dim, d_originMemSpace.data(), originHost.data());

      d_originOffsetPerCellEnrich.resize(totalEnrich + 1);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        totalEnrich + 1,
        d_originOffsetPerCellEnrich.data(),
        originOffsetHost.data());
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      getEnrichmentValuesInCellRangeAtQuadPts(
        const quadrature::QuadratureRuleContainer &  quadRuleContainer,
        double *                                     basisEnrichQuadStoragePtr,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const std::pair<size_type, size_type>        cellRange) const
    {
      // utils::Profiler<memorySpace> profiler(d_comm,
      //                                       "getEnrichmentValuesInCellRangeAtQuadPts");

      std::vector<size_type> numQuadInAllCells(quadRuleContainer.nCells());
      for (size_type iCell = 0; iCell < quadRuleContainer.nCells(); iCell++)
        {
          numQuadInAllCells[iCell] =
            quadRuleContainer.nCellQuadraturePoints(iCell);
        }

      // profiler.registerStart("timing");
      EnrichmentDataEvalKernels<memorySpace>::getEnrichmentValuesInCellRange(
        quadRuleContainer.template getRealPointsPtr<memorySpace>(),
        d_originMemSpace.data(),
        d_originOffsetPerCellEnrich.data(),
        cellRange,
        d_numEnrichInAllCells,
        numQuadInAllCells,
        d_sphericalDataNumericalFuncPtrVec,
        basisEnrichQuadStoragePtr,
        linAlgOpContext);

      // profiler.registerEnd("timing");
      // profiler.print();
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                          memorySpace,
                                          dim>::
      getEnrichmentGradientsInCellRangeAtQuadPts(
        const quadrature::QuadratureRuleContainer &quadRuleContainer,
        double *basisGradientEnrichQuadStoragePtr,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const std::pair<size_type, size_type>        cellRange) const
    {
      // utils::Profiler<memorySpace> profiler(d_comm,
      //                                       "getEnrichmentGradientsInCellRangeAtQuadPts");

      std::vector<size_type> numQuadInAllCells(quadRuleContainer.nCells());
      for (size_type iCell = 0; iCell < quadRuleContainer.nCells(); iCell++)
        {
          numQuadInAllCells[iCell] =
            quadRuleContainer.nCellQuadraturePoints(iCell);
        }

      // profiler.registerStart("timing");
      EnrichmentDataEvalKernels<memorySpace>::getEnrichmentGradientsInCellRange(
        quadRuleContainer.template getRealPointsPtr<memorySpace>(),
        d_originMemSpace.data(),
        d_originOffsetPerCellEnrich.data(),
        cellRange,
        d_numEnrichInAllCells,
        numQuadInAllCells,
        d_sphericalDataNumericalFuncPtrVec,
        basisGradientEnrichQuadStoragePtr,
        linAlgOpContext);

      // profiler.registerEnd("timing");
      // profiler.print();
    }

  } // namespace basis
} // namespace dftefe
