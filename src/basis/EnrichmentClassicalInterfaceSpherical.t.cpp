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
#include <basis/Defaults.h>
#include <utils/PointImpl.h>
#include <basis/L2ProjectionLinearSolverFunction.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/CFEOverlapInverseOpContextGLL.h>
#include <utils/ScalarSpatialFunction.h>
#include <algorithm>
#include <set>
#include <string>

namespace dftefe
{
  namespace basis
  {
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
        const utils::mpi::MPIComm &comm)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_enrichmentIdsPartition(nullptr)
      , d_atomIdsPartition(nullptr)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_overlappingEnrichmentIdsInCells(0)
      , d_linAlgOpContext(linAlgOpContext)
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

      for (; cell != endc; cell++)
        {
          (*cell)->getVertices(cellVertices);
          cellVerticesVector.push_back(cellVertices);
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
      d_enrichmentIdsPartition =
        std::make_shared<const EnrichmentIdsPartition<dim>>(
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

      global_size_type maxEnrich = 0;
      global_size_type minEnrich = 0;
      global_size_type avgEnrich = 0;
      size_type        cellIndex = 0;
      cell                       = d_triangulation->beginLocal();
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

      utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        &maxEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        &minEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMin,
        comm);

      utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        &avgEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPISum,
        comm);

      avgEnrich /= numProcs;

      rootCout << "Maximum " << fieldName
               << " Enrichment Ids In a Cell in Processor: " << maxEnrich
               << "\n";
      rootCout << "Minimum " << fieldName
               << " Enrichment Ids In a Cell in Processor: " << minEnrich
               << "\n";
      rootCout << "Average " << fieldName
               << " Enrichment Ids In a Cell In Processor: " << avgEnrich
               << "\n";

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

      std::shared_ptr<
        linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>
        basisInterfaceCoeff = std::make_shared<
          linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>(
          d_cfeBasisManager->getMPIPatternP2P(),
          linAlgOpContext,
          nTotalEnrichmentIds,
          ValueTypeBasisData());

      quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace>
        quadValuesEnrichmentFunction(
          cfeBasisDataStorageRhs->getQuadratureRuleContainer(),
          nTotalEnrichmentIds);

      const size_type numLocallyOwnedCells =
        d_cfeBasisDofHandler->nLocallyOwnedCells();
      std::vector<size_type> nQuadPointsInCell(0);
      nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
      cellIndex = 0;
      auto locallyOwnedCellIter =
        d_cfeBasisDofHandler->beginLocallyOwnedCells();
      for (;
           locallyOwnedCellIter != d_cfeBasisDofHandler->endLocallyOwnedCells();
           ++locallyOwnedCellIter)
        {
          size_type nQuadPointInCell =
            cfeBasisDataStorageRhs->getQuadratureRuleContainer()
              ->nCellQuadraturePoints(cellIndex);
          std::vector<utils::Point> quadRealPointsVec =
            cfeBasisDataStorageRhs->getQuadratureRuleContainer()
              ->getCellRealPoints(cellIndex);

          // std::vector<ValueTypeBasisData> enrichmentsAtQuadValues(
          //   nTotalEnrichmentIds * nQuadPointInCell, (ValueTypeBasisData)0);
          // for (auto enrichmentId :
          // d_overlappingEnrichmentIdsInCells[cellIndex])
          //   {
          //     basis::EnrichmentIdAttribute eIdAttr =
          //       d_enrichmentIdsPartition->getEnrichmentIdAttribute(
          //         enrichmentId);
          //     utils::Point origin(d_atomCoordinatesVec[eIdAttr.atomId]);
          //     auto         sphericalData =
          //       d_atomSphericalDataContainer->getSphericalData(
          //         d_atomSymbolVec[eIdAttr.atomId],
          //         d_fieldName)[eIdAttr.localIdInAtom];

          //     for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
          //     qPoint++)
          //       {
          //         enrichmentsAtQuadValues[qPoint * nTotalEnrichmentIds +
          //                                 enrichmentId] =
          //           sphericalData->getValue(quadRealPointsVec[qPoint],
          //           origin);
          //       }
          //     quadValuesEnrichmentFunction
          //       .template setCellValues<utils::MemorySpace::HOST>(
          //         cellIndex, enrichmentsAtQuadValues.data());
          //   }

          std::vector<ValueTypeBasisData> enrichmentsAtQuadValues(
            nTotalEnrichmentIds * nQuadPointInCell, (ValueTypeBasisData)0);
          std::vector<double> val =
            getEnrichmentValue(cellIndex, quadRealPointsVec);
          int enrichIdInCell = 0;
          for (auto enrichmentId : d_overlappingEnrichmentIdsInCells[cellIndex])
            {
              for (unsigned int qPoint = 0; qPoint < nQuadPointInCell; qPoint++)
                {
                  enrichmentsAtQuadValues[qPoint * nTotalEnrichmentIds +
                                          enrichmentId] =
                    *(val.data() + nQuadPointInCell * enrichIdInCell + qPoint);
                }
              quadValuesEnrichmentFunction
                .template setCellValues<utils::MemorySpace::HOST>(
                  cellIndex, enrichmentsAtQuadValues.data());

              enrichIdInCell += 1;
            }
          cellIndex = cellIndex + 1;
        }

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
          nTotalEnrichmentIds,
          linAlgOpContext);

      std::shared_ptr<linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                          ValueTypeBasisData,
                                                          memorySpace>>
        linearSolverFunction =
          std::make_shared<L2ProjectionLinearSolverFunction<ValueTypeBasisData,
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
            nTotalEnrichmentIds);

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

      CGSolve->solve(*linearSolverFunction);
      linearSolverFunction->getSolution(*basisInterfaceCoeff);

      /**
      // Can also do via the M^(-1) route withot solving CG.

      linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace> d(
            d_cfeBasisManager->getMPIPatternP2P(),
            linAlgOpContext,
            nTotalEnrichmentIds);
      d.setValue(0.0);

      FEBasisOperations<ValueTypeBasisData, ValueTypeBasisData, memorySpace,
      dim> cfeBasisOperations(cfeBasisDataStorageRhs,
      L2ProjectionDefaults::CELL_BATCH_SIZE, nTotalEnrichmentIds; rootCout <<
      "Begin creating integrateWithBasisValues\n";
      // Integrate this with different quarature rule. (i.e. adaptive for the
      // enrichment functions) , inp will be in adaptive grid
      cfeBasisOperations.integrateWithBasisValues(quadValuesEnrichmentFunction,
                                                 *d_cfeBasisManager,
                                                 d);
      rootCout << "End creating integrateWithBasisValues\n";
      utils::throwException(
        cfeBasisDataStorageOverlapMatrix->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL
      if Mc = d" "is not solved via a poisson solve. Contact developers if extra
      options are needed.");

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

      MInvContext->apply(d,*basisInterfaceCoeff);
      **/
      // populate an unordered_map<id, <vec1, vec2>>  i.e. map from enrichedId
      // -> pair(localId, coeff)

      std::vector<ValueTypeBasisData> basisInterfaceCoeffSTL(
        nTotalEnrichmentIds * d_cfeBasisManager->nLocal(),
        ValueTypeBasisData());

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        nTotalEnrichmentIds * d_cfeBasisManager->nLocal(),
        basisInterfaceCoeffSTL.data(),
        basisInterfaceCoeff->data());

      d_enrichmentIdToClassicalLocalIdMap.clear();
      d_enrichmentIdToInterfaceCoeffMap.clear();

      std::unordered_map<global_size_type, std::set<size_type>>
        enrichmentIdToClassicalLocalIdMapSet;
      enrichmentIdToClassicalLocalIdMapSet.clear();

      for (size_type i = 0; i < d_cfeBasisManager->nLocal(); i++)
        {
          for (global_size_type j = 0; j < nTotalEnrichmentIds; j++)
            {
              if (std::abs(*(basisInterfaceCoeffSTL.data() +
                             i * nTotalEnrichmentIds + j)) > 1e-12)
                {
                  enrichmentIdToClassicalLocalIdMapSet[j].insert(i);
                  d_enrichmentIdToInterfaceCoeffMap[j].push_back(
                    *(basisInterfaceCoeffSTL.data() + i * nTotalEnrichmentIds +
                      j));
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

      for (; cell != endc; cell++)
        {
          (*cell)->getVertices(cellVertices);
          cellVerticesVector.push_back(cellVertices);
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
      d_enrichmentIdsPartition =
        std::make_shared<const EnrichmentIdsPartition<dim>>(
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
      size_type        cellIndex = 0;
      cell                       = d_triangulation->beginLocal();
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

      utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        &maxEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        comm);

      utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        &minEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPIMin,
        comm);

      utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        &avgEnrich,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        utils::mpi::MPISum,
        comm);

      avgEnrich /= numProcs;

      rootCout << "Maximum " << fieldName
               << " Enrichment Ids In a Processor: " << maxEnrich << "\n";
      rootCout << "Minimum " << fieldName
               << " Enrichment Ids In a Cell in Processor: " << minEnrich
               << "\n";
      rootCout << "Average " << fieldName
               << " Enrichment Ids In a Processor: " << avgEnrich << "\n";

      rootCout
        << "Completed creating Pristine EnrichmentClassicalInterfaceSpherical for "
        << d_enrichmentIdsPartition->nTotalEnrichmentIds() << " " << fieldName
        << " enrichments." << std::endl;
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
      std::vector<double> retValue(dim * numPoints * numEnrichIdsInCell, 0),
        rVec(numPoints, 0), thetaVec(numPoints, 0), phiVec(numPoints, 0);
      std::vector<dftefe::utils::Point> x(numPoints, utils::Point(dim));
      DFTEFE_AssertWithMsg(
        !enrichIdVec.empty(),
        "The requested cell does not have any enrichment ids.");
      unsigned int numEnrichedIdsSkipped = 0;
      unsigned int l                     = 0;

      for (int iEnrich = 0; iEnrich < numEnrichIdsInCell;
           iEnrich += numEnrichedIdsSkipped)
        {
          basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(
              enrichIdVec[iEnrich]);

          size_type atomId  = eIdAttr.atomId;
          size_type localId = eIdAttr.localIdInAtom;

          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::transform(points.begin(),
                         points.end(),
                         x.begin(),
                         [origin](utils::Point p) { return p - origin; });

          atoms::convertCartesianToSpherical(
            x,
            rVec,
            thetaVec,
            phiVec,
            atoms::SphericalDataDefaults::POL_ANG_TOL);

          auto sphericalDataVec =
            d_atomSphericalDataContainer->getSphericalData(
              d_atomSymbolVec[atomId], d_fieldName);

          auto quantumNoVec =
            d_atomSphericalDataContainer->getQNumbers(d_atomSymbolVec[atomId],
                                                      d_fieldName);

          l = quantumNoVec[localId][1];

          auto radialValue = sphericalDataVec[localId]->getRadialValue(rVec);

          for (int mCount = 0; mCount < 2 * l + 1; mCount++)
            {
              auto angularValue = (sphericalDataVec[localId + mCount])
                                    ->getAngularValue(rVec, thetaVec, phiVec);

              linearAlgebra::blasLapack::hadamardProduct<
                ValueTypeBasisData,
                ValueTypeBasisData,
                utils::MemorySpace::HOST>(numPoints,
                                          radialValue.data(),
                                          angularValue.data(),
                                          retValue.data() +
                                            (iEnrich + mCount) * numPoints,
                                          *getLinAlgOpContext());
            }
          numEnrichedIdsSkipped = (2 * l + 1);
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
      unsigned int numEnrichedIdsSkipped = 0;
      unsigned int l                     = 0;

      for (int iEnrich = 0; iEnrich < numEnrichIdsInCell;
           iEnrich += numEnrichedIdsSkipped)
        {
          basis::EnrichmentIdAttribute eIdAttr =
            d_enrichmentIdsPartition->getEnrichmentIdAttribute(
              enrichIdVec[iEnrich]);

          size_type atomId  = eIdAttr.atomId;
          size_type localId = eIdAttr.localIdInAtom;

          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::transform(points.begin(),
                         points.end(),
                         x.begin(),
                         [origin](utils::Point p) { return p - origin; });

          atoms::convertCartesianToSpherical(
            x,
            rVec,
            thetaVec,
            phiVec,
            atoms::SphericalDataDefaults::POL_ANG_TOL);

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

          for (int mCount = 0; mCount < 2 * l + 1; mCount++)
            {
              auto angularValue = (sphericalDataVec[localId + mCount])
                                    ->getAngularValue(rVec, thetaVec, phiVec);
              auto angularDerivative =
                (sphericalDataVec[localId + mCount])
                  ->getAngularDerivative(rVec, thetaVec, phiVec);

              for (int i = 0; i < numPoints; i++)
                {
                  double dValueDR = radialDerivative[i] * angularValue[i];
                  double dValueDThetaByr = 0.;
                  dValueDThetaByr = radialValue[i] * angularDerivative[0][i];
                  double dValueDPhiByrsinTheta = 0.;
                  dValueDPhiByrsinTheta =
                    radialValue[i] * angularDerivative[1][i];
                  double theta = thetaVec[i], phi = phiVec[i];

                  retValue[(iEnrich + mCount) * numPoints * dim + i * dim + 0] =
                    dValueDR * (sin(theta) * cos(phi)) +
                    dValueDThetaByr * (cos(theta) * cos(phi)) -
                    sin(phi) * dValueDPhiByrsinTheta;
                  retValue[(iEnrich + mCount) * numPoints * dim + i * dim + 1] =
                    dValueDR * (sin(theta) * sin(phi)) +
                    dValueDThetaByr * (cos(theta) * sin(phi)) +
                    cos(phi) * dValueDPhiByrsinTheta;
                  retValue[(iEnrich + mCount) * numPoints * dim + i * dim + 2] =
                    dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));
                }
            }
          numEnrichedIdsSkipped = (2 * l + 1);
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

  } // namespace basis
} // namespace dftefe
