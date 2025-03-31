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
 #include <linearAlgebra/BlasLapack.h>
 #include <linearAlgebra/Vector.h>
 #include <linearAlgebra/BlasLapackTypedef.h>
 #include <linearAlgebra/LinAlgOpContext.h>
 #include <basis/FECellWiseDataOperations.h>
 
 namespace dftefe
 {
   namespace basis
   {
     namespace AtomCenterNonLocalOpContextFEInternal
     {
        template <typename ValueTypeOperator,
                  typename ValueTypeOperand,
                  utils::MemorySpace memorySpace>
        void
        computeCXCellWiseLocal(
          const utils::MemoryStorage<ValueTypeOperator, memorySpace>
            &                     cellWiseC,
          const ValueTypeOperand *x,
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                  ValueTypeOperand> *y,
          bool                                         isCConjTransX,                                  
          const size_type                              numVecs,
          const size_type                              numLocallyOwnedCells,
          const std::vector<size_type> &               numCellDofs,
          const std::vector<size_type> &               numCellProjectors,
          const size_type *                            cellLocalIdsStartPtrX,
          const size_type *                            cellLocalIdsStartPtrY,
          const size_type                              cellBlockSize,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          if(memorySpace == utils::MemorySpace::HOST) 
            cellBlockSize = 1;
          
          linearAlgebra::blasLapack::Layout layout =
            linearAlgebra::blasLapack::Layout::ColMajor;

          size_type cellWiseCStartOffset = 0;
          size_type cellLocalIdsOffset = 0;

          std::vector<size_type> &numCellContractingIds = isCConjTransX ? numCellProjectors : numCellDofs;

          size_type maxContractingIdsInCell = *std::max_element(numCellContractingIds.begin(), numCellContractingIds.end());

          utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
            cellBlockSize * numVecs * maxContractingIdsInCell,
            utils::Types<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                      ValueTypeOperand>>::zero);

          utils::MemoryStorage<linearAlgebra::blasLapack::
                                  scalar_type<ValueTypeOperator, ValueTypeOperand>,
                                memorySpace>
            yCellValues(
              cellBlockSize * numVecs * maxContractingIdsInCell,
              utils::Types<
                linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand>>::zero);

          // can be make agnostic if the gemmStridedVarBatched handles zero rows and columns
          if(memorySpace == utils::MemorySpace::HOST) 
          {
            for (size_type iCell = 0; iCell < numLocallyOwnedCells;
                iCell += 1)
            {
              if(numCellProjectors[iCell] > 0)
              {
                std::vector<size_type> cellsNumContractIds(1, numCellContractingIds[iCell]);

                // copy x to cell-wise data
                basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
                  copyFieldToCellWiseData(x,
                                          numVecs,
                                          cellLocalIdsStartPtrX +
                                            cellLocalIdsOffset,
                                          cellsNumContractIds,
                                          xCellValues);

                std::vector<linearAlgebra::blasLapack::Op> transA = isCConjTransX ? 
                  linearAlgebra::blasLapack::Op::Trans : 
                  linearAlgebra::blasLapack::Op::NoTrans;
                std::vector<linearAlgebra::blasLapack::Op> transB = linearAlgebra::blasLapack::Op::NoTrans;
      
                size_type m   = isCConjTransX ? numVecs : numCellProjectors[iCell];
                size_type n   = isCConjTransX ? numCellDofs[iCell] : numVecs; 
                size_type k   = isCConjTransX ? numCellProjectors[iCell] : numCellDofs[iCell];
                size_type ldaSize = m;
                size_type ldbSize = isCConjTransX ? n : k;
                size_type ldcSize = m;

                const ValueTypeOperator *A = isCConjTransX ? xCellValues.data()
                  : cellWiseC.data() + cellWiseCStartOffset;

                const ValueTypeOperator *B = isCConjTransX ?
                  cellWiseC.data() + cellWiseCStartOffset : xCellValues.data();

                linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand> *C =
                  yCellValues.begin();

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperand,
                                                utils::MemorySpace::HOST>(
                  layout,
                  transA,
                  transB,
                  m,
                  n,
                  k,
                  (ValueType)1.0,
                  A,
                  ldaSize,
                  B,
                  ldbSize,
                  (ValueType)0.0,
                  C,
                  ldcSize,
                  linAlgOpContext);                

                basis::FECellWiseDataOperations<
                  linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                          ValueTypeOperand>,
                  memorySpace>::addCellWiseDataToFieldData(yCellValues,
                                                            numVecs,
                                                            cellLocalIdsStartPtrY +
                                                              cellLocalIdsOffset,
                                                            cellsNumContractIds,
                                                            y);

                cellWiseCStartOffset +=
                  numCellProjectors[iCell] * numCellDofs[iCell];
                cellLocalIdsOffset += cellsNumContractIds[0];
              }
            }
          }
          else if(memorySpace == utils::MemorySpace::DEVICE) 
          {
            utils::throwException(
              false, "computeCXCellWiseLocal not implemented in DEVICE.");
          }
        }
     }
 
     template <typename ValueTypeOperator,
               typename ValueTypeOperand,
               utils::MemorySpace memorySpace,
               size_type          dim>
     AtomCenterNonLocalOpContextFE<ValueTypeOperator,
                               ValueTypeOperand,
                               memorySpace,
                               dim>::
       AtomCenterNonLocalOpContextFE(
         const FEBasisManager<ValueTypeOperand,
                              ValueTypeOperator,
                              memorySpace,
                              dim> &feBasisManager,
         const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
           &feBasisDataStorage,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const double                     atomPartitionTolerance,
        const std::vector<std::string> & atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinatesVec,
        const std::string                fieldNameProjector,
        const size_type maxCellBlock,
        const size_type maxFieldBlock,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm)
       : d_feBasisManager(&feBasisManager)
       , d_maxCellBlock(maxCellBlock)
       , d_maxFieldBlock(maxFieldBlock)
     {
      // Construct C_cell = \integral_\omega \beta_lp * Y_lm 
      // * N_j

      // Create enrichmentIdsPartition for non-local projectors

      int rank;
      utils::mpi::MPICommRank(comm, &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      int numProcs;
      utils::mpi::MPICommSize(comm, &numProcs);

      if (dim != 3)
        utils::throwException(
          false, "Dimension should be 3 for Spherical Projector Dofs.");

      feBasisDofHandler = std::dynamic_pointer_cast<
        const FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>(
          feBasisDataStorage->getBasisDofHandler());
      utils::throwException(
        feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in AtomCenterNonLocalOpContextFE");

      triangulation = feBasisDofHandler->getTriangulation();

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto cell = triangulation->beginLocal();
      auto endc = triangulation->endLocal();

      size_type cellIndex                        = 0;
      int       locallyOwnedCellsInTriangulation = 0;

      for (; cell != endc; cell++)
        {
          (*cell)->getVertices(cellVertices);
          cellVerticesVector.push_back(cellVertices);
          locallyOwnedCellsInTriangulation++;
        }

      utils::throwException(
        feBasisDofHandler->nLocallyOwnedCells() ==
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
      d_projectorIdsPartition = std::make_shared<EnrichmentIdsPartition<dim>>(
        atomSphericalDataContainer,
        d_atomIdsPartition,
        atomSymbolVec,
        atomCoordinatesVec,
        fieldNameProjector,
        minbound,
        maxbound,
        0,
        triangulation->getDomainVectors(),
        triangulation->getPeriodicFlags(),
        cellVerticesVector,
        comm);

      d_overlappingProjectorIdsInCells =
        d_projectorIdsPartition->overlappingEnrichmentIdsInCells();

      // C_cell is N_projectors_val(r, theta, phi) * 
      // N_dofs which is got by contraction over a cell of quadpoints 

      size_type cellWiseCSize = 0;
      auto locallyOwnedCellIter = feBasisDofHandler->beginLocallyOwnedCells();
      int cellIndex = 0;

      for (; locallyOwnedCellIter != feBasisDofHandler->endLocallyOwnedCells();
           ++locallyOwnedCellIter)
        {
          d_numProjsInCells[cellIndex] = d_overlappingProjectorIdsInCells[cellIndex].size();
          size_type nDofsInCell = feBasisDofHandler->nCellDofs(cellIndex);
          cellWiseCSize += d_numProjsInCells[cellIndex] * nDofsInCell;
          cellIndex++;
        }
      d_cellWiseC.resize(cellWiseCSize);

      d_maxProjInCell = *std::max_element(d_numProjsInCells.begin(), d_numProjsInCells.end());

      cellIndex = 0;
      size_type cumulativeDofxProj = 0;
      locallyOwnedCellIter = feBasisDofHandler->beginLocallyOwnedCells();
      for (; locallyOwnedCellIter != feBasisDofHandler->endLocallyOwnedCells();
            ++locallyOwnedCellIter)
        {
          size_type nQuadsInCell = quadratureRuleContainer->nCellQuadraturePoints(cellIndex);
          size_type numProjsInCell = d_numProjsInCells[cellIndex];
          size_type numDofsInCell = feBasisDofHandler->nCellDofs(cellIndex);
          
          utils::MemoryStorage<ValueTypeOperator, utils::memorySpace::HOST> 
            projectorQuadStorageJxW(numProjsInCell * nQuadsInCell);
          std::vector<double> &cellJxW = quadratureRuleContainer->getCellJxW(cellIndex);

          const std::vector<double> &projValAtQuadPts = getProjectorValues(cellIndex);

          for (unsigned int iProj = 0 ; iProj < numProjsInCell; iProj++)
            {
              for (unsigned int qPoint = 0; qPoint < nQuadsInCell;
                    qPoint++)
                {
                  *(projectorQuadStorageJxW.data() + qPoint * numProjsInCell; + iProj) +=
                    *(projValAtQuadPts.data() + nQuadsInCell * iProj + qPoint) * cellJxW[qPoint];
                }
            }

          if(numProjsInCell > 0)
          {
            utils::MemoryStorage<ValueTypeOperator, utils::memorySpace::HOST> 
              basisData(numDofsInCell * nQuadsInCell);

            feBasisDataStorage->getBasisDataInCellRange(
                std::make_pair<size_type, size_type>{cellIndex} , basisData);
    
            linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                            ValueTypeOperator,
                                            utils::MemorySpace::HOST>(
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::Op::NoTrans,
              linearAlgebra::blasLapack::Op::Trans,
              numProjsInCell,
              numDofsInCell,
              nQuadsInCell,
              (ValueTypeOperator)1.0,
              projectorQuadStorageJxW.data(),
              numProjsInCell,
              basisData.data(),
              numDofsInCell,
              (ValueTypeOperator)0.0,
              d_cellWiseC.data() + cumulativeDofxProj,
              numProjsInCell,
              linAlgOpContext);
          }
          cumulativeDofxProj += numDofsInCell * nQuadsInCell;
          cellIndex++;
        }
      
      // Get the coupling matrix V = D_ij 

      // Create mpiPatternP2P for locOwned and ghost Projectors 

      // create the d_locallyOwnedCellLocalProjectorIds

      size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
      size_type cumulativeProjectors       = 0;
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          const size_type numCellProjectors = d_numProjInCells[iCell];
          for (size_type iProj = 0; iProj < numCellProjectors; ++iProj)
            {
              const global_size_type globalProjId = d_overlappingProjectorIdsInCells[iProj];
              d_locallyOwnedCellLocalProjectorIds[cumulativeProjectors + iProj] =
                d_mpiPatternP2P->globalToLocal(globalProjId);
            }
          cumulativeProjectors += numCellProjectors;
        }

      // Initilize the d_CX
      d_CX = linearAlgebra::MultiVector<ValueType, memorySpace>(
                                              d_mpiPatternP2P,
                                              d_linAlgOpContext,
                                              d_numComponents);

     }
 
     template <typename ValueTypeOperator,
               typename ValueTypeOperand,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     AtomCenterNonLocalOpContextFE<
       ValueTypeOperator,
       ValueTypeOperand,
       memorySpace,
       dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                   linearAlgebra::MultiVector<
                     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                            ValueTypeOperand>,
                     memorySpace> &Y,
                   bool            updateGhostX,
                   bool            updateGhostY) const
     {
        const size_type numLocallyOwnedCells =
          d_feBasisManager->nLocallyOwnedCells();
        std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);
          }

        const size_type numVecs = X.getNumberComponents();

        // get handle to constraints
        const basis::ConstraintsLocal<ValueType, memorySpace> &constraintsX =
          d_feBasisManager->getConstraints();

        const basis::ConstraintsLocal<ValueType, memorySpace> &constraintsY =
          d_feBasisManager->getConstraints();

        // Do CVC^T X 
        if (updateGhostX)
          X.updateGhostValues();
        constraintsX.distributeParentToChild(X, numVecs);

        Y.setValue(0.0);

        const size_type cellBlockSize =
          (d_maxCellBlock * d_maxWaveFnBatch) / numVecs;

        // Do d_CX = d_cellWiseC * X cellwise (field to cellwise data)

        auto itCellLocalIdsBeginX =
          d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

        auto itCellLocalIdsBeginY =
          d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

        AtomCenterNonLocalOpContextFEInternal::computeCXCellWiseLocal(
          d_cellWiseC.data(),
          X.begin(),
          d_CX.begin(),
          false,
          numVecs,
          numLocallyOwnedCells,
          numCellDofs,
          itCellLocalIdsBeginX,
          itCellLocalIdsBeginY,
          cellBlockSize,
          *(X.getLinAlgOpContext()));

        // acumulate add, update ghost of d_CX

        d_CX.accumulateAddLocallyOwned();
        d_CX.updateGhostValues();

        // Do d_CX = d_V * d_CX
        size_type stride = 0;
        size_type m = 1, n = numVec, k = d_CX.localSize();

        linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                          ValueTypeOperator,
                                                          memorySpace>(
          1,
          linearAlgebra::blasLapack::Layout::RowMajor,
          linearAlgebra::blasLapack::ScalarOp::Identity,
          linearAlgebra::blasLapack::ScalarOp::Identity,
          &stride,
          &stride,
          &stride,
          &m,
          &n,
          &k,
          d_V.data(),
          d_CX.data(),
          d_CX.data(),
          linAlgOpContext);

        // Do Y = d_cellwiseC * d_CX cellwise (field to cellwise data)

        itCellLocalIdsBeginX =
          d_locallyOwnedCellLocalProjectorIds.begin();

        itCellLocalIdsBeginY =
          d_locallyOwnedCellLocalProjectorIds.begin();

        AtomCenterNonLocalOpContextFEInternal::computeCXCellWiseLocal(
          d_cellWiseC.data(),
          d_CX.begin(),
          Y.begin(),
          true,
          numVecs,
          numLocallyOwnedCells,
          numCellDofs,
          itCellLocalIdsBeginX,
          itCellLocalIdsBeginY,
          cellBlockSize,
          *(X.getLinAlgOpContext()));

        constraintsY.distributeChildToParent(Y, numVecs);
        Y.accumulateAddLocallyOwned();
        if (updateGhostY)
          Y.updateGhostValues();
     }
 
   } // namespace basis
 } // end of namespace dftefe
 