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
            &                     hamiltonianInAllCells,
          const ValueTypeOperand *x,
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                  ValueTypeOperand> *y,
          const size_type                                           numVecs,
          const size_type                              numLocallyOwnedCells,
          const std::vector<size_type> &               numCellDofs,
          const size_type *                            cellLocalIdsStartPtrX,
          const size_type *                            cellLocalIdsStartPtrY,
          const size_type                              cellBlockSize,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          linearAlgebra::blasLapack::Layout layout =
            linearAlgebra::blasLapack::Layout::ColMajor;

          size_type BStartOffset       = 0;
          size_type cellLocalIdsOffset = 0;

          size_type maxDofInCell =
            *std::max_element(numCellDofs.begin(), numCellDofs.end());

          utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
            cellBlockSize * numVecs * maxDofInCell,
            utils::Types<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                      ValueTypeOperand>>::zero);

          utils::MemoryStorage<linearAlgebra::blasLapack::
                                  scalar_type<ValueTypeOperator, ValueTypeOperand>,
                                memorySpace>
            yCellValues(
              cellBlockSize * numVecs * maxDofInCell,
              utils::Types<
                linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand>>::zero);

          for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
                cellStartId += cellBlockSize)
            {
              const size_type cellEndId =
                std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
              const size_type        numCellsInBlock = cellEndId - cellStartId;
              std::vector<size_type> cellsInBlockNumDoFsSTL(numCellsInBlock, 0);
              std::copy(numCellDofs.begin() + cellStartId,
                        numCellDofs.begin() + cellEndId,
                        cellsInBlockNumDoFsSTL.begin());

              const size_type cellsInBlockNumCumulativeDoFs =
                std::accumulate(cellsInBlockNumDoFsSTL.begin(),
                                cellsInBlockNumDoFsSTL.end(),
                                0);

              utils::MemoryStorage<size_type, memorySpace> cellsInBlockNumDoFs(
                numCellsInBlock);
              cellsInBlockNumDoFs.copyFrom(cellsInBlockNumDoFsSTL);

              // copy x to cell-wise data
              basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
                copyFieldToCellWiseData(x,
                                        numVecs,
                                        cellLocalIdsStartPtrX +
                                          cellLocalIdsOffset,
                                        cellsInBlockNumDoFs,
                                        xCellValues);

              std::vector<linearAlgebra::blasLapack::Op> transA(
                numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
              std::vector<linearAlgebra::blasLapack::Op> transB(
                numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);

              utils::MemoryStorage<size_type, memorySpace> mSizes(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> nSizes(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> kSizes(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> ldaSizes(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> ldbSizes(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> ldcSizes(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> strideA(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> strideB(
                numCellsInBlock);
              utils::MemoryStorage<size_type, memorySpace> strideC(
                numCellsInBlock);

              KohnShamOperatorContextFEInternal::storeSizes(
                mSizes,
                nSizes,
                kSizes,
                ldaSizes,
                ldbSizes,
                ldcSizes,
                strideA,
                strideB,
                strideC,
                cellsInBlockNumDoFsSTL,
                numVecs);

              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                      ValueTypeOperand>
                alpha = 1.0;
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                      ValueTypeOperand>
                beta = 0.0;

              const ValueTypeOperator *B =
                hamiltonianInAllCells.data() + BStartOffset;
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                      ValueTypeOperand> *C =
                yCellValues.begin();
              linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperator,
                                                                ValueTypeOperand,
                                                                memorySpace>(
                layout,
                numCellsInBlock,
                transA.data(),
                transB.data(),
                strideA.data(),
                strideB.data(),
                strideC.data(),
                mSizes.data(),
                nSizes.data(),
                kSizes.data(),
                alpha,
                xCellValues.data(),
                ldaSizes.data(),
                B,
                ldbSizes.data(),
                beta,
                C,
                ldcSizes.data(),
                linAlgOpContext);

              basis::FECellWiseDataOperations<
                linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand>,
                memorySpace>::addCellWiseDataToFieldData(yCellValues,
                                                          numVecs,
                                                          cellLocalIdsStartPtrY +
                                                            cellLocalIdsOffset,
                                                          cellsInBlockNumDoFs,
                                                          y);

              for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                {
                  BStartOffset +=
                    cellsInBlockNumDoFsSTL[iCell] * cellsInBlockNumDoFsSTL[iCell];
                  cellLocalIdsOffset += cellsInBlockNumDoFsSTL[iCell];
                }
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
        // Do CVC^T X 
        if (updateGhostX)
          X.updateGhostValues();

        // Do d_CX = d_cellWiseC * X cellwise (field to cellwise data)

        // acumulate add, update ghost of d_CX

        // Do d_CX = d_V * d_CX

        // Do Y = d_cellwiseC * d_CX cellwise (field to cellwise data)

        if (updateGhostY)
          Y.updateGhostValues();
     }
 
   } // namespace basis
 } // end of namespace dftefe
 