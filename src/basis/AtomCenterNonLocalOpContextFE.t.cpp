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
#include <utils/ConditionalOStream.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <basis/FECellWiseDataOperations.h>
#include <utils/StringOperations.h>

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
      cellWiseGEMM(
        std::pair<size_type, size_type>                             cellRange,
        const utils::MemoryStorage<ValueTypeOperator, memorySpace> &cellWiseC,
        bool                          isCConjTransX,
        const size_type               numVecs,
        const std::vector<size_type> &numCellXLocalIds,
        const std::vector<size_type> &numCellYLocalIds,
        const ValueTypeOperand *      xCellValues,
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand> *yCellValues,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        using ValueType =
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>;

        linearAlgebra::blasLapack::Layout layout =
          linearAlgebra::blasLapack::Layout::ColMajor;

        size_type cellStartId          = cellRange.first;
        size_type cellEndId            = cellRange.second;
        size_type cellWiseCStartOffset = 0;
        for (size_type iCell = 0; iCell < cellStartId; ++iCell)
          {
            cellWiseCStartOffset +=
              numCellXLocalIds[iCell] * numCellYLocalIds[iCell];
          }

        size_type numCellsInBlock = cellEndId - cellStartId;

        std::vector<linearAlgebra::blasLapack::Op> transA(
          numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
        std::vector<linearAlgebra::blasLapack::Op> transB(
          numCellsInBlock,
          isCConjTransX ? linearAlgebra::blasLapack::Op::ConjTrans :
                          linearAlgebra::blasLapack::Op::NoTrans);

        utils::MemoryStorage<size_type, memorySpace> mSizes(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> nSizes(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> kSizes(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> ldaSizes(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> ldbSizes(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> ldcSizes(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> strideA(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> strideB(numCellsInBlock);
        utils::MemoryStorage<size_type, memorySpace> strideC(numCellsInBlock);

        std::vector<size_type> mSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> nSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> kSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> ldaSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> ldbSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> ldcSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> strideASTL(numCellsInBlock, 0);
        std::vector<size_type> strideBSTL(numCellsInBlock, 0);
        std::vector<size_type> strideCSTL(numCellsInBlock, 0);

        for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
          {
            mSizesSTL[iCell] = numVecs;
            nSizesSTL[iCell] = numCellYLocalIds
              [iCell + cellStartId]; // !isCConjTransX ? numCellDofs[iCell]
                                     // : numCellProjectors[iCell] ;
            kSizesSTL[iCell] =
              numCellXLocalIds[iCell + cellStartId]; // !isCConjTransX ?
                                                     // numCellProjectors[iCell]
                                                     // : numCellDofs[iCell];
            ldaSizesSTL[iCell] = mSizesSTL[iCell];
            ldbSizesSTL[iCell] =
              isCConjTransX ? nSizesSTL[iCell] : kSizesSTL[iCell];
            ldcSizesSTL[iCell] = mSizesSTL[iCell];
            strideASTL[iCell]  = mSizesSTL[iCell] * kSizesSTL[iCell];
            strideBSTL[iCell]  = kSizesSTL[iCell] * nSizesSTL[iCell];
            strideCSTL[iCell]  = mSizesSTL[iCell] * nSizesSTL[iCell];
          }

        mSizes.copyFrom(mSizesSTL);
        nSizes.copyFrom(nSizesSTL);
        kSizes.copyFrom(kSizesSTL);
        ldaSizes.copyFrom(ldaSizesSTL);
        ldbSizes.copyFrom(ldbSizesSTL);
        ldcSizes.copyFrom(ldcSizesSTL);
        strideA.copyFrom(strideASTL);
        strideB.copyFrom(strideBSTL);
        strideC.copyFrom(strideCSTL);

        const ValueTypeOperator *A = xCellValues;

        const ValueTypeOperator *B = cellWiseC.data() + cellWiseCStartOffset;

        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand> *C =
          yCellValues;

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
          (ValueType)1.0,
          A,
          ldaSizes.data(),
          B,
          ldbSizes.data(),
          isCConjTransX ?
            (ValueType)0.0 :
            (ValueType)1.0, //--------- NOTE the 1.0 here ----------
          C,
          ldcSizes.data(),
          linAlgOpContext);
      }

      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      void
      computeCXCellWiseLocal(
        const utils::MemoryStorage<ValueTypeOperator, memorySpace> &cellWiseC,
        const ValueTypeOperand *                                    x,
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand> *  y,
        bool                                         isCConjTransX,
        const size_type                              numVecs,
        const size_type                              d_numLocallyOwnedCells,
        const std::vector<size_type> &               numCellXLocalIds,
        const std::vector<size_type> &               numCellYLocalIds,
        const size_type *                            cellLocalIdsStartPtrX,
        const size_type *                            cellLocalIdsStartPtrY,
        const size_type                              cellBlockSize,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        size_type cellLocalIdsOffsetX = 0;
        size_type cellLocalIdsOffsetY = 0;

        size_type maxXLocalIdsInCell =
          *std::max_element(numCellXLocalIds.begin(), numCellXLocalIds.end());

        size_type maxYLocalIdsInCell =
          *std::max_element(numCellYLocalIds.begin(), numCellYLocalIds.end());

        utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
          cellBlockSize * numVecs * maxXLocalIdsInCell, (ValueTypeOperand)0.);

        utils::MemoryStorage<linearAlgebra::blasLapack::
                               scalar_type<ValueTypeOperator, ValueTypeOperand>,
                             memorySpace>
          yCellValues(
            cellBlockSize * numVecs * maxYLocalIdsInCell,
            utils::Types<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>>::zero);

        for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
             cellStartId += cellBlockSize)
          {
            // This had to be done for Y cellval to 0 for Y = C*CX as C may have
            // 0 proj but addCellWiseDataToFieldData cannot sense it.
            // xCellValues.setValue((ValueTypeOperand)0.);
            if (isCConjTransX == false)
              yCellValues.setValue(
                utils::Types<linearAlgebra::blasLapack::scalar_type<
                  ValueTypeOperator,
                  ValueTypeOperand>>::zero);

            const size_type cellEndId =
              std::min(cellStartId + cellBlockSize, d_numLocallyOwnedCells);
            const size_type numCellsInBlock = cellEndId - cellStartId;

            std::vector<size_type> cellsInBlockNumXLocalIdsSTL(numCellsInBlock);
            std::copy(numCellXLocalIds.begin() + cellStartId,
                      numCellXLocalIds.begin() + cellEndId,
                      cellsInBlockNumXLocalIdsSTL.begin());

            std::vector<size_type> cellsInBlockNumYLocalIdsSTL(numCellsInBlock);
            std::copy(numCellYLocalIds.begin() + cellStartId,
                      numCellYLocalIds.begin() + cellEndId,
                      cellsInBlockNumYLocalIdsSTL.begin());

            utils::MemoryStorage<size_type, memorySpace>
              cellsInBlockNumXLocalIds(numCellsInBlock);
            cellsInBlockNumXLocalIds.copyFrom(cellsInBlockNumXLocalIdsSTL);

            utils::MemoryStorage<size_type, memorySpace>
              cellsInBlockNumYLocalIds(numCellsInBlock);
            cellsInBlockNumYLocalIds.copyFrom(cellsInBlockNumYLocalIdsSTL);

            // copy x to cell-wise data
            basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
              copyFieldToCellWiseData(x,
                                      numVecs,
                                      cellLocalIdsStartPtrX +
                                        cellLocalIdsOffsetX,
                                      cellsInBlockNumXLocalIds,
                                      xCellValues);

            cellWiseGEMM(std::make_pair(cellStartId, cellEndId),
                         cellWiseC,
                         isCConjTransX,
                         numVecs,
                         numCellXLocalIds,
                         numCellYLocalIds,
                         xCellValues.data(),
                         yCellValues.data(),
                         linAlgOpContext);

            basis::FECellWiseDataOperations<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>,
              memorySpace>::addCellWiseDataToFieldData(yCellValues,
                                                       numVecs,
                                                       cellLocalIdsStartPtrY +
                                                         cellLocalIdsOffsetY,
                                                       cellsInBlockNumYLocalIds,
                                                       y);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                cellLocalIdsOffsetX += cellsInBlockNumXLocalIdsSTL[iCell];
                cellLocalIdsOffsetY += cellsInBlockNumYLocalIdsSTL[iCell];
              }
          }
      }
    } // namespace AtomCenterNonLocalOpContextFEInternal

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
        const size_type                  maxCellBlock,
        const size_type                  maxFieldBlock,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &mpiComm)
      : d_feBasisManager(&feBasisManager)
      , d_maxCellBlock(maxCellBlock)
      , d_maxWaveFnBatch(maxFieldBlock)
      , d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_linAlgOpContext(linAlgOpContext)
      , d_fieldNameProjector("beta")
    {
      const std::string metadataNameCouplingConst = "dij";
      const std::string metadataNameNumProj       = "number_of_proj";
      const std::string metadataNamelMax          = "l_max";
      // Construct C_cell = \integral_\omega \beta_lp * Y_lm
      // * N_j

      // Create Partition for non-local projectors
      int rank;
      utils::mpi::MPICommRank(mpiComm, &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      int numProcs;
      utils::mpi::MPICommSize(mpiComm, &numProcs);

      if (dim != 3)
        utils::throwException(
          false, "Dimension should be 3 for Spherical Projector Dofs.");

      std::shared_ptr<
        const FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>
        feBasisDofHandler = std::dynamic_pointer_cast<
          const FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>(
          feBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        feBasisDofHandler.get() != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in AtomCenterNonLocalOpContextFE");

      std::shared_ptr<const TriangulationBase> triangulation =
        feBasisDofHandler->getTriangulation();

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto                                   cell = triangulation->beginLocal();
      auto                                   endc = triangulation->endLocal();

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
        "locallyOwnedCellsInTriangulation does not match to that in dofhandler in AtomCenterNonLocalOpContext()");

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
      std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition =
        std::make_shared<const AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                      minbound,
                                                      maxbound,
                                                      cellVerticesVector,
                                                      atomPartitionTolerance,
                                                      mpiComm);

      // Create projectorIdsPartition Object.
      d_projectorIdsPartition =
        std::make_shared<const EnrichmentIdsPartition<dim>>(
          atomSphericalDataContainer,
          atomIdsPartition,
          atomSymbolVec,
          atomCoordinatesVec,
          d_fieldNameProjector,
          minbound,
          maxbound,
          0,
          triangulation->getDomainVectors(),
          triangulation->getPeriodicFlags(),
          cellVerticesVector,
          mpiComm);

      d_overlappingProjectorIdsInCells =
        d_projectorIdsPartition->overlappingEnrichmentIdsInCells();

      std::pair<global_size_type, global_size_type> locOwnPair =
        d_projectorIdsPartition->locallyOwnedEnrichmentIds();

      std::vector<global_size_type> ghostVec =
        d_projectorIdsPartition->ghostEnrichmentIds();

      // C_cell is N_projectors_val(r, theta, phi) *
      // N_dofs which is got by contraction over a cell of quadpoints
      size_type cellWiseCSize   = 0;
      auto locallyOwnedCellIter = feBasisDofHandler->beginLocallyOwnedCells();
      cellIndex                 = 0;
      d_numProjsInCells.resize(feBasisDofHandler->nLocallyOwnedCells());

      for (; locallyOwnedCellIter != feBasisDofHandler->endLocallyOwnedCells();
           ++locallyOwnedCellIter)
        {
          d_numProjsInCells[cellIndex] =
            d_overlappingProjectorIdsInCells[cellIndex].size();
          size_type nDofsInCell = feBasisDofHandler->nCellDofs(cellIndex);
          cellWiseCSize += d_numProjsInCells[cellIndex] * nDofsInCell;
          cellIndex++;
        }
      d_cellWiseC.resize(cellWiseCSize);

      d_maxProjInCell =
        *std::max_element(d_numProjsInCells.begin(), d_numProjsInCells.end());
      d_totProjInProc =
        std::accumulate(d_numProjsInCells.begin(), d_numProjsInCells.end(), 0);

      cellIndex                                                    = 0;
      size_type                                 cumulativeDofxProj = 0;
      const quadrature::QuadratureRuleContainer quadratureRuleContainer =
        *feBasisDataStorage.getQuadratureRuleContainer();
      locallyOwnedCellIter = feBasisDofHandler->beginLocallyOwnedCells();
      for (; locallyOwnedCellIter != feBasisDofHandler->endLocallyOwnedCells();
           ++locallyOwnedCellIter)
        {
          size_type nQuadsInCell =
            quadratureRuleContainer.nCellQuadraturePoints(cellIndex);
          size_type numProjsInCell = d_numProjsInCells[cellIndex];
          size_type numDofsInCell  = feBasisDofHandler->nCellDofs(cellIndex);

          if (numProjsInCell > 0)
            {
              const std::vector<double> &cellJxW =
                quadratureRuleContainer.getCellJxW(cellIndex);

              std::vector<utils::Point> quadRealPointsVec =
                quadratureRuleContainer.getCellRealPoints(cellIndex);

              std::vector<double> projectorQuadStorageJxW =
                getProjectorValues(cellIndex, quadRealPointsVec);

              for (unsigned int iProj = 0; iProj < numProjsInCell; iProj++)
                {
                  for (unsigned int qPoint = 0; qPoint < nQuadsInCell; qPoint++)
                    {
                      // std::cout << quadRealPointsVec[qPoint][0]
                      // <<quadRealPointsVec[qPoint][1]
                      // <<quadRealPointsVec[qPoint][2]
                      // <<*(projectorQuadStorageJxW.data() + nQuadsInCell *
                      // iProj + qPoint) << "\n";
                      *(projectorQuadStorageJxW.data() + nQuadsInCell * iProj +
                        qPoint) *= cellJxW[qPoint];
                    }
                }

              utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
                basisData(numDofsInCell * nQuadsInCell);

              feBasisDataStorage.getBasisDataInCellRange(
                std::make_pair(cellIndex, cellIndex + 1), basisData);

              linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                              ValueTypeOperator,
                                              utils::MemorySpace::HOST>(
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::Op::Trans,
                linearAlgebra::blasLapack::Op::Trans,
                numProjsInCell,
                numDofsInCell,
                nQuadsInCell,
                (ValueTypeOperator)1.0,
                projectorQuadStorageJxW.data(),
                nQuadsInCell,
                basisData.data(),
                numDofsInCell,
                (ValueTypeOperator)0.0,
                d_cellWiseC.data() + cumulativeDofxProj,
                numProjsInCell,
                *linAlgOpContext);

              // //std::cout << cellIndex<<" -> ";
              // for(int iDof = 0 ; iDof < numDofsInCell ; iDof++)
              //   std::cout << *(d_cellWiseC.data() + cumulativeDofxProj +
              //   iDof) << "\n";
            }
          cumulativeDofxProj += numDofsInCell * numProjsInCell;
          cellIndex++;
        }

      // Create mpiPatternP2P for locOwned and ghost Projectors
      d_mpiPatternP2PProj =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          std::vector<std::pair<global_size_type, global_size_type>>{
            locOwnPair},
          ghostVec,
          mpiComm);

      // create the d_locallyOwnedCellLocalProjectorIds
      d_locallyOwnedCellLocalProjectorIds.resize(d_totProjInProc);
      size_type *ptr         = d_locallyOwnedCellLocalProjectorIds.data();
      d_numLocallyOwnedCells = feBasisDofHandler->nLocallyOwnedCells();
      size_type cumulativeProjectors = 0;
      for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; ++iCell)
        {
          const size_type numCellProjectors = d_numProjsInCells[iCell];
          for (size_type iProj = 0; iProj < numCellProjectors; ++iProj)
            {
              const global_size_type globalProjId =
                d_overlappingProjectorIdsInCells[iCell][iProj];
              *(ptr + cumulativeProjectors + iProj) =
                d_mpiPatternP2PProj->globalToLocal(globalProjId);
            }
          cumulativeProjectors += numCellProjectors;
        }

      // Initilize the d_CX
      d_CX =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          d_mpiPatternP2PProj, linAlgOpContext, d_maxWaveFnBatch);

      // Get the coupling matrix d_V = D_ij
      for (auto atomSymbol :
           d_projectorIdsPartition->getAtomSymbolsForLocalEnrichments())
        {
          utils::stringOps::strToInt(atomSphericalDataContainer->getMetadata(
                                       atomSymbol, metadataNameNumProj),
                                     d_atomSymbolToNumProjMap[atomSymbol]);

          bool convSuccess = utils::stringOps::splitStringToDoubles(
            atomSphericalDataContainer->getMetadata(atomSymbol,
                                                    metadataNameCouplingConst),
            d_atomSymbolToCouplingConstVecMap[atomSymbol],
            d_atomSymbolToNumProjMap[atomSymbol] *
              d_atomSymbolToNumProjMap[atomSymbol]);
          utils::throwException(
            convSuccess,
            "Error while converting Coupling Constant Vector to double vector in AtomCenterNonLocalOpContext");

          std::vector<std::vector<int>> qNumbers =
            d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                      d_fieldNameProjector);

          int lMax = 0;
          utils::stringOps::strToInt(atomSphericalDataContainer->getMetadata(
                                       atomSymbol, metadataNamelMax),
                                     lMax);

          // TODO: best way to get the num projectors for each l ?
          // assumption m is the fastest index
          std::vector<int> numProj(lMax + 1);
          for (int lId = 0; lId <= lMax; lId++)
            {
              int count = 0, mCount = 0;
              for (int i = 0; i < qNumbers.size(); i += mCount)
                {
                  int p = qNumbers[i][0], l = qNumbers[i][1];
                  if (l == lId)
                    {
                      count += 1;
                    }
                  mCount = 2 * l + 1;
                }
              numProj[lId] = count;
            }

          std::vector<int> betaIndexVec(qNumbers.size());
          for (int i = 0; i < qNumbers.size(); i += 1)
            {
              int p = qNumbers[i][0], l = qNumbers[i][1];
              int index = 0;
              for (int lId = 0; lId < l; lId++)
                {
                  index += numProj[lId];
                }
              betaIndexVec[i] = index + p;
            }
          d_atomSymbolToBetaIndexVecMap[atomSymbol] = betaIndexVec;
        }

      std::vector<ValueTypeOperator> V(d_CX->localSize(), 0.);
      d_V.resize(V.size());

      for (size_type iProjLocal = 0; iProjLocal < d_CX->localSize();
           iProjLocal++)
        {
          global_size_type iProjGlobal =
            d_mpiPatternP2PProj->localToGlobal(iProjLocal);

          basis::EnrichmentIdAttribute pIdAttr =
            d_projectorIdsPartition->getEnrichmentIdAttribute(iProjGlobal);

          size_type atomId  = pIdAttr.atomId;
          size_type localId = pIdAttr.localIdInAtom;

          std::string atomSymbol = atomSymbolVec[atomId];

          size_type index = d_atomSymbolToBetaIndexVecMap[atomSymbol][localId];

          // Coupling const is diagonal
          V[iProjLocal] =
            *(d_atomSymbolToCouplingConstVecMap[atomSymbol].data() +
              index * d_atomSymbolToNumProjMap[atomSymbol] + index);
        }
      d_V.copyFrom(V);

      // for (int i = 0 ; i < d_V.size() ; i++)
      //   std::cout << *(d_V.data() + i) << std::endl;

      d_numCellDofs.resize(d_numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; ++iCell)
        {
          d_numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);
        }

      size_type maxProjInCell =
        *std::max_element(d_numProjsInCells.begin(), d_numProjsInCells.end());
      d_CXCellValues.resize(maxCellBlock * maxFieldBlock * maxProjInCell,
                            ValueType());
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::vector<double>
    AtomCenterNonLocalOpContextFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      getProjectorValues(const size_type                          cellId,
                         const std::vector<dftefe::utils::Point> &points) const
    {
      // Assumption for each l,p pair the m values are consecutive
      std::vector<global_size_type> projIdVec =
        d_overlappingProjectorIdsInCells[cellId];
      unsigned int        numProjIdsInCell = projIdVec.size();
      unsigned int        numPoints        = points.size();
      std::vector<double> retValue(numPoints * numProjIdsInCell, 0),
        rVec(numPoints, 0), thetaVec(numPoints, 0), phiVec(numPoints, 0);
      std::vector<dftefe::utils::Point> x(numPoints, utils::Point(dim));
      DFTEFE_AssertWithMsg(!projIdVec.empty(),
                           "The requested cell does not have any proj ids.");
      unsigned int numProjIdsSkipped = 0;
      unsigned int l                 = 0;

      for (int iProj = 0; iProj < numProjIdsInCell; iProj += numProjIdsSkipped)
        {
          basis::EnrichmentIdAttribute pIdAttr =
            d_projectorIdsPartition->getEnrichmentIdAttribute(projIdVec[iProj]);

          size_type atomId  = pIdAttr.atomId;
          size_type localId = pIdAttr.localIdInAtom;

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
              d_atomSymbolVec[atomId], d_fieldNameProjector);

          auto quantumNoVec =
            d_atomSphericalDataContainer->getQNumbers(d_atomSymbolVec[atomId],
                                                      d_fieldNameProjector);

          l = quantumNoVec[localId][1];

          auto radialValue = sphericalDataVec[localId]->getRadialValue(rVec);

          // assumption m is the fastest index
          for (int mCount = 0; mCount < 2 * l + 1; mCount++)
            {
              auto angularValue = (sphericalDataVec[localId + mCount])
                                    ->getAngularValue(rVec, thetaVec, phiVec);

              linearAlgebra::blasLapack::hadamardProduct<
                ValueTypeOperator,
                ValueTypeOperator,
                utils::MemorySpace::HOST>(numPoints,
                                          radialValue.data(),
                                          angularValue.data(),
                                          retValue.data() +
                                            (iProj + mCount) * numPoints,
                                          *d_linAlgOpContext);
            }
          numProjIdsSkipped = (2 * l + 1);
        }
      return retValue;
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
      const size_type numVecs = X.getNumberComponents();

      if (d_CX->getNumberComponents() != numVecs)
        {
          d_CX = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2PProj, d_linAlgOpContext, numVecs);
        }

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
      d_CX->setValue(0.0);

      const size_type cellBlockSize =
        (d_maxCellBlock * d_maxWaveFnBatch) / numVecs;

      // Do d_CX = d_cellWiseC * X cellwise (field to cellwise data)

      auto itCellLocalIdsBeginX =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      auto itCellLocalIdsBeginY = d_locallyOwnedCellLocalProjectorIds.begin();

      AtomCenterNonLocalOpContextFEInternal::computeCXCellWiseLocal(
        d_cellWiseC,
        X.begin(),
        d_CX->begin(),
        true,
        numVecs,
        d_numLocallyOwnedCells,
        d_numCellDofs,
        d_numProjsInCells,
        itCellLocalIdsBeginX,
        itCellLocalIdsBeginY,
        cellBlockSize,
        *(X.getLinAlgOpContext()));

      // acumulate add, update ghost of d_CX

      d_CX->accumulateAddLocallyOwned();
      d_CX->updateGhostValues();

      // Do d_CX = d_V * d_CX
      size_type stride = 0;
      size_type m = 1, n = numVecs, k = d_CX->localSize();

      linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                        ValueTypeOperator,
                                                        memorySpace>(
        1,
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::ScalarOp::Identity,
        linearAlgebra::blasLapack::ScalarOp::Identity,
        &stride,
        &stride,
        &stride,
        &m,
        &n,
        &k,
        d_V.data(),
        d_CX->data(),
        d_CX->data(),
        *X.getLinAlgOpContext());

      // Do Y = d_cellwiseC * d_CX cellwise (field to cellwise data)

      itCellLocalIdsBeginX = d_locallyOwnedCellLocalProjectorIds.begin();

      itCellLocalIdsBeginY =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      // std::cout << "LocallyOwned : \n";
      // for(int i = 0 ; i < d_mpiPatternP2PProj->localOwnedSize() ; i++)
      // {
      //   //*(d_CX->data()+i) = 1.0;
      //   std::cout << d_mpiPatternP2PProj->localToGlobal(i) << " - " <<
      //   *(d_CX->data()+i) << "\t" ;
      // }
      // std::cout << "Ghost : \n";
      // for(int i = 0 ; i < d_mpiPatternP2PProj->localGhostSize() ; i++)
      // {
      //   //*(d_CX->data()+i) = 1.0;
      //   std::cout <<
      //   d_mpiPatternP2PProj->localToGlobal(i+d_mpiPatternP2PProj->localOwnedSize())
      //   << " - " << *(d_CX->data()+i+d_mpiPatternP2PProj->localOwnedSize())
      //   << "\t" ;
      // }

      AtomCenterNonLocalOpContextFEInternal::computeCXCellWiseLocal(
        d_cellWiseC,
        d_CX->begin(),
        Y.begin(),
        false,
        numVecs,
        d_numLocallyOwnedCells,
        d_numProjsInCells,
        d_numCellDofs,
        itCellLocalIdsBeginX,
        itCellLocalIdsBeginY,
        cellBlockSize,
        *(X.getLinAlgOpContext()));

      constraintsY.distributeChildToParent(Y, numVecs);
      Y.accumulateAddLocallyOwned();
      if (updateGhostY)
        Y.updateGhostValues();
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    AtomCenterNonLocalOpContextFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::reinitCX(size_type waveFuncBlockSize)
      const
    {
      if (d_CX->getNumberComponents() != waveFuncBlockSize)
        {
          d_CX = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2PProj, d_linAlgOpContext, waveFuncBlockSize);
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    AtomCenterNonLocalOpContextFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::setCXToZero() const
    {
      d_CX->setValue(0.0);
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
      dim>::applyCconjtransOnX(std::pair<size_type, size_type> cellRange,
                               const ValueTypeOperand *xCellValuesBegin) const
    {
      size_type numCellsInBlock = cellRange.second - cellRange.first;
      if (numCellsInBlock <=
          (d_maxCellBlock * d_maxWaveFnBatch) / d_CX->getNumberComponents())
        {
          AtomCenterNonLocalOpContextFEInternal::cellWiseGEMM(
            cellRange,
            d_cellWiseC,
            true,
            d_CX->getNumberComponents(),
            d_numCellDofs,
            d_numProjsInCells,
            xCellValuesBegin,
            d_CXCellValues.data(),
            *d_linAlgOpContext);

          size_type cellLocalIdsOffsetY = 0;
          for (size_type iCell = 0; iCell < cellRange.first; ++iCell)
            {
              cellLocalIdsOffsetY += d_numProjsInCells[iCell];
            }

          std::vector<size_type> cellsInBlockLocalIdsSTL(numCellsInBlock);
          std::copy(d_numProjsInCells.begin() + cellRange.first,
                    d_numProjsInCells.begin() + cellRange.second,
                    cellsInBlockLocalIdsSTL.begin());

          utils::MemoryStorage<size_type, memorySpace> cellsInBlockLocalIds(
            numCellsInBlock);
          cellsInBlockLocalIds.copyFrom(cellsInBlockLocalIdsSTL);

          basis::FECellWiseDataOperations<
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>,
            memorySpace>::
            addCellWiseDataToFieldData(
              d_CXCellValues,
              d_CX->getNumberComponents(),
              d_locallyOwnedCellLocalProjectorIds.begin() + cellLocalIdsOffsetY,
              cellsInBlockLocalIds,
              d_CX->data());
        }
      else
        {
          utils::throwException(
            false, "The cellRange has to be smaller than cellBlock max given.");
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    AtomCenterNonLocalOpContextFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::applyAllReduceOnCconjtransX() const
    {
      d_CX->accumulateAddLocallyOwned();
      d_CX->updateGhostValues();
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    AtomCenterNonLocalOpContextFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::applyVOnCconjtransX() const
    {
      // Do d_CX = d_V * d_CX
      size_type stride = 0;
      size_type m = 1, n = d_CX->getNumberComponents(), k = d_CX->localSize();

      linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                        ValueTypeOperator,
                                                        memorySpace>(
        1,
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::ScalarOp::Identity,
        linearAlgebra::blasLapack::ScalarOp::Identity,
        &stride,
        &stride,
        &stride,
        &m,
        &n,
        &k,
        d_V.data(),
        d_CX->data(),
        d_CX->data(),
        *d_linAlgOpContext);
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
      dim>::applyCOnVCconjtransX(std::pair<size_type, size_type> cellRange,
                                 ValueTypeOperand *yCellValuesBegin) const
    {
      size_type numCellsInBlock = cellRange.second - cellRange.first;
      if (numCellsInBlock <=
          (d_maxCellBlock * d_maxWaveFnBatch) / d_CX->getNumberComponents())
        {
          size_type cellLocalIdsOffsetX = 0;
          for (size_type iCell = 0; iCell < cellRange.first; ++iCell)
            {
              cellLocalIdsOffsetX += d_numProjsInCells[iCell];
            }

          std::vector<size_type> cellsInBlockLocalIdsSTL(numCellsInBlock);
          std::copy(d_numProjsInCells.begin() + cellRange.first,
                    d_numProjsInCells.begin() + cellRange.second,
                    cellsInBlockLocalIdsSTL.begin());

          utils::MemoryStorage<size_type, memorySpace> cellsInBlockLocalIds(
            numCellsInBlock);
          cellsInBlockLocalIds.copyFrom(cellsInBlockLocalIdsSTL);

          basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
            copyFieldToCellWiseData(
              d_CX->data(),
              d_CX->getNumberComponents(),
              d_locallyOwnedCellLocalProjectorIds.begin() + cellLocalIdsOffsetX,
              cellsInBlockLocalIds,
              d_CXCellValues);

          AtomCenterNonLocalOpContextFEInternal::cellWiseGEMM(
            cellRange,
            d_cellWiseC,
            false,
            d_CX->getNumberComponents(),
            d_numProjsInCells,
            d_numCellDofs,
            d_CXCellValues.data(),
            yCellValuesBegin,
            *d_linAlgOpContext);
        }
      else
        {
          utils::throwException(
            false, "The cellRange has to be smaller than cellBlock max given.");
        }
    }

  } // namespace basis
} // end of namespace dftefe
