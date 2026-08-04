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
    namespace CFEOverlapOperatorContextInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &feBasisDataStorage,
        std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
          &                                          basisOverlap,
        std::vector<size_type> &                     cellStartIdsBasisOverlap,
        std::vector<size_type> &                     dofsInCellVec,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          feBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            feBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          feBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in CFEOverlapOperatorContext");

        const size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        const size_type cellId  = 0;
        const size_type feOrder = feBDH->getFEOrder(cellId);

        // NOTE: cellId 0 passed as we assume only H refined in this function
        const size_type dofsPerCell = feBDH->nCellDofs(cellId);

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        basisOverlap = std::make_shared<
          utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          numLocallyOwnedCells * dofsPerCell * dofsPerCell);
        basisOverlapTmp.resize(numLocallyOwnedCells * dofsPerCell * dofsPerCell,
                               ValueTypeOperator(0));

        auto locallyOwnedCellIter = feBDH->beginLocallyOwnedCells();
        // auto      basisOverlapTmpIter  = basisOverlapTmp.begin();
        size_type cellIndex = 0;

        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCells = feBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeQuadPoints = 0, cumulativeDofQuadPointsOffset = 0;
        bool      isConstantDofsAndQuadPointsInCell = false;
        quadrature::QuadratureFamily quadFamily =
          feBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily == quadrature::QuadratureFamily::GAUSS ||
             quadFamily == quadrature::QuadratureFamily::GLL ||
             quadFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED) &&
            !feBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCell = true;

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsHost(basisDataInAllCells.size());

        basisDataInAllCellsHost.copyFrom(basisDataInAllCells);

        for (; locallyOwnedCellIter != feBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellIndex] = dofsPerCell;
            size_type nQuadPointInCell =
              feBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValues =
              feBasisDataStorage.getQuadratureRuleContainer()->getCellJxW(
                cellIndex);

            std::vector<ValueTypeOperator> JxWxNCellConj(dofsPerCell *
                                                         nQuadPointInCell);

            size_type stride = 0;
            size_type m = 1, n = dofsPerCell, k = nQuadPointInCell;

            linearAlgebra::blasLapack::scaleStridedVarBatched<
              ValueTypeOperator,
              ValueTypeOperator,
              utils::MemorySpace::HOST>(
              1,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Conj,
              &stride,
              &stride,
              &stride,
              &m,
              &n,
              &k,
              cellJxWValues.data(),
              basisDataInAllCellsHost.data() + cumulativeDofQuadPointsOffset,
              JxWxNCellConj.data(),
              *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);

            linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                            ValueTypeOperand,
                                            utils::MemorySpace::HOST>(
              'N',
              'T',
              dofsPerCell,
              dofsPerCell,
              nQuadPointInCell,
              (ValueTypeOperand)1.0,
              JxWxNCellConj.data(),
              dofsPerCell,
              basisDataInAllCellsHost.data() + cumulativeDofQuadPointsOffset,
              dofsPerCell,
              (ValueTypeOperand)0.0,
              basisOverlapTmp.data() + cumulativeBasisOverlapId,
              dofsPerCell,
              *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);

            // const ValueTypeOperator *cumulativeDofQuadPoints =
            //   basisDataInAllCells.data() + cumulativeDofQuadPointsOffset;

            // for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
            //   {
            //     for (size_type jNode = 0; jNode < dofsPerCell; jNode++)
            //       {
            //         *basisOverlapTmpIter = 0.0;
            //         for (size_type qPoint = 0; qPoint < nQuadPointInCell;
            //              qPoint++)
            //           {
            //             *basisOverlapTmpIter +=
            //               *(cumulativeDofQuadPoints + dofsPerCell * qPoint +
            //                 iNode
            //                 /*nQuadPointInCell * iNode + qPoint*/) *
            //               *(cumulativeDofQuadPoints + dofsPerCell * qPoint +
            //                 jNode
            //                 /*nQuadPointInCell * jNode + qPoint*/) *
            //               cellJxWValues[qPoint];
            //           }
            //         basisOverlapTmpIter++;
            //       }
            //   }

            cellStartIdsBasisOverlap[cellIndex] = cumulativeBasisOverlapId;
            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cumulativeQuadPoints += nQuadPointInCell;
            if (!isConstantDofsAndQuadPointsInCell)
              cumulativeDofQuadPointsOffset += nQuadPointInCell * dofsPerCell;
            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap->data(), basisOverlapTmp.data());
      }

      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      void
      computeAxCellWiseLocal(
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &                     basisOverlapInAllCells,
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
        //
        // Perform ye = Ae * xe, where
        // Ae is the discrete Overlap operator for the e-th cell.
        // That is, \f$Ae_ij=\int_{\Omega_e}  N_i \cdot N_j
        // d\textbf{r} $\f,
        // (\f$Ae_ij$\f is the integral of the dot product of the
        // i-th and j-th basis function in the e-th cell.
        //
        // xe, ye are the part of the input (x) and output vector (y),
        // respectively, belonging to e-th cell.
        //

        //
        // For better performance, we evaluate ye for multiple cells at a time
        //

        size_type BStartOffset       = 0;
        size_type cellLocalIdsOffset = 0;
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

            // utils::MemoryStorage<size_type, memorySpace> cellsInBlockNumDoFs(
            //   numCellsInBlock);
            // cellsInBlockNumDoFs.copyFrom(cellsInBlockNumDoFsSTL);

            // allocate memory for cell-wise data for x
            utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
              cellsInBlockNumCumulativeDoFs * numVecs,
              utils::Types<linearAlgebra::blasLapack::scalar_type<
                ValueTypeOperator,
                ValueTypeOperand>>::zero);

            // copy x to cell-wise data
            basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
              copyFieldToCellWiseData(x,
                                      numVecs,
                                      cellLocalIdsStartPtrX +
                                        cellLocalIdsOffset,
                                      // cellsInBlockNumDoFs,
                                      cellsInBlockNumCumulativeDoFs,
                                      xCellValues,
                                      linAlgOpContext);

            std::vector<char> transA(numCellsInBlock, 'N');
            std::vector<char> transB(numCellsInBlock, 'N');

            std::vector<size_type> mSizes(numCellsInBlock, 0);
            std::vector<size_type> nSizes(numCellsInBlock, 0);
            std::vector<size_type> kSizes(numCellsInBlock, 0);
            std::vector<size_type> ldaSizes(numCellsInBlock, 0);
            std::vector<size_type> ldbSizes(numCellsInBlock, 0);
            std::vector<size_type> ldcSizes(numCellsInBlock, 0);
            std::vector<size_type> strideA(numCellsInBlock, 0);
            std::vector<size_type> strideB(numCellsInBlock, 0);
            std::vector<size_type> strideC(numCellsInBlock, 0);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = numVecs;
                nSizes[iCell]   = cellsInBlockNumDoFsSTL[iCell];
                kSizes[iCell]   = cellsInBlockNumDoFsSTL[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = kSizes[iCell];
                ldcSizes[iCell] = mSizes[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = mSizes[iCell] * nSizes[iCell];
              }

            // allocate memory for cell-wise data for y
            utils::MemoryStorage<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>,
              memorySpace>
              yCellValues(cellsInBlockNumCumulativeDoFs * numVecs,
                          utils::Types<linearAlgebra::blasLapack::scalar_type<
                            ValueTypeOperator,
                            ValueTypeOperand>>::zero);

            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>
              alpha = 1.0;
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>
              beta = 0.0;

            const ValueTypeOperator *B =
              basisOverlapInAllCells.data() + BStartOffset;
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand> *C =
              yCellValues.begin();
            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperator,
                                                             ValueTypeOperand,
                                                             memorySpace>(
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
              memorySpace>::
              addCellWiseDataToFieldData(yCellValues,
                                         numVecs,
                                         cellLocalIdsStartPtrY +
                                           cellLocalIdsOffset,
                                         // cellsInBlockNumDoFs,
                                         cellsInBlockNumCumulativeDoFs,
                                         y,
                                         linAlgOpContext);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                BStartOffset +=
                  cellsInBlockNumDoFsSTL[iCell] * cellsInBlockNumDoFsSTL[iCell];
                cellLocalIdsOffset += cellsInBlockNumDoFsSTL[iCell];
              }
          }
      }

    } // end of namespace CFEOverlapOperatorContextInternal


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    CFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::
      CFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             feBasisDataStorage,
        const size_type maxCellBlock,
        const size_type maxFieldBlock,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_feBasisManager(&feBasisManager)
      , d_maxCellBlock(maxCellBlock)
      , d_maxFieldBlock(maxFieldBlock)
      , d_cellStartIdsBasisOverlap(0)
      , d_isMassLumping(false)
    {
      CFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(feBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             *linAlgOpContext);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    CFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::
      CFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &feBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_feBasisManager(&feBasisManager)
      , d_cellStartIdsBasisOverlap(0)
      , d_maxCellBlock(0)
      , d_maxFieldBlock(0)
      , d_isMassLumping(true)
    {
      utils::throwException(
        feBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      CFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(feBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             *linAlgOpContext);

      d_diagonal =
        std::make_shared<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
          d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      const size_type numLocallyOwnedCells =
        d_feBasisManager->nLocallyOwnedCells();
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);

      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

      const size_type numCumulativeDofsCells =
        std::accumulate(locallyOwnedCellsNumDoFsSTL.begin(),
                        locallyOwnedCellsNumDoFsSTL.end(),
                        0);

      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(d_basisOverlap->data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           numCumulativeDofsCells,
                                           d_diagonal->data(),
                                           *linAlgOpContext);

      d_feBasisManager->getConstraints().distributeChildToParent(*d_diagonal,
                                                                 1);

      d_diagonal->accumulateAddLocallyOwned();

      d_diagonal->updateGhostValues();

      d_feBasisManager->getConstraints().setConstrainedNodesToZero(*d_diagonal,
                                                                   1);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEOverlapOperatorContext<
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
      if (d_isMassLumping)
        {
          updateGhostX = false;
          updateGhostY = false;
          const basis::ConstraintsLocal<
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>,
            memorySpace> &constraints = d_feBasisManager->getConstraints();

          const size_type numVecs = X.getNumberComponents();

          if (updateGhostX)
            X.updateGhostValues();
          // update the child nodes based on the parent nodes
          constraints.distributeParentToChild(X, numVecs);

          Y.setValue(0.0);

          linearAlgebra::blasLapack::khatriRaoProduct(
            linearAlgebra::blasLapack::Layout::ColMajor,
            1,
            X.getNumberComponents(),
            d_diagonal->localSize(),
            d_diagonal->data(),
            X.data(),
            Y.data(),
            *(d_diagonal->getLinAlgOpContext()));

          // function to do a static condensation to send the constraint nodes
          // to its parent nodes
          constraints.distributeChildToParent(Y, numVecs);

          if (updateGhostY)
            Y.updateGhostValues();
        }
      else
        {
          // get handle to constraints
          const basis::ConstraintsLocal<
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>,
            memorySpace> &constraints = d_feBasisManager->getConstraints();

          const size_type numVecs = X.getNumberComponents();

          if (updateGhostX)
            X.updateGhostValues();
          // update the child nodes based on the parent nodes
          constraints.distributeParentToChild(X, numVecs);

          Y.setValue(0.0);

          const size_type numLocallyOwnedCells =
            d_feBasisManager->nLocallyOwnedCells();
          std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
          for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
            numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

          auto itCellLocalIdsBegin =
            d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

          // access cell-wise discrete Overlap operator
          const utils::MemoryStorage<ValueTypeOperator, memorySpace>
            &basisOverlapInAllCells = *d_basisOverlap;

          const size_type cellBlockSize =
            (d_maxCellBlock * d_maxFieldBlock) / numVecs;

          //
          // perform Ax on the local part of A and x
          // (A = discrete Overlap operator)
          //
          CFEOverlapOperatorContextInternal::computeAxCellWiseLocal(
            basisOverlapInAllCells,
            X.begin(),
            Y.begin(),
            numVecs,
            numLocallyOwnedCells,
            numCellDofs,
            itCellLocalIdsBegin,
            itCellLocalIdsBegin,
            cellBlockSize,
            *(X.getLinAlgOpContext()));

          // function to do a static condensation to send the constraint nodes
          // to its parent nodes
          constraints.distributeChildToParent(Y, numVecs);

          // Function to add the values to the local node from its corresponding
          // ghost nodes from other processors.
          Y.accumulateAddLocallyOwned();
          if (updateGhostY)
            Y.updateGhostValues();
        }
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
    CFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::getBasisOverlapInAllCells() const
    {
      return *(d_basisOverlap);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    utils::MemoryStorage<ValueTypeOperator, memorySpace>
    CFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::getBasisOverlapInCell(const size_type
                                                            cellId) const
    {
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
                      basisOverlapStorage = d_basisOverlap;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      utils::MemoryStorage<ValueTypeOperator, memorySpace> returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId]);
      return returnValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    utils::MemoryStorage<ValueTypeOperator, memorySpace>
    CFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::getBasisOverlap(const size_type cellId,
                                                    const size_type basisId1,
                                                    const size_type basisId2)
      const
    {
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
        basisOverlapStorage = d_basisOverlap;
      utils::MemoryStorage<ValueTypeOperator, memorySpace> returnValue(1);
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId] +
          basisId1 * d_dofsInCell[cellId] + basisId2);
      return returnValue;
    }

  } // namespace basis
} // end of namespace dftefe
