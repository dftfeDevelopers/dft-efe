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
 * @author Bikash Kanungo
 */
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <basis/FECellWiseDataOperations.h>

namespace dftefe
{
  namespace physics
  {
    namespace LaplaceOperatorContextFEInternal
    {
      template <utils::MemorySpace memorySpace>
      void
      storeSizes(utils::MemoryStorage<size_type, memorySpace> &mSizes,
                 utils::MemoryStorage<size_type, memorySpace> &nSizes,
                 utils::MemoryStorage<size_type, memorySpace> &kSizes,
                 utils::MemoryStorage<size_type, memorySpace> &ldaSizes,
                 utils::MemoryStorage<size_type, memorySpace> &ldbSizes,
                 utils::MemoryStorage<size_type, memorySpace> &ldcSizes,
                 utils::MemoryStorage<size_type, memorySpace> &strideA,
                 utils::MemoryStorage<size_type, memorySpace> &strideB,
                 utils::MemoryStorage<size_type, memorySpace> &strideC,
                 const std::vector<size_type> &cellsInBlockNumDoFs,
                 const size_type               numVecs)
      {
        const size_type        numCellsInBlock = cellsInBlockNumDoFs.size();
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
            mSizesSTL[iCell]   = cellsInBlockNumDoFs[iCell];
            nSizesSTL[iCell]   = numVecs;
            kSizesSTL[iCell]   = cellsInBlockNumDoFs[iCell];
            ldaSizesSTL[iCell] = mSizesSTL[iCell];
            ldbSizesSTL[iCell] = nSizesSTL[iCell];
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
      }

      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      void
      computeAxCellWiseLocal(
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &                     gradNiGradNjInAllCells,
        const ValueTypeOperand *x,
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand> *
            y,
        const size_type                              numVecs,
        const size_type                              numLocallyOwnedCells,
        const std::vector<size_type> &               numCellDofs,
        const size_type *                            cellLocalIdsStartPtr,
        const size_type                              cellBlockSize,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        //
        // Perform ye = Ae * xe, where
        // Ae is the discrete Laplace operator for the e-th cell.
        // That is, \f$Ae_ij=\int_{\Omega_e} \nabla N_i \cdot \nabla N_j
        // d\textbf{r} $\f,
        // (\f$Ae_ij$\f is the integral of the dot product of the gradient of
        // i-th and j-th basis function in the e-th cell.
        //
        // xe, ye are the part of the input (x) and output vector (y),
        // respectively, belonging to e-th cell.
        //

        //
        // For better performance, we evaluate ye for multiple cells at a time
        //

        //
        // @note: The Ae and xe matrix are stored in row major-format.
        // Hence, we specify the layout to be row major
        //
        linearAlgebra::blasLapack::Layout layout =
          linearAlgebra::blasLapack::Layout::RowMajor;

        size_type AStartOffset       = 0;
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

            utils::MemoryStorage<size_type, memorySpace> cellsInBlockNumDoFs(
              numCellsInBlock);
            cellsInBlockNumDoFs.copyFrom(cellsInBlockNumDoFsSTL);

            // allocate memory for cell-wise data for x
            utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
              cellsInBlockNumCumulativeDoFs * numVecs,
              utils::Types<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>>::zero);

            // copy x to cell-wise data
            basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
              copyFieldToCellWiseData(x,
                                      numVecs,
                                      cellLocalIdsStartPtr + cellLocalIdsOffset,
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

            LaplaceOperatorContextFEInternal::storeSizes(mSizes,
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

            // allocate memory for cell-wise data for y
            utils::MemoryStorage<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>, memorySpace> yCellValues(
              cellsInBlockNumCumulativeDoFs * numVecs,
              utils::Types<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>>::zero);

              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand> alpha = 1.0;
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand> beta  = 0.0;

            ValueTypeOperator *A = gradNiGradNjInAllCells.data() + AStartOffset;
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
              A,
              ldaSizes.data(),
              xCellValues.data(),
              ldbSizes.data(),
              beta,
              yCellValues.data(),
              ldcSizes.data(),
              linAlgOpContext);

            basis::FECellWiseDataOperations<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>, memorySpace>::
              addCellWiseDataToFieldData(yCellValues,
                                         numVecs,
                                         cellLocalIdsStartPtr +
                                           cellLocalIdsOffset,
                                         cellsInBlockNumDoFs,
                                         y);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                AStartOffset +=
                  cellsInBlockNumDoFsSTL[iCell] * cellsInBlockNumDoFsSTL[iCell];
                cellLocalIdsOffset += cellsInBlockNumDoFsSTL[iCell];
              }
          }
      }

    } // end of namespace LaplaceOperatorContextFEInternal


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    LaplaceOperatorContextFE<ValueTypeOperator,
                             ValueTypeOperand,
                             memorySpace,
                             dim>::
      LaplaceOperatorContextFE(
        const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>
          &feBasisHandler,
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &                             feBasisDataStorage,
        const std::string               constraintsName,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 maxCellTimesNumVecs)
      : d_feBasisHandler(&feBasisHandler)
      , d_feBasisDataStorage(&feBasisDataStorage)
      , d_constraintsName(constraintsName)
      , d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_maxCellTimesNumVecs(maxCellTimesNumVecs)
    {}

    // template <typename ValueTypeOperator,
    //           typename ValueTypeOperand,
    //           utils::MemorySpace memorySpace,
    //           size_type          dim>
    // void
    // LaplaceOperatorContextFE<
    //   ValueTypeOperator,
    //   ValueTypeOperand,
    //   memorySpace,
    //   dim>::apply(const linearAlgebra::Vector<ValueTypeOperand, memorySpace> &x,
    //               linearAlgebra::Vector<ValueType, memorySpace> &y) const
    // {
    //   const size_type numLocallyOwnedCells =
    //     d_feBasisHandler->nLocallyOwnedCells();
    //   std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
    //   for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
    //     numCellDofs[iCell] = feBasisHandler.nLocallyOwnedCellDofs(iCell);

    //   auto itCellLocalIdsBegin =
    //     d_feBasisHandler->locallyOwnedCellLocalDofIdsBegin(d_constraintsName);

    //   const size_type numVecs = 1;
      
    //   // get handle to constraints
    //   const Constraints<ValueType, memorySpace> &constraints =
    //     d_feBasisHandler->getConstraints(d_constraintsName);

    //   // update the child nodes based on the parent nodes
    //   constraints.distributeParentToChild(x);


    //   // access cell-wise discrete Laplace operator
    //   auto gradNiGradNjInAllCells =
    //     d_feBasisDataStorage->getBasisGradNiGradNjInAllCells(
    //       d_quadratureRuleAttributes);

    //   const size_type cellBlockSize = d_maxCellTimesNumVecs / numVecs;
    //   linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
    //     *(x.getLinAlgOpContext());

    //   //
    //   // perform Ax on the local part of A and x
    //   // (A = discrete Laplace operator)
    //   //
    //   LaplaceOperatorContextFEInternal::computeAxCellWiseLocal(
    //     gradNiGradNjAllCells,
    //     x.begin(),
    //     y.begin(),
    //     numVecs,
    //     numLocallyOwnedCells,
    //     numCellDofs,
    //     itCellLocalIdsBegin,
    //     cellBlockSize,
    //     linAlgOpContext);

    //   // Function to add the values to the local node from its corresponding
    //   // ghost nodes from other processors.
    //   y.accumulateAddLocallyOwned();

    //   // function to do a static condensation to send the constraint nodes to
    //   // its parent nodes
    //   constraints.distributeChildToParent(y);
    // }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    LaplaceOperatorContextFE<ValueTypeOperator,
                             ValueTypeOperand,
                             memorySpace,
                             dim>::
      apply(const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
            linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const
    {
      const size_type numLocallyOwnedCells =
        d_feBasisHandler->nLocallyOwnedCells();
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisHandler->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisHandler->locallyOwnedCellLocalDofIdsBegin(d_constraintsName);

      const size_type numVecs = X.numberVectors();
      
      // get handle to constraints
      const basis::Constraints<ValueType, memorySpace> &constraints =
        d_feBasisHandler->getConstraints(d_constraintsName);

      // update the child nodes based on the parent nodes
      constraints.distributeParentToChild(X);

      // access cell-wise discrete Laplace operator
      auto gradNiGradNjInAllCells =
        d_feBasisDataStorage->getBasisGradNiGradNjInAllCells(
          d_quadratureRuleAttributes);

      const size_type cellBlockSize = d_maxCellTimesNumVecs / numVecs;
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
        *(X.getLinAlgOpContext());

      //
      // perform Ax on the local part of A and x
      // (A = discrete Laplace operator)
      //
      LaplaceOperatorContextFEInternal::computeAxCellWiseLocal(
        gradNiGradNjInAllCells,
        X.begin(),
        Y.begin(),
        numVecs,
        numLocallyOwnedCells,
        numCellDofs,
        itCellLocalIdsBegin,
        cellBlockSize,
        linAlgOpContext);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      Y.accumulateAddLocallyOwned();

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      constraints.distributeChildToParent(Y);
    }

  } // end of namespace physics
} // end of namespace dftefe