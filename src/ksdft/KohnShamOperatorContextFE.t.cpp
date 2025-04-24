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

#include <basis/FECellWiseDataOperations.h>
#include <cstdlib>
#include <utils/ConditionalOStream.h>
#include <utils/MathFunctions.h>
#include <basis/EFEBasisDofHandler.h>

namespace dftefe
{
  namespace ksdft
  {
    namespace
    {
      template <typename ValueTypeOperator, utils::MemorySpace memorySpace>
      class HamiltonianComponentsOperations
      {
      public:
        static void
        addLocalComponent(
          utils::MemoryStorage<ValueTypeOperator, memorySpace>
            &localHamiltonianCumulative,
          std::variant<Hamiltonian<float, memorySpace> *,
                       Hamiltonian<double, memorySpace> *,
                       Hamiltonian<std::complex<float>, memorySpace> *,
                       Hamiltonian<std::complex<double>, memorySpace> *>
                                                       hamiltonianComponent,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          utils::throwException(
            false,
            "The valuetypes of the localHamiltonianCumulative in KohnShamOperatorContextFE can be double, float, complex float or complex double.");
        }
      };

      //--------float----------

      template <utils::MemorySpace memorySpace>
      class HamiltonianComponentsOperations<float, memorySpace>
      {
      public:
        static void
        addLocalComponent(
          utils::MemoryStorage<float, memorySpace> &localHamiltonianCumulative,
          std::variant<Hamiltonian<float, memorySpace> *,
                       Hamiltonian<double, memorySpace> *,
                       Hamiltonian<std::complex<float>, memorySpace> *,
                       Hamiltonian<std::complex<double>, memorySpace> *>
                                                       hamiltonianComponent,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          if (std::holds_alternative<Hamiltonian<float, memorySpace> *>(
                hamiltonianComponent))
            {
              const Hamiltonian<float, memorySpace> &b =
                *(std::get<Hamiltonian<float, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {
              utils::MemoryStorage<float, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::axpby<float, float, memorySpace>(
                localHamiltonianCumulative.size(),
                (float)1.0,
                localHamiltonianCumulative.data(),
                (float)1.0,
                temp.data(),
                localHamiltonianCumulative.data(),
                linAlgOpContext);
              }
            }
        }
      };

      //-----------double-------

      template <utils::MemorySpace memorySpace>
      class HamiltonianComponentsOperations<double, memorySpace>
      {
      public:
        static void
        addLocalComponent(
          utils::MemoryStorage<double, memorySpace> &localHamiltonianCumulative,
          std::variant<Hamiltonian<float, memorySpace> *,
                       Hamiltonian<double, memorySpace> *,
                       Hamiltonian<std::complex<float>, memorySpace> *,
                       Hamiltonian<std::complex<double>, memorySpace> *>
                                                       hamiltonianComponent,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          if (std::holds_alternative<Hamiltonian<double, memorySpace> *>(
                hamiltonianComponent))
            {
              const Hamiltonian<double, memorySpace> &b =
                *(std::get<Hamiltonian<double, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {
              utils::MemoryStorage<double, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::axpby<double, double, memorySpace>(
                localHamiltonianCumulative.size(),
                (double)1.0,
                localHamiltonianCumulative.data(),
                (double)1.0,
                temp.data(),
                localHamiltonianCumulative.data(),
                linAlgOpContext);
              }
            }
          else if (std::holds_alternative<Hamiltonian<float, memorySpace> *>(
                     hamiltonianComponent))
            {
              const Hamiltonian<float, memorySpace> &b =
                *(std::get<Hamiltonian<float, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {
              utils::MemoryStorage<float, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::axpby<double, float, memorySpace>(
                localHamiltonianCumulative.size(),
                (double)1.0,
                localHamiltonianCumulative.data(),
                (float)1.0,
                temp.data(),
                localHamiltonianCumulative.data(),
                linAlgOpContext);
              }
            }
          else
            {
              utils::throwException(
                false,
                "The valuetypes of the hamiltonian vector in KohnShamOperatorContextFE can be double, float, complex float or complex double.");
            }
        }
      };

      //----------complex float--------

      template <utils::MemorySpace memorySpace>
      class HamiltonianComponentsOperations<std::complex<float>, memorySpace>
      {
      public:
        static void
        addLocalComponent(
          utils::MemoryStorage<std::complex<float>, memorySpace>
            &localHamiltonianCumulative,
          std::variant<Hamiltonian<float, memorySpace> *,
                       Hamiltonian<double, memorySpace> *,
                       Hamiltonian<std::complex<float>, memorySpace> *,
                       Hamiltonian<std::complex<double>, memorySpace> *>
                                                       hamiltonianComponent,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          if (std::holds_alternative<Hamiltonian<double, memorySpace> *>(
                hamiltonianComponent))
            {
              const Hamiltonian<double, memorySpace> &b =
                *(std::get<Hamiltonian<double, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {
              utils::MemoryStorage<double, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<float>, double, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<float>)1.0,
                  localHamiltonianCumulative.data(),
                  (double)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else if (std::holds_alternative<Hamiltonian<float, memorySpace> *>(
                     hamiltonianComponent))
            {
              const Hamiltonian<float, memorySpace> &b =
                *(std::get<Hamiltonian<float, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {
              utils::MemoryStorage<float, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<float>, float, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<float>)1.0,
                  localHamiltonianCumulative.data(),
                  (float)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else if (std::holds_alternative<
                     Hamiltonian<std::complex<float>, memorySpace> *>(
                     hamiltonianComponent))
            {
              const Hamiltonian<std::complex<float>, memorySpace> &b =
                *(std::get<Hamiltonian<std::complex<float>, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {                  
              utils::MemoryStorage<std::complex<float>, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<float>, std::complex<float>, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<float>)1.0,
                  localHamiltonianCumulative.data(),
                  (std::complex<float>)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else
            {
              utils::throwException(
                false,
                "The valuetypes of the hamiltonian vector in KohnShamOperatorContextFE can be double, float, complex float or complex double.");
            }
        }
      };

      //--------------complex double---------

      template <utils::MemorySpace memorySpace>
      class HamiltonianComponentsOperations<std::complex<double>, memorySpace>
      {
      public:
        static void
        addLocalComponent(
          utils::MemoryStorage<std::complex<double>, memorySpace>
            &localHamiltonianCumulative,
          std::variant<Hamiltonian<float, memorySpace> *,
                       Hamiltonian<double, memorySpace> *,
                       Hamiltonian<std::complex<float>, memorySpace> *,
                       Hamiltonian<std::complex<double>, memorySpace> *>
                                                       hamiltonianComponent,
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
        {
          if (std::holds_alternative<Hamiltonian<double, memorySpace> *>(
                hamiltonianComponent))
            {
              const Hamiltonian<double, memorySpace> &b =
                *(std::get<Hamiltonian<double, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {                  
              utils::MemoryStorage<double, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<double>, double, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<double>)1.0,
                  localHamiltonianCumulative.data(),
                  (double)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else if (std::holds_alternative<Hamiltonian<float, memorySpace> *>(
                     hamiltonianComponent))
            {
              const Hamiltonian<float, memorySpace> &b =
                *(std::get<Hamiltonian<float, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {                  
              utils::MemoryStorage<float, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<double>, float, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<double>)1.0,
                  localHamiltonianCumulative.data(),
                  (float)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else if (std::holds_alternative<
                     Hamiltonian<std::complex<float>, memorySpace> *>(
                     hamiltonianComponent))
            {
              const Hamiltonian<std::complex<float>, memorySpace> &b =
                *(std::get<Hamiltonian<std::complex<float>, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {                  
              utils::MemoryStorage<std::complex<float>, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<double>, std::complex<float>, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<double>)1.0,
                  localHamiltonianCumulative.data(),
                  (std::complex<float>)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else if (std::holds_alternative<
                     Hamiltonian<std::complex<double>, memorySpace> *>(
                     hamiltonianComponent))
            {
              const Hamiltonian<std::complex<double>, memorySpace> &b =
                *(std::get<Hamiltonian<std::complex<double>, memorySpace> *>(
                  hamiltonianComponent));
              if(b.hasLocalComponent())
              {                 
              utils::MemoryStorage<std::complex<double>, memorySpace> temp(0);
              b.getLocal(temp);

              utils::throwException(
                temp.size() == localHamiltonianCumulative.size(),
                "size of hamiltonian does not match with number"
                " cumulative dofxdofs in locally owned cells in KohnShamOperatorContextFE");

              linearAlgebra::blasLapack::
                axpby<std::complex<double>, std::complex<double>, memorySpace>(
                  localHamiltonianCumulative.size(),
                  (std::complex<double>)1.0,
                  localHamiltonianCumulative.data(),
                  (std::complex<double>)1.0,
                  temp.data(),
                  localHamiltonianCumulative.data(),
                  linAlgOpContext);
              }
            }
          else
            {
              utils::throwException(
                false,
                "The valuetypes of the hamiltonian vector in KohnShamOperatorContextFE can be double, float, complex float or complex double.");
            }
        }
      };
    } // namespace

    namespace KohnShamOperatorContextFEInternal
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
            mSizesSTL[iCell]   = numVecs;
            nSizesSTL[iCell]   = cellsInBlockNumDoFs[iCell];
            kSizesSTL[iCell]   = cellsInBlockNumDoFs[iCell];
            ldaSizesSTL[iCell] = mSizesSTL[iCell];
            ldbSizesSTL[iCell] = kSizesSTL[iCell];
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

            // allocate memory for cell-wise data for x
            // utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
            //   cellsInBlockNumCumulativeDoFs * numVecs,
            //   utils::Types<linearAlgebra::blasLapack::scalar_type<
            //     ValueTypeOperator,
            //     ValueTypeOperand>>::zero);

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

            // allocate memory for cell-wise data for y
            // utils::MemoryStorage<
            //   linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
            //                                          ValueTypeOperand>,
            //   memorySpace>
            //   yCellValues(cellsInBlockNumCumulativeDoFs * numVecs,
            //               utils::Types<linearAlgebra::blasLapack::scalar_type<
            //                 ValueTypeOperator,
            //                 ValueTypeOperand>>::zero);

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

    } // end of namespace KohnShamOperatorContextFEInternal


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    KohnShamOperatorContextFE<ValueTypeOperator,
                              ValueTypeOperand,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      KohnShamOperatorContextFE(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeBasisData, memorySpace, dim>
            &                                        feBasisManager,
        const std::vector<HamiltonianPtrVariant>           &hamiltonianComponentsVec,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const size_type                              maxCellBlock,
        const size_type                              maxWaveFnBatch)
      : d_maxCellBlock(maxCellBlock)
      , d_maxWaveFnBatch(maxWaveFnBatch)
      , d_linAlgOpContext(linAlgOpContext)
    {
      reinit(feBasisManager, hamiltonianComponentsVec);
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KohnShamOperatorContextFE<
      ValueTypeOperator,
      ValueTypeOperand,
      ValueTypeBasisData,
      memorySpace,
      dim>::reinit(const basis::FEBasisManager<ValueTypeOperand,
                                               ValueTypeBasisData,
                                               memorySpace,
                                               dim> & feBasisManager,
                  const std::vector<HamiltonianPtrVariant> &hamiltonianComponentsVec)
    {
      d_hamiltonianComponentsVec = hamiltonianComponentsVec;
      
      const size_type numLocallyOwnedCells =
        feBasisManager.nLocallyOwnedCells();

      size_type cellWiseDataSize = 0;
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
        {
          size_type x = feBasisManager.nLocallyOwnedCellDofs(iCell);
          cellWiseDataSize += x * x;
        }

      d_feBasisManager = &feBasisManager;

      d_hamiltonianInAllCells.resize(cellWiseDataSize, (ValueTypeOperator)0);

      HamiltonianComponentsOperations<ValueTypeOperator, memorySpace> op;

      for (unsigned int i = 0; i < hamiltonianComponentsVec.size(); ++i)
        {
          op.addLocalComponent(d_hamiltonianInAllCells,
                          hamiltonianComponentsVec[i],
                          d_linAlgOpContext);
        }

      /**
        //------------------------ print Hamiltonian EE block---------------------------
        const basis::BasisDofHandler &basisDofHandler =
          feBasisManager.getBasisDofHandler();

        const basis::EFEBasisDofHandler<ValueTypeOperand,
                                ValueTypeOperator,
                                memorySpace,
                                dim> &efebasisDofHandler =
          dynamic_cast<const basis::EFEBasisDofHandler<ValueTypeOperand,
                                                ValueTypeOperator,
                                                memorySpace,
                                                dim> &>(basisDofHandler);

        const size_type numCellClassicalDofs =
      utils::mathFunctions::sizeTypePow( (efebasisDofHandler.getFEOrder(0) + 1),
      dim); const size_type nglobalEnrichmentIds =
      efebasisDofHandler.nGlobalEnrichmentNodes(); const size_type
      nglobalClassicalIds = (efebasisDofHandler.getGlobalRanges())[0].second;

        std::vector<ValueTypeOperator> hamEnrichmentBlockSTL(
          nglobalEnrichmentIds * nglobalEnrichmentIds, 0),
          hamEnrichmentBlockSTLTmp(nglobalEnrichmentIds *
                                              nglobalEnrichmentIds, 0);

        size_type cellId                     = 0;
        size_type cumulativeBasisDataInCells = 0;
        for (auto enrichmentVecInCell :
            efebasisDofHandler.getEnrichmentIdsPartition()
              ->overlappingEnrichmentIdsInCells())
          {
            size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
            for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
              {
                for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
                  {
                    *(hamEnrichmentBlockSTLTmp.data() +
                      enrichmentVecInCell[j] * nglobalEnrichmentIds +
                      enrichmentVecInCell[k]) +=
                      *(d_hamiltonianInAllCells.data() +
      cumulativeBasisDataInCells + (numCellClassicalDofs + nCellEnrichmentDofs)
      * (numCellClassicalDofs + j) + numCellClassicalDofs + k);
                  }
              }
            cumulativeBasisDataInCells += utils::mathFunctions::sizeTypePow(
              (nCellEnrichmentDofs + numCellClassicalDofs), 2);
            cellId += 1;
          }

        int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          hamEnrichmentBlockSTLTmp.data(),
          hamEnrichmentBlockSTL.data(),
          hamEnrichmentBlockSTLTmp.size(),
          utils::mpi::MPIDouble,
          utils::mpi::MPISum,
          d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        int rank;
        utils::mpi::MPICommRank(d_feBasisManager->getMPIPatternP2P()->mpiCommunicator(),
      &rank);

        utils::ConditionalOStream rootCout(std::cout);
        rootCout.setCondition(rank == 0);

        rootCout << "Hamiltonian Enrichment Block Matrix: "<<std::endl;
        for(size_type i = 0 ; i < nglobalEnrichmentIds ; i++)
        {
          rootCout << "[";
          for(size_type j = 0 ; j < nglobalEnrichmentIds ; j++)
          {
            rootCout << *(hamEnrichmentBlockSTL.data() + i*nglobalEnrichmentIds
      + j) << "\t";
          }
          rootCout << "]" <<std::endl;
        }

        //------------------------ print Hamiltonian EC block to
      file--------------------------- std::vector<ValueTypeOperator>
      hamECBlockSTL( (nglobalClassicalIds + nglobalEnrichmentIds) *
      nglobalEnrichmentIds, 0), hamECBlockSTLTmp((nglobalClassicalIds +
      nglobalEnrichmentIds) * nglobalEnrichmentIds, 0);


        cumulativeBasisDataInCells = 0;
        for (size_type iCell = 0; iCell <
      efebasisDofHandler.nLocallyOwnedCells() ; iCell++)
        {
          // get cell dof global ids
          std::vector<global_size_type> cellGlobalNodeIds(0);
          efebasisDofHandler.getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

          std::vector<global_size_type> enrichmentVecInCell =
      efebasisDofHandler.getEnrichmentIdsPartition()
                  ->overlappingEnrichmentIdsInCells()[iCell];

          size_type nCellEnrichmentDofs = enrichmentVecInCell.size();

          // loop over nodes of a cell
          for ( size_type iNode = 0 ; iNode < numCellClassicalDofs +
      nCellEnrichmentDofs; iNode++)
          {
            for ( size_type jNode = 0 ; jNode < nCellEnrichmentDofs ; jNode++)
            {
              *(hamECBlockSTLTmp.data() +
                cellGlobalNodeIds[iNode] * nglobalEnrichmentIds +
                  enrichmentVecInCell[jNode]) +=
                *(d_hamiltonianInAllCells.data() + cumulativeBasisDataInCells +
                  (numCellClassicalDofs + nCellEnrichmentDofs) * iNode + jNode +
      numCellClassicalDofs);
            }
          }
          cumulativeBasisDataInCells +=
      utils::mathFunctions::sizeTypePow((cellGlobalNodeIds.size()),2);
        }

        err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          hamECBlockSTLTmp.data(),
          hamECBlockSTL.data(),
          hamECBlockSTLTmp.size(),
          utils::mpi::MPIDouble,
          utils::mpi::MPISum,
          d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
        mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        std::ofstream myfile;
        myfile.open("wings");
        utils::ConditionalOStream rootCout1(myfile);
        rootCout1.setCondition(rank == 0);

        for(size_type i = 0 ; i < nglobalClassicalIds + nglobalEnrichmentIds ;
      i++)
        {
          rootCout1 << i <<": [";
          for(size_type j = 0 ; j < nglobalEnrichmentIds ; j++)
          {
            rootCout1 << *(hamECBlockSTL.data() + i*nglobalEnrichmentIds + j) <<
      "\t";
          }
          rootCout1 << "]" <<std::endl;
        }
        myfile.close();
      **/
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KohnShamOperatorContextFE<
      ValueTypeOperator,
      ValueTypeOperand,
      ValueTypeBasisData,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
                  bool updateGhostX,
                  bool updateGhostY) const
    {
      const size_type numLocallyOwnedCells =
        d_feBasisManager->nLocallyOwnedCells();
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);
        }

      auto itCellLocalIdsBeginX =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      auto itCellLocalIdsBeginY =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      const size_type numVecs = X.getNumberComponents();

      // get handle to constraints
      const basis::ConstraintsLocal<ValueType, memorySpace> &constraintsX =
        d_feBasisManager->getConstraints();

      const basis::ConstraintsLocal<ValueType, memorySpace> &constraintsY =
        d_feBasisManager->getConstraints();

      if (updateGhostX)
        X.updateGhostValues();
      // update the child nodes based on the parent nodes
      constraintsX.distributeParentToChild(X, numVecs);

      const size_type cellBlockSize =
        (d_maxCellBlock * d_maxWaveFnBatch) / numVecs;
      Y.setValue(0.0);

      //
      // perform Ax on the local part of A and x
      // (A = discrete Laplace operator)
      //
      KohnShamOperatorContextFEInternal::computeAxCellWiseLocal(
        d_hamiltonianInAllCells,
        X.begin(),
        Y.begin(),
        numVecs,
        numLocallyOwnedCells,
        numCellDofs,
        itCellLocalIdsBeginX,
        itCellLocalIdsBeginY,
        cellBlockSize,
        *(X.getLinAlgOpContext()));

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      constraintsY.distributeChildToParent(Y, numVecs);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      Y.accumulateAddLocallyOwned();
      if (updateGhostY)
        Y.updateGhostValues();

      HamiltonianComponentsOperations<ValueTypeOperator, memorySpace> op;

      for (unsigned int i = 0; i < d_hamiltonianComponentsVec.size(); ++i)
      {
        const Hamiltonian<ValueTypeOperand, memorySpace> &b =
        *(std::get<Hamiltonian<ValueTypeOperand, memorySpace> *>(
          d_hamiltonianComponentsVec[i]));
        if(b.hasNonLocalComponent())
        {
          linearAlgebra::MultiVector<ValueTypeOperator, memorySpace> temp(X, (ValueTypeOperator)0);
          b.applyNonLocal(X, temp, updateGhostX, updateGhostY);
          linearAlgebra::blasLapack::axpby<ValueTypeOperator, ValueTypeOperator, memorySpace>(
            Y.getNumberComponents()*Y.localSize(),
            (ValueTypeOperator)1.0,
            Y.data(),
            (ValueTypeOperator)1.0,
            temp.data(),
            Y.data(),
            *X.getLinAlgOpContext());
        }
      }
    }

  } // end of namespace ksdft
} // end of namespace dftefe
